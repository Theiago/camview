import cv2
import threading
import time
import os
import json
import requests
import subprocess
from datetime import datetime
from collections import deque
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

RTSP_URL = os.getenv("RTSP_URL", "rtsp://192.168.0.200:554/h264/ch3/sub/av_stream")
RECORDINGS_DIR = "recordings"
ROI_FILE = "roi.json"
MOTION_THRESHOLD = 400       # área mínima de contorno (pixels²) para disparar
MOTION_CONFIRM_TIME = 2      # segundos contínuos de foreground para confirmar movimento
MOTION_COOLDOWN = 10         # segundos sem presença antes de resetar
LOITER_TIME = 10             # segundos parado para disparar alerta de permanência
PRE_BUFFER_SECONDS = 3
FPS_RECORDING = 10


def capture_snapshot():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def load_roi():
    if not os.path.exists(ROI_FILE):
        return None
    with open(ROI_FILE) as f:
        return json.load(f)


def save_roi(data: dict):
    with open(ROI_FILE, "w") as f:
        json.dump(data, f)


class MotionCamera:
    def __init__(self, roi=None):
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_jpg = None
        self.motion_detected = False
        self.last_motion_time = 0
        self.candidate_start = 0.0  # quando o foreground candidato começou
        self.presence_start = 0.0   # quando a presença confirmada começou
        self.loitering = False
        self.recording = False
        self.writer = None
        self._current_recording_path = None
        self.frame_buffer = deque(maxlen=PRE_BUFFER_SECONDS * FPS_RECORDING)
        self.running = False
        self.reconnect_delay = 5
        self.roi = roi
        self.last_max_area = 0
        self._yolo = YOLO("yolo11n.pt")
        self._yolo_counter = 0
        self._person_present = False
        self._last_detections = []
        self._last_detection_time = 0.0
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=60, detectShadows=False
        )
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

    def set_roi(self, roi):
        with self.lock:
            self.roi = roi

    def connect(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()

    def _crop_frame(self, frame):
        if self.roi is None:
            return frame
        x, y, w, h = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
        return frame[y:y+h, x:x+w]

    # classes YOLO: 0=pessoa, 2=carro, 3=moto
    _YOLO_CLASSES  = [0, 2, 3]
    _CLASS_LABEL   = {0: "pessoa", 2: "carro", 3: "moto"}
    _CLASS_COLOR   = {0: (0, 255, 255), 2: (255, 180, 0), 3: (255, 100, 180)}

    def _run_yolo(self, frame):
        """Roda YOLO a cada 5 frames; retorna lista de (x1,y1,x2,y2,cls,conf)."""
        self._yolo_counter += 1
        if self._yolo_counter % 15 != 0:
            return self._last_detections

        results = self._yolo(frame, classes=self._YOLO_CLASSES, verbose=False, imgsz=320)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self._last_detections = []
            return []

        dets = []
        for box, cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            dets.append((box[0], box[1], box[2], box[3], int(cls), float(conf)))
        self._last_detections = dets
        self._last_detection_time = time.time()
        return dets

    def _has_person(self, frame) -> bool:
        self._run_yolo(frame)
        persons = [d for d in self._last_detections if d[4] == 0]
        if not persons:
            self._person_present = False
            return False
        if self.roi is None:
            self._person_present = True
            return True
        rx, ry, rw, rh = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
        for x1, y1, x2, y2, *_ in persons:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
                self._person_present = True
                return True
        self._person_present = False
        return False

    def _draw_detections(self, display):
        if time.time() - self._last_detection_time > 2.0:
            self._last_detections = []
        for x1, y1, x2, y2, cls, conf in self._last_detections:
            color = self._CLASS_COLOR[cls]
            label = f"{self._CLASS_LABEL[cls]} {conf:.0%}"
            cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(display, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _detect_foreground(self, frame, freeze_bg: bool):
        crop = self._crop_frame(frame)
        lr = 0.0 if freeze_bg else -1.0
        mask = self.bg_sub.apply(crop, learningRate=lr)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.last_max_area = 0
            return False
        self.last_max_area = max(cv2.contourArea(c) for c in contours)
        return self.last_max_area > MOTION_THRESHOLD

    def _start_recording(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RECORDINGS_DIR, f"motion_{ts}.mp4")
        h, w = self.latest_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, FPS_RECORDING, (w, h))
        for f in self.frame_buffer:
            self.writer.write(f)
        self.recording = True
        self._current_recording_path = path
        print(f"[{datetime.now()}] Gravação iniciada: {path}")

    def _stop_recording(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.recording = False
        if self._current_recording_path:
            threading.Thread(target=self._fix_mp4, args=(self._current_recording_path,), daemon=True).start()
        print(f"[{datetime.now()}] Gravação encerrada")

    def _fix_mp4(self, path):
        tmp = path.replace(".mp4", "_tmp.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-movflags", "faststart", tmp],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            os.replace(tmp, path)
            print(f"[{datetime.now()}] Vídeo convertido: {path}")
        except Exception as e:
            print(f"ffmpeg erro: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)

    def _send_telegram(self, frame):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": "Permanência detectada no portão"},
                files={"photo": ("alert.jpg", buf.tobytes(), "image/jpeg")},
                timeout=10,
            )
        except Exception as e:
            print(f"Telegram erro: {e}")

    def _encode_jpg(self, frame):
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    def run(self):
        self.running = True

        while self.running:
            if not self.connect():
                print(f"Sem conexão com câmera. Tentando novamente em {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                continue

            print("Câmera conectada.")
            # reinicia o modelo de fundo ao reconectar
            self.bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=60, detectShadows=False
            )

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Frame perdido — reconectando...")
                    break

                now = time.time()
                foreground = self._detect_foreground(frame, freeze_bg=self.motion_detected)

                if foreground and self._has_person(frame):
                    if self.candidate_start == 0.0:
                        self.candidate_start = now
                    self.last_motion_time = now

                    if not self.motion_detected and (now - self.candidate_start) >= MOTION_CONFIRM_TIME:
                        self.motion_detected = True
                        self.presence_start = self.candidate_start
                        self.loitering = False
                        print(f"[{datetime.now()}] Movimento confirmado! área={self.last_max_area:.0f}px²")

                    if self.motion_detected:
                        elapsed = now - self.presence_start
                        if not self.loitering and elapsed >= LOITER_TIME:
                            self.loitering = True
                            print(f"[{datetime.now()}] PERMANÊNCIA no portão ({elapsed:.0f}s)")
                            threading.Thread(target=self._send_telegram, args=(frame.copy(),), daemon=True).start()
                else:
                    if not self.motion_detected:
                        self.candidate_start = 0.0
                    if self.motion_detected and (now - self.last_motion_time) > MOTION_COOLDOWN:
                        self.motion_detected = False
                        self.loitering = False
                        self.presence_start = 0.0

                display = frame.copy()
                self._draw_detections(display)

                with self.lock:
                    roi = self.roi

                if roi:
                    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                    color = (0, 0, 255) if self.motion_detected else (0, 255, 0)
                    cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

                if self.loitering:
                    elapsed = int(now - self.presence_start)
                    cv2.putText(display, f"PERMANENCIA NO PORTAO ({elapsed}s)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
                elif self.motion_detected:
                    cv2.putText(display, "MOVIMENTO DETECTADO", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                ts_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                cv2.putText(display, ts_str, (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                with self.lock:
                    self.latest_frame = frame.copy()
                    self.latest_frame_jpg = self._encode_jpg(display)

                self.frame_buffer.append(frame.copy())
                if self.motion_detected:
                    if not self.recording:
                        self._start_recording()
                    if self.writer:
                        self.writer.write(frame)
                elif self.recording:
                    self._stop_recording()

            self.cap.release()

        if self.recording:
            self._stop_recording()

    def get_frame_jpg(self):
        with self.lock:
            return self.latest_frame_jpg

    def is_motion(self):
        return self.motion_detected

    def stop(self):
        self.running = False
