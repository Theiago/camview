import asyncio
import cv2
import threading
import time
import os
import json
import tempfile
import requests
import subprocess
import numpy as np
from datetime import datetime
from collections import deque
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")
SINRIC_APP_KEY    = os.getenv("SINRIC_APP_KEY")
SINRIC_APP_SECRET = os.getenv("SINRIC_APP_SECRET")
SINRIC_API_KEY    = os.getenv("SINRIC_API_KEY")
SINRIC_DEVICE_ID  = os.getenv("SINRIC_DEVICE_ID")

RTSP_URL = os.getenv("RTSP_URL", "")
RECORDINGS_DIR  = "recordings"
ROI_FILE        = "roi.json"
LOITER_ROI_FILE = "loiter_roi.json"
GATE_ROI_FILE        = "gate_roi.json"
GATE_REF_FILE        = "gate_ref.jpg"
GATE_REF_OPEN_FILE   = "gate_ref_open.jpg"
GATE_DIFF_THRESHOLD  = 20   # usado apenas quando só uma referência está salva
MOTION_THRESHOLD = 100       # área mínima de contorno (pixels²) para disparar
MOTION_CONFIRM_TIME = 0      # segundos contínuos de foreground para confirmar movimento
MOTION_COOLDOWN = 10         # segundos sem presença antes de resetar
LOITER_TIME = 15             # segundos parado para disparar alerta de permanência
NIGHT_CHANNEL_DIFF = 8       # diferença média entre canais RGB abaixo da qual a imagem é IR (modo noturno)
NIGHT_VAR_THRESHOLD = 25     # varThreshold do MOG2 no modo noturno
NIGHT_MOTION_THRESHOLD = 40  # área mínima de contorno no modo noturno
PRE_BUFFER_SECONDS = 3
FPS_RECORDING = 10
VIDEO_BUFFER_SECONDS = 60  # segundos mantidos em RAM para o comando /video

_sinric_loop: asyncio.AbstractEventLoop | None = None
_sinric_sensor = None


def _sinric_start():
    global _sinric_loop, _sinric_sensor
    if not (SINRIC_APP_KEY and SINRIC_APP_SECRET and SINRIC_DEVICE_ID):
        return
    try:
        from sinricpro.core.sinric_pro import SinricPro
        from sinricpro.core.types import SinricProConfig
        from sinricpro.devices.sinric_pro_contact_sensor import SinricProContactSensor

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _sinric_loop = loop

        sinric = SinricPro()
        sensor = SinricProContactSensor(SINRIC_DEVICE_ID)
        sinric.add(sensor)
        _sinric_sensor = sensor

        async def _run():
            await sinric.begin(SinricProConfig(app_key=SINRIC_APP_KEY, app_secret=SINRIC_APP_SECRET))
            await asyncio.Event().wait()

        loop.run_until_complete(_run())
    except Exception as e:
        print(f"Sinric Pro erro de conexão: {e}", flush=True)


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


def load_loiter_roi():
    if not os.path.exists(LOITER_ROI_FILE):
        return None
    with open(LOITER_ROI_FILE) as f:
        return json.load(f)


def save_loiter_roi(data: dict):
    with open(LOITER_ROI_FILE, "w") as f:
        json.dump(data, f)


def load_gate_roi():
    if not os.path.exists(GATE_ROI_FILE):
        return None
    with open(GATE_ROI_FILE) as f:
        return json.load(f)


def save_gate_roi(data: dict):
    with open(GATE_ROI_FILE, "w") as f:
        json.dump(data, f)


def load_gate_ref():
    if not os.path.exists(GATE_REF_FILE):
        return None
    return cv2.imread(GATE_REF_FILE, cv2.IMREAD_GRAYSCALE)


def load_gate_ref_open():
    if not os.path.exists(GATE_REF_OPEN_FILE):
        return None
    return cv2.imread(GATE_REF_OPEN_FILE, cv2.IMREAD_GRAYSCALE)


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
        self.video_buffer = deque(maxlen=VIDEO_BUFFER_SECONDS * FPS_RECORDING)
        self.running = False
        self.reconnect_delay = 5
        self.roi = roi
        self.loiter_roi = None
        self.gate_roi = None
        self.gate_ref = None
        self.gate_ref_open = None
        self.gate_open = False
        self.last_gate_score = 0.0
        self.show_roi = True
        self.loiter_alerts = True
        self.last_max_area = 0
        self._night_mode = False
        self._effective_threshold = MOTION_THRESHOLD
        self._yolo = YOLO("yolo11n.pt")
        self._yolo_counter = 0
        self._person_present = False
        self._last_detections = []
        self._last_detection_time = 0.0
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=60, detectShadows=False
        )
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

    def set_roi(self, roi):
        with self.lock:
            self.roi = roi

    def set_loiter_roi(self, roi):
        with self.lock:
            self.loiter_roi = roi

    def set_gate_roi(self, roi):
        with self.lock:
            self.gate_roi = roi

    def set_gate_ref(self, ref):
        with self.lock:
            self.gate_ref = ref

    def set_gate_ref_open(self, ref):
        with self.lock:
            self.gate_ref_open = ref

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def _check_gate(self, frame):
        with self.lock:
            gate_roi    = self.gate_roi
            ref_closed  = self.gate_ref
            ref_open    = self.gate_ref_open
        if gate_roi is None or (ref_closed is None and ref_open is None):
            return
        x, y, w, h = gate_roi["x"], gate_roi["y"], gate_roi["w"], gate_roi["h"]
        crop = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

        def _diff(ref):
            r = ref if crop.shape == ref.shape else cv2.resize(ref, (crop.shape[1], crop.shape[0]))
            return float(cv2.absdiff(crop, r).mean())

        if ref_closed is not None and ref_open is not None:
            sc, so = _diff(ref_closed), _diff(ref_open)
            self.last_gate_score = round(sc - so, 2)  # positivo = mais parecido com fechado
            self.gate_open = bool(so < sc)
        elif ref_closed is not None:
            sc = _diff(ref_closed)
            self.last_gate_score = round(sc, 2)
            self.gate_open = bool(sc > GATE_DIFF_THRESHOLD)
        else:
            so = _diff(ref_open)
            self.last_gate_score = round(so, 2)
            self.gate_open = bool(so < GATE_DIFF_THRESHOLD)

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
        if self._yolo_counter % 8 != 0:
            return self._last_detections

        results = self._yolo(frame, classes=self._YOLO_CLASSES, verbose=False, imgsz=320, conf=0.15)
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

    def _person_in_loiter_roi(self) -> bool:
        if self.loiter_roi is None:
            return True  # sem loiter_roi definido, qualquer presença conta
        persons = [d for d in self._last_detections if d[4] == 0]
        lx, ly, lw, lh = self.loiter_roi["x"], self.loiter_roi["y"], self.loiter_roi["w"], self.loiter_roi["h"]
        for x1, y1, x2, y2, *_ in persons:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if lx <= cx <= lx + lw and ly <= cy <= ly + lh:
                return True
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

    def _check_night_mode(self, frame):
        # câmeras IR: em modo noturno R≈G≈B (monocromático); de dia há diferença entre canais
        b, g, r = cv2.split(frame)
        channel_diff = float((cv2.absdiff(r, g).mean() + cv2.absdiff(r, b).mean()) / 2)
        is_night = bool(channel_diff < NIGHT_CHANNEL_DIFF)
        if is_night != self._night_mode:
            self._night_mode = is_night
            vt = NIGHT_VAR_THRESHOLD if is_night else 60
            self._effective_threshold = NIGHT_MOTION_THRESHOLD if is_night else MOTION_THRESHOLD
            self.bg_sub.setVarThreshold(vt)
            print(f"[{datetime.now()}] Modo {'noturno' if is_night else 'diurno'} "
                  f"(diff_canais={channel_diff:.1f}, varThr={vt}, motThr={self._effective_threshold})")

    def _detect_foreground(self, frame, freeze_bg: bool):
        crop = self._crop_frame(frame)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        lr = 0.0 if freeze_bg else -1.0
        mask = self.bg_sub.apply(enhanced, learningRate=lr)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.last_max_area = 0
            return False
        self.last_max_area = max(cv2.contourArea(c) for c in contours)
        return self.last_max_area > self._effective_threshold

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

    def _trigger_sinric(self, state: str):
        if _sinric_sensor is None or _sinric_loop is None:
            print("Sinric: não conectado", flush=True)
            return
        try:
            detected = state == "open"
            future = asyncio.run_coroutine_threadsafe(
                _sinric_sensor.send_contact_event(detected), _sinric_loop
            )
            result = future.result(timeout=5)
            print(f"[{datetime.now()}] Sinric: sensor {state} → {result}", flush=True)
        except Exception as e:
            print(f"Sinric erro: {e}", flush=True)

    def brightness(self):
        with self.lock:
            f = self.latest_frame
        if f is None:
            return 0.0, 0.0
        b, g, r = cv2.split(f)
        bri = round(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()), 1)
        diff = round(float((cv2.absdiff(r, g).mean() + cv2.absdiff(r, b).mean()) / 2), 1)
        return bri, diff

    def get_video_buffer(self):
        with self.lock:
            return list(self.video_buffer)

    def _send_telegram_video(self, chat_id):
        frames = self.get_video_buffer()
        if not frames:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    data={"chat_id": chat_id, "text": "Buffer vazio, aguarde alguns segundos."},
                    timeout=10,
                )
            except Exception:
                pass
            return

        tmp_raw = tempfile.mktemp(suffix="_raw.mp4")
        tmp_out = tempfile.mktemp(suffix=".mp4")
        try:
            first = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
            h, w = first.shape[:2]
            writer = cv2.VideoWriter(tmp_raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS_RECORDING, (w, h))
            for jpg in frames:
                f = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if f is not None:
                    writer.write(f)
            writer.release()

            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_raw,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "28",
                 "-movflags", "faststart", tmp_out],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            duration = len(frames) // FPS_RECORDING
            with open(tmp_out, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo",
                    data={"chat_id": chat_id, "caption": f"Últimos {duration}s"},
                    files={"video": ("clip.mp4", f, "video/mp4")},
                    timeout=120,
                )
        except Exception as e:
            print(f"Telegram video erro: {e}")
        finally:
            for p in (tmp_raw, tmp_out):
                if os.path.exists(p):
                    os.remove(p)

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
                self._check_night_mode(frame)
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
                        if not self.loitering and elapsed >= LOITER_TIME and self._person_in_loiter_roi():
                            self.loitering = True
                            print(f"[{datetime.now()}] PERMANÊNCIA no portão ({elapsed:.0f}s)")
                            if self.loiter_alerts:
                                threading.Thread(target=self._send_telegram, args=(frame.copy(),), daemon=True).start()
                                threading.Thread(target=self._trigger_sinric, args=("open",), daemon=True).start()
                else:
                    if not self.motion_detected:
                        self.candidate_start = 0.0
                    if self.motion_detected and (now - self.last_motion_time) > MOTION_COOLDOWN:
                        was_loitering = self.loitering
                        self.motion_detected = False
                        self.loitering = False
                        self.presence_start = 0.0
                        if was_loitering and self.loiter_alerts:
                            threading.Thread(target=self._trigger_sinric, args=("closed",), daemon=True).start()

                self._check_gate(frame)

                display = frame.copy()
                self._draw_detections(display)

                with self.lock:
                    roi = self.roi
                    loiter_roi = self.loiter_roi
                    gate_roi = self.gate_roi
                    show_roi = self.show_roi

                if show_roi and roi:
                    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                    color = (0, 0, 255) if self.motion_detected else (0, 255, 0)
                    cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

                if show_roi and loiter_roi:
                    lx, ly, lw, lh = loiter_roi["x"], loiter_roi["y"], loiter_roi["w"], loiter_roi["h"]
                    lcolor = (0, 140, 255) if self.loitering else (0, 255, 255)
                    cv2.rectangle(display, (lx, ly), (lx+lw, ly+lh), lcolor, 2)
                    cv2.putText(display, "PERMANENCIA", (lx, ly - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, lcolor, 1)

                if show_roi and gate_roi:
                    gx, gy, gw, gh = gate_roi["x"], gate_roi["y"], gate_roi["w"], gate_roi["h"]
                    gcolor = (0, 0, 255) if self.gate_open else (200, 200, 200)
                    cv2.rectangle(display, (gx, gy), (gx+gw, gy+gh), gcolor, 2)
                    glabel = "PORTAO ABERTO" if self.gate_open else "PORTAO FECHADO"
                    cv2.putText(display, glabel, (gx, gy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, gcolor, 1)

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
                    jpg = self._encode_jpg(display)
                    self.latest_frame_jpg = jpg
                    self.video_buffer.append(jpg)

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
