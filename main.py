import os
import threading
from flask import Flask, Response, render_template, jsonify, request, send_from_directory, abort
from camera import (MotionCamera, capture_snapshot,
                    load_roi, save_roi, load_loiter_roi, save_loiter_roi,
                    load_gate_roi, save_gate_roi, load_gate_ref, load_gate_ref_open,
                    MOTION_THRESHOLD, LOITER_TIME, RECORDINGS_DIR,
                    LOITER_ROI_FILE, GATE_ROI_FILE, GATE_REF_FILE, GATE_REF_OPEN_FILE,
                    NIGHT_CHANNEL_DIFF, _sinric_start)
import cv2

app = Flask(__name__)
cam = MotionCamera(roi=load_roi())
cam.set_loiter_roi(load_loiter_roi())
cam.set_gate_roi(load_gate_roi())
cam.set_gate_ref(load_gate_ref())
cam.set_gate_ref_open(load_gate_ref_open())


def gen_frames():
    while True:
        frame = cam.get_frame_jpg()
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(motion=cam.is_motion(), loitering=cam.loitering, roi=cam.roi,
                   show_roi=cam.show_roi, gate_open=cam.gate_open, gate_configured=cam.gate_roi is not None,
                   loiter_alerts=cam.loiter_alerts)


@app.route("/toggle-roi", methods=["POST"])
def toggle_roi():
    cam.show_roi = not cam.show_roi
    return jsonify(ok=True, show_roi=cam.show_roi)


@app.route("/toggle-loiter-alerts", methods=["POST"])
def toggle_loiter_alerts():
    cam.loiter_alerts = not cam.loiter_alerts
    return jsonify(ok=True, loiter_alerts=cam.loiter_alerts)


@app.route("/debug")
def debug():
    bri, ch_diff = cam.brightness()
    return jsonify(
        motion=cam.is_motion(),
        loitering=cam.loitering,
        last_max_area=float(cam.last_max_area),
        threshold=MOTION_THRESHOLD,
        loiter_time=LOITER_TIME,
        roi=cam.roi,
        gate_open=cam.gate_open,
        gate_score=float(cam.last_gate_score),
        night_mode=bool(cam._night_mode),
        effective_threshold=int(cam._effective_threshold),
        brightness=bri,
        channel_diff=ch_diff,
        night_channel_diff_threshold=NIGHT_CHANNEL_DIFF,
    )


@app.route("/test-alexa", methods=["POST"])
def test_alexa():
    from camera import _sinric_sensor, SINRIC_DEVICE_ID
    if not _sinric_sensor or not SINRIC_DEVICE_ID:
        return jsonify(ok=False, error="Sinric não conectado — aguarde ou verifique o .env")
    import time as _time
    def _fire():
        cam._trigger_sinric("open")
        _time.sleep(5)
        cam._trigger_sinric("closed")
    threading.Thread(target=_fire, daemon=True).start()
    return jsonify(ok=True, msg="Sensor aberto — fechará em 5s")


@app.route("/setup")
def setup():
    return render_template("setup.html")


@app.route("/snapshot")
def snapshot():
    data = capture_snapshot()
    if data is None:
        return "Câmera indisponível", 503
    return Response(data, mimetype="image/jpeg")


@app.route("/save-roi", methods=["POST"])
def save_roi_route():
    body = request.get_json()
    roi = {k: int(body[k]) for k in ("x", "y", "w", "h")}
    save_roi(roi)
    cam.set_roi(roi)
    return jsonify(ok=True, roi=roi)


@app.route("/clear-roi", methods=["POST"])
def clear_roi_route():
    from camera import ROI_FILE
    if os.path.exists(ROI_FILE):
        os.remove(ROI_FILE)
    cam.set_roi(None)
    return jsonify(ok=True)


@app.route("/save-loiter-roi", methods=["POST"])
def save_loiter_roi_route():
    body = request.get_json()
    roi = {k: int(body[k]) for k in ("x", "y", "w", "h")}
    save_loiter_roi(roi)
    cam.set_loiter_roi(roi)
    return jsonify(ok=True, roi=roi)


@app.route("/clear-loiter-roi", methods=["POST"])
def clear_loiter_roi_route():
    if os.path.exists(LOITER_ROI_FILE):
        os.remove(LOITER_ROI_FILE)
    cam.set_loiter_roi(None)
    return jsonify(ok=True)


@app.route("/save-gate-roi", methods=["POST"])
def save_gate_roi_route():
    body = request.get_json()
    roi = {k: int(body[k]) for k in ("x", "y", "w", "h")}
    save_gate_roi(roi)
    cam.set_gate_roi(roi)
    return jsonify(ok=True, roi=roi)


@app.route("/clear-gate-roi", methods=["POST"])
def clear_gate_roi_route():
    for f in (GATE_ROI_FILE, GATE_REF_FILE, GATE_REF_OPEN_FILE):
        if os.path.exists(f):
            os.remove(f)
    cam.set_gate_roi(None)
    cam.set_gate_ref(None)
    cam.set_gate_ref_open(None)
    return jsonify(ok=True)


def _save_gate_ref_frame(dest_file, setter):
    frame = cam.get_latest_frame()
    if frame is None:
        return jsonify(ok=False, error="Sem frame disponível")
    roi = cam.gate_roi
    if roi is None:
        return jsonify(ok=False, error="Defina a área do portão primeiro")
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    cv2.imwrite(dest_file, gray)
    setter(gray)
    return jsonify(ok=True)


@app.route("/save-gate-ref", methods=["POST"])
def save_gate_ref_route():
    return _save_gate_ref_frame(GATE_REF_FILE, cam.set_gate_ref)


@app.route("/save-gate-ref-open", methods=["POST"])
def save_gate_ref_open_route():
    return _save_gate_ref_frame(GATE_REF_OPEN_FILE, cam.set_gate_ref_open)


def _parse_recording_dt(name):
    try:
        from datetime import datetime as dt
        stem = name.replace(".mp4", "")           # motion_20260512_205031
        parts = stem.split("_")                   # ['motion', '20260512', '205031']
        d = dt.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
        return d.strftime("%d/%m/%Y"), d.strftime("%H:%M:%S")
    except Exception:
        return "", ""


@app.route("/recordings")
def recordings_page():
    files = []
    if os.path.exists(RECORDINGS_DIR):
        for name in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
            if name.endswith(".mp4"):
                path = os.path.join(RECORDINGS_DIR, name)
                size = os.path.getsize(path)
                date, time_ = _parse_recording_dt(name)
                files.append({"name": name, "size": size, "date": date, "time": time_})
    return render_template("recordings.html", files=files)


@app.route("/recordings/<path:filename>")
def serve_recording(filename):
    if not filename.endswith(".mp4"):
        abort(404)
    return send_from_directory(os.path.abspath(RECORDINGS_DIR), filename)


@app.route("/recordings/<path:filename>/delete", methods=["POST"])
def delete_recording(filename):
    if not filename.endswith(".mp4"):
        abort(404)
    path = os.path.join(RECORDINGS_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
    return jsonify(ok=True)


def telegram_polling():
    import requests, time
    from camera import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    offset = 0
    while True:
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=35,
            )
            for update in r.json().get("result", []):
                offset = update["update_id"] + 1
                text = update.get("message", {}).get("text", "")
                chat_id = update.get("message", {}).get("chat", {}).get("id")
                if str(chat_id) != TELEGRAM_CHAT_ID:
                    continue
                if text.strip() == "/foto":
                    frame = cam.get_frame_jpg()
                    if frame:
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                            data={"chat_id": chat_id},
                            files={"photo": ("foto.jpg", frame, "image/jpeg")},
                            timeout=10,
                        )
                elif text.strip() == "/video":
                    threading.Thread(
                        target=cam._send_telegram_video, args=(chat_id,), daemon=True
                    ).start()
        except Exception as e:
            print(f"Telegram polling erro: {e}")
            time.sleep(5)


if __name__ == "__main__":
    t = threading.Thread(target=cam.run, daemon=True)
    t.start()
    threading.Thread(target=telegram_polling, daemon=True).start()
    threading.Thread(target=_sinric_start, daemon=True).start()
    if cam.roi is None:
        print("Nenhuma ROI configurada — acesse http://localhost:5000/setup para definir a área monitorada.")
    print("Acesse: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
