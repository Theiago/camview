import os
import threading
from flask import Flask, Response, render_template, jsonify, request, send_from_directory, abort
from camera import MotionCamera, capture_snapshot, load_roi, save_roi, MOTION_THRESHOLD, LOITER_TIME, RECORDINGS_DIR

app = Flask(__name__)
cam = MotionCamera(roi=load_roi())


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
    return jsonify(motion=cam.is_motion(), loitering=cam.loitering, roi=cam.roi)


@app.route("/debug")
def debug():
    return jsonify(
        motion=cam.is_motion(),
        loitering=cam.loitering,
        last_max_area=cam.last_max_area,
        threshold=MOTION_THRESHOLD,
        loiter_time=LOITER_TIME,
        roi=cam.roi,
    )


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


@app.route("/recordings")
def recordings_page():
    files = []
    if os.path.exists(RECORDINGS_DIR):
        for name in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
            if name.endswith(".mp4"):
                path = os.path.join(RECORDINGS_DIR, name)
                size = os.path.getsize(path)
                files.append({"name": name, "size": size})
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
                if text.strip() == "/foto" and str(chat_id) == TELEGRAM_CHAT_ID:
                    frame = cam.get_frame_jpg()
                    if frame:
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                            data={"chat_id": chat_id},
                            files={"photo": ("foto.jpg", frame, "image/jpeg")},
                            timeout=10,
                        )
        except Exception as e:
            print(f"Telegram polling erro: {e}")
            time.sleep(5)


if __name__ == "__main__":
    t = threading.Thread(target=cam.run, daemon=True)
    t.start()
    threading.Thread(target=telegram_polling, daemon=True).start()
    if cam.roi is None:
        print("Nenhuma ROI configurada — acesse http://localhost:5000/setup para definir a área monitorada.")
    print("Acesse: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
