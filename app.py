import os
import cv2
import time
import threading
import torch
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

# Optimization
torch.set_num_threads(2)

app = Flask(__name__)

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/weed_best.pt")
CAMERA_SOURCES = {
    "camA": os.environ.get("CAM_A", "/dev/v4l/by-id/usb-046d_0825_D103FA40-video-index0"),
    "camB": os.environ.get("CAM_B", "/dev/v4l/by-id/usb-046d_0825_41B63E10-video-index0"),
}

CONFIG = {
    "active_camera": os.environ.get("ACTIVE_CAMERA", "camA"),
    "run_detection": os.environ.get("RUN_DETECTION", "1") == "1",
    "confidence": float(os.environ.get("CONF_THRES", "0.12")),
    "img_size": int(os.environ.get("IMG_SIZE", "640")),
    "min_area": int(os.environ.get("MIN_AREA", "1200")),
    "max_area": int(os.environ.get("MAX_AREA", "0")),  # 0 disables upper area filter
    "top_k": int(os.environ.get("TOP_K", "20")),  # 0 disables top-k cap
    "max_det": int(os.environ.get("MAX_DET", "120")),
    "jpeg_quality": int(os.environ.get("JPEG_QUALITY", "80")),
    "log_interval_sec": float(os.environ.get("LOG_INTERVAL_SEC", "2.0")),
    "_status": "Initializing",
    "_latency": 0.0,
    "_inference_fps": 0.0,
    "_diag": {
        "before": 0,
        "after": 0,
        "discarded_small": 0,
        "discarded_large": 0,
        "camera": "-",
        "active_conf": 0.0,
    },
}

config_lock = threading.Lock()

# ------------------------------
# Model
# ------------------------------
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"Loaded model: {MODEL_PATH}")
        print(f"Class names: {getattr(model, 'names', {})}")
    else:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")


# ------------------------------
# Camera workers (camA/camB)
# ------------------------------
class CameraWorker:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.last_ok = 0.0

    def open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def start(self):
        self.running = True
        self.cap = self.open()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.2)
                self.cap = self.open()
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            self.last_ok = time.time()
            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


workers = {name: CameraWorker(source) for name, source in CAMERA_SOURCES.items()}
for worker in workers.values():
    worker.start()


last_predictions = []
predictions_lock = threading.Lock()
last_diag_log = 0.0


def _parse_bool(value):
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


def _safe_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _effective_inference_options(query_args):
    with config_lock:
        base = {
            "confidence": CONFIG["confidence"],
            "img_size": CONFIG["img_size"],
            "min_area": CONFIG["min_area"],
            "max_area": CONFIG["max_area"],
            "top_k": CONFIG["top_k"],
            "max_det": CONFIG["max_det"],
            "run_detection": CONFIG["run_detection"],
            "active_camera": CONFIG["active_camera"],
            "jpeg_quality": CONFIG["jpeg_quality"],
            "log_interval_sec": CONFIG["log_interval_sec"],
        }

    # Runtime overrides through query params
    base["confidence"] = _safe_float(query_args.get("conf"), base["confidence"])
    base["img_size"] = _safe_int(query_args.get("imgsz"), base["img_size"])
    base["min_area"] = _safe_int(query_args.get("min_area"), base["min_area"])
    base["max_area"] = _safe_int(query_args.get("max_area"), base["max_area"])
    base["top_k"] = _safe_int(query_args.get("top_k"), base["top_k"])
    base["max_det"] = _safe_int(query_args.get("max_det"), base["max_det"])
    base["jpeg_quality"] = max(50, min(95, _safe_int(query_args.get("jpeg_quality"), base["jpeg_quality"])))

    detect_override = _parse_bool(query_args.get("detect"))
    if detect_override is not None:
        base["run_detection"] = detect_override

    cam_override = query_args.get("cam")
    if cam_override in workers:
        base["active_camera"] = cam_override

    base["confidence"] = max(0.01, min(0.95, base["confidence"]))
    base["img_size"] = max(320, min(1280, base["img_size"]))
    base["min_area"] = max(0, base["min_area"])
    base["max_area"] = max(0, base["max_area"])
    base["top_k"] = max(0, base["top_k"])
    base["max_det"] = max(1, min(300, base["max_det"]))

    return base


def perform_detection(frame, options, camera_name):
    global last_diag_log

    if model is None:
        return frame, 0.0

    start_time = time.time()
    results = model.predict(
        frame,
        conf=options["confidence"],
        imgsz=options["img_size"],
        max_det=options["max_det"],
        verbose=False,
        device="cpu",
        half=False,
    )
    latency_ms = (time.time() - start_time) * 1000

    filtered = []
    before_count = 0
    discarded_small = 0
    discarded_large = 0

    if results:
        res = results[0]
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            before_count = len(boxes)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0]) if box.cls is not None else 0

                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area < options["min_area"]:
                    discarded_small += 1
                    continue
                if options["max_area"] > 0 and area > options["max_area"]:
                    discarded_large += 1
                    continue

                label_name = model.names.get(cls, "Weeds") if hasattr(model, "names") else "Weeds"
                filtered.append({
                    "class": label_name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "area": area,
                })

    filtered.sort(key=lambda item: item["confidence"], reverse=True)
    if options["top_k"] > 0:
        filtered = filtered[:options["top_k"]]

    for item in filtered:
        x1, y1, x2, y2 = item["box"]
        label = f"{item['class']} {item['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    with predictions_lock:
        global last_predictions
        last_predictions = [
            {"class": item["class"], "confidence": item["confidence"], "area": item["area"]}
            for item in filtered
        ]

    diag = {
        "before": before_count,
        "after": len(filtered),
        "discarded_small": discarded_small,
        "discarded_large": discarded_large,
        "camera": camera_name,
        "active_conf": options["confidence"],
    }

    with config_lock:
        CONFIG["_diag"] = diag
        CONFIG["_latency"] = round(latency_ms, 1)
        CONFIG["_status"] = "Detecting Weeds"

    now = time.time()
    if now - last_diag_log >= options["log_interval_sec"]:
        print(
            "[diag] cam=%s conf=%.2f before=%d after=%d small=%d large=%d" % (
                camera_name,
                options["confidence"],
                before_count,
                len(filtered),
                discarded_small,
                discarded_large,
            )
        )
        last_diag_log = now

    cv2.putText(
        frame,
        f"Detections: {len(filtered)} (raw {before_count})",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 220, 255),
        2,
    )

    return frame, latency_ms


def generate_frames(options):
    frame_count = 0
    start_time = time.time()

    while True:
        camera_name = options["active_camera"] if options["active_camera"] in workers else "camA"
        worker = workers.get(camera_name)
        if worker is None:
            time.sleep(0.05)
            continue

        raw_frame = worker.get_frame()
        if raw_frame is None:
            time.sleep(0.01)
            continue

        if options["run_detection"]:
            processed_frame, _ = perform_detection(raw_frame, options, camera_name)
        else:
            processed_frame = raw_frame
            with predictions_lock:
                global last_predictions
                last_predictions = []
            with config_lock:
                CONFIG["_status"] = "Stream Only"
                CONFIG["_latency"] = 0.0
                CONFIG["_diag"] = {
                    "before": 0,
                    "after": 0,
                    "discarded_small": 0,
                    "discarded_large": 0,
                    "camera": camera_name,
                    "active_conf": options["confidence"],
                }

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            with config_lock:
                CONFIG["_inference_fps"] = round(frame_count / elapsed, 1)
            frame_count = 0
            start_time = time.time()

        ok, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), options["jpeg_quality"]])
        if not ok:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    with config_lock:
        snapshot = dict(CONFIG)
    return render_template('index.html', config=snapshot, camera_sources=CAMERA_SOURCES)


@app.route('/video_feed')
def video_feed():
    options = _effective_inference_options(request.args)
    return Response(generate_frames(options), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_telemetry')
def get_telemetry():
    with predictions_lock:
        preds = list(last_predictions)

    with config_lock:
        cfg = dict(CONFIG)

    return jsonify({
        "status": cfg["_status"],
        "latency": cfg["_latency"],
        "fps": cfg["_inference_fps"],
        "predictions": preds,
        "diag": cfg["_diag"],
        "active_camera": cfg["active_camera"],
        "run_detection": cfg["run_detection"],
        "confidence": cfg["confidence"],
        "img_size": cfg["img_size"],
        "min_area": cfg["min_area"],
        "max_area": cfg["max_area"],
        "top_k": cfg["top_k"],
        "max_det": cfg["max_det"],
    })


@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json or {}

    with config_lock:
        if "active_camera" in data and data["active_camera"] in workers:
            CONFIG["active_camera"] = data["active_camera"]

        if "confidence" in data:
            CONFIG["confidence"] = max(0.01, min(0.95, _safe_float(data["confidence"], CONFIG["confidence"])))

        if "img_size" in data:
            CONFIG["img_size"] = max(320, min(1280, _safe_int(data["img_size"], CONFIG["img_size"])))

        if "min_area" in data:
            CONFIG["min_area"] = max(0, _safe_int(data["min_area"], CONFIG["min_area"]))

        if "max_area" in data:
            CONFIG["max_area"] = max(0, _safe_int(data["max_area"], CONFIG["max_area"]))

        if "top_k" in data:
            CONFIG["top_k"] = max(0, _safe_int(data["top_k"], CONFIG["top_k"]))

        if "max_det" in data:
            CONFIG["max_det"] = max(1, min(300, _safe_int(data["max_det"], CONFIG["max_det"])))

        if "run_detection" in data:
            parsed = _parse_bool(data["run_detection"])
            if parsed is not None:
                CONFIG["run_detection"] = parsed

    return jsonify({"success": True, "config": CONFIG})


@app.route('/health')
def health():
    now = time.time()
    out = {}
    for name, worker in workers.items():
        out[name] = {
            "source": worker.source,
            "alive": (now - worker.last_ok) < 2 if worker.last_ok else False,
        }
    return jsonify(out)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
