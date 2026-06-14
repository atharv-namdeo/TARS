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

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configuration
MODEL_PATH = "models/weed_best.pt"
CONFIG = {
    "active_mode": "local",
    "local_path": MODEL_PATH,
    "confidence": 0.25,
    "camera_source_type": "ip",
    "camera_index": 0,
    "ip_cam_url": "http://192.168.29.1:4747/video",
    "run_detection": True,
    "_status": "Initializing",
    "_latency": 0.0,
    "_inference_fps": 0.0,
    "_cam_backend": "None",
}

# Global objects
camera = None
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"Loaded local model: {MODEL_PATH}")
    else:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

last_predictions = []
predictions_lock = threading.Lock()


class CameraStream:
    def __init__(self, source_type="ip", url="", index=0):
        import numpy as np
        self.lock = threading.Lock()
        self.running = False
        self.source_type = source_type
        self.url = url
        self.index = index
        self.cap = None

        self.placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frame = self.placeholder.copy()
        self._set_error_frame(self.frame, "Connecting to camera...", "Please wait...")
        
        self.running = True
        self.thread = threading.Thread(target=self._update, name="CameraThread", daemon=True)
        self.thread.start()

    def _set_error_frame(self, frame, line1, line2=""):
        frame[:] = (0, 0, 0)
        cv2.putText(frame, line1, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if line2:
            cv2.putText(frame, line2, (30, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    def _update(self):
        if self.source_type == "ip":
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                self._set_error_frame(self.placeholder, f"Cannot reach: {self.url}", "Check IP cam status")
                CONFIG["_cam_backend"] = "IP_FAILED"
            else:
                self.cap = cap
                CONFIG["_cam_backend"] = "IP_OK"
        else:
            cap = cv2.VideoCapture(self.index)
            if not cap.isOpened():
                self._set_error_frame(self.placeholder, f"USB Cam {self.index} Not Found", "Check connections")
                CONFIG["_cam_backend"] = "USB_FAILED"
            else:
                self.cap = cap
                CONFIG["_cam_backend"] = "USB_OK"

        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.1)
            else:
                time.sleep(1.0)

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


def perform_detection(frame):
    global last_predictions, model
    if model is None:
        return frame, 0.0

    start_time = time.time()
    results = model.predict(frame, conf=CONFIG["confidence"], verbose=False)
    latency = (time.time() - start_time) * 1000

    new_preds = []
    if len(results) > 0:
        res = results[0]
        annotated_frame = res.plot()
        for box in res.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            new_preds.append({"class": name, "confidence": conf})
        
        with predictions_lock:
            last_predictions = new_preds
        return annotated_frame, latency
    
    return frame, latency


def generate_frames():
    global camera
    frame_count = 0
    start_time = time.time()

    while True:
        if camera is None:
            time.sleep(0.1)
            continue

        raw_frame = camera.get_frame()
        
        if CONFIG["run_detection"]:
            processed_frame, latency = perform_detection(raw_frame)
            CONFIG["_latency"] = round(latency, 1)
            CONFIG["_status"] = "Detecting Weeds"
        else:
            processed_frame = raw_frame
            CONFIG["_latency"] = 0.0
            CONFIG["_status"] = "Stream Only"

        # Stats
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            CONFIG["_inference_fps"] = round(frame_count / elapsed, 1)
            frame_count = 0
            start_time = time.time()

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', config=CONFIG)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_telemetry')
def get_telemetry():
    with predictions_lock:
        preds = list(last_predictions)
    return jsonify({
        "status": CONFIG["_status"],
        "latency": CONFIG["_latency"],
        "fps": CONFIG["_inference_fps"],
        "predictions": preds,
        "cam_backend": CONFIG["_cam_backend"]
    })


@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json
    global camera
    
    if "camera_source_type" in data or "ip_cam_url" in data or "camera_index" in data:
        if "camera_source_type" in data: CONFIG["camera_source_type"] = data["camera_source_type"]
        if "ip_cam_url" in data: CONFIG["ip_cam_url"] = data["ip_cam_url"]
        if "camera_index" in data: CONFIG["camera_index"] = data["camera_index"]
        
        if camera:
            camera.stop()
        camera = CameraStream(
            source_type=CONFIG["camera_source_type"],
            url=CONFIG["ip_cam_url"],
            index=CONFIG["camera_index"]
        )

    if "confidence" in data:
        CONFIG["confidence"] = float(data["confidence"])
    
    if "run_detection" in data:
        CONFIG["run_detection"] = data["run_detection"]

    return jsonify({"success": True})


if __name__ == '__main__':
    # Default startup
    camera = CameraStream(
        source_type=CONFIG["camera_source_type"],
        url=CONFIG["ip_cam_url"],
        index=CONFIG["camera_index"]
    )
    app.run(host='0.0.0.0', port=5000, threaded=True)
