"""
Project Name : Drone Navigator's Telemetry System
Author : Gurucharan C
Model Used   : YOLOv8n (Nano)
"""
# ===================== IMPORTS =====================
import cv2
import time
from ultralytics import YOLO
from collections import Counter

"""
Confidence vs IOU Logic:
- Confidence threshold is used to filter out low-probability detections and reduce false positives.
- IOU (Intersection over Union) is used internally by YOLO during Non-Maximum Suppression (NMS)
  to remove overlapping bounding boxes referring to the same object.
- While confidence filtering is explicitly handled in this code, IOU-based suppression is
  automatically managed by the YOLO inference pipeline.
"""

# ===================== CONFIG =====================
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "video.mp4"
FRAME_SIZE = 640
CONF_THRESHOLD = 0.6
DETECT_EVERY = 3   # detection every N frames

# ===================== LOAD MODEL =====================
model = YOLO(MODEL_PATH)

# ===================== VIDEO =====================
cap = cv2.VideoCapture(VIDEO_PATH)

# ===================== VARIABLES =====================
prev_time = time.time()
frame_count = 0
last_detections = []   # (box, class_id)

# ===================== MAIN LOOP =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    h, w, _ = frame.shape
    frame_count += 1

    class_counter = Counter()

    # ---------------- DETECTION (SLOW PART) ----------------
    if frame_count % DETECT_EVERY == 0:
        last_detections = []

        results = model.predict(
            frame,
            verbose=False
        )

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(boxes, confs, class_ids):
                if conf >= CONF_THRESHOLD:
                    last_detections.append((box.astype(int), int(cls_id)))
                    class_counter[model.names[int(cls_id)]] += 1

        # -------- TERMINAL OUTPUT (GENERAL OBJECTS) --------
        detected_summary = ", ".join(
            [f"{count} {cls}" for cls, count in class_counter.items()]
        )

        print(
            f"{frame_count}: {FRAME_SIZE}x{FRAME_SIZE} "
            f"{detected_summary if detected_summary else 'No objects detected'}"
        )

    # ---------------- DRAWING (FAST PART) ----------------
    for box, cls_id in last_detections:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2

        lock_min = int(w * 0.4)
        lock_max = int(w * 0.6)

        if lock_min <= cx <= lock_max:
            color = (0, 0, 255)   # Target locked
            status = "TARGET LOCKED"
        else:
            color = (0, 255, 0)   # Scanning
            status = "SCANNING"

        label = model.names[cls_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} | {status}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ---------------- FPS (DISPLAY FPS) ----------------
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time + 1e-6))
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "DRONE HUD ACTIVE", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Drone Navigator Telemetry System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()