import cv2
import yaml
import time
import json
import math
import numpy as np
from collections import deque
from ultralytics import YOLO
from datetime import datetime
class CentroidTracker:
    def __init__(self, maxDisappeared=30, maxDistance=50):
        self.nextObjectID = 0
        self.objects = dict()      # objectID -> centroid (x, y)
        self.bboxes = dict()       # objectID -> bbox (x1,y1,x2,y2)
        self.disappeared = dict()  # objectID -> disappeared frames
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid, bbox):
        oid = self.nextObjectID
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.nextObjectID += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.bboxes[oid]
        del self.disappeared[oid]

    def update(self, rects):
        """
        rects: list of bboxes [(x1,y1,x2,y2), ...]
        returns current dict objectID -> bbox
        """
        # If no detections, mark all as disappeared
        if len(rects) == 0:
            remove = []
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    remove.append(oid)
            for oid in remove:
                self.deregister(oid)
            return self.bboxes.copy()

        # Compute centroids for new rects
        input_centroids = []
        for (x1,y1,x2,y2) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY))

        # If no existing objects, register all
        if len(self.objects) == 0:
            for centroid, bbox in zip(input_centroids, rects):
                self.register(centroid, bbox)
            return self.bboxes.copy()

        # Build distance matrix between existing centroids and incoming centroids
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())
        D = np.zeros((len(objectCentroids), len(input_centroids)), dtype="float")

        for i, oc in enumerate(objectCentroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = math.hypot(oc[0]-ic[0], oc[1]-ic[1])

        # Greedy assignment: repeatedly pick the smallest distance pair
        assignedRows = set()
        assignedCols = set()
        while True:
            idx = np.unravel_index(np.argmin(D), D.shape)
            minval = D[idx]
            if np.isinf(minval):
                break
            r, c = idx
            if r in assignedRows or c in assignedCols:
                D[r, c] = np.inf
                continue
            if D[r, c] > self.maxDistance:
                # Remaining pairs are too far
                break
            # Assign objectIDs[r] -> input_centroids[c]
            oid = objectIDs[r]
            self.objects[oid] = input_centroids[c]
            self.bboxes[oid] = rects[c]
            self.disappeared[oid] = 0
            assignedRows.add(r)
            assignedCols.add(c)
            # invalidate this row & col
            D[r, :] = np.inf
            D[:, c] = np.inf

        # Unassigned existing object rows -> increment disappeared
        for i, oid in enumerate(objectIDs):
            if i not in assignedRows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)

        # Unassigned input cols -> register as new objects
        for j, centroid in enumerate(input_centroids):
            if j not in assignedCols:
                self.register(centroid, rects[j])

        return self.bboxes.copy()


# ----------------------------
# Helper: occupancy level
# ----------------------------
def occupancy_level(count, empty_th, moderate_th):
    if count <= empty_th:
        return "Empty"
    if count <= moderate_th:
        return "Moderate"
    return "Overcrowded"


# ----------------------------
# Main detection loop
# ----------------------------
def main():
    # Load configs
    with open("configs/camera.yaml", "r") as f:
        cam_cfg = yaml.safe_load(f)
    with open("configs/detection.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    VIDEO_PATH = cam_cfg["video_path"]
    RES = tuple(cam_cfg["resolution"])
    TARGET_FPS = cam_cfg.get("fps", 5)

    model_path = cfg["model"]
    CONF = cfg["conf_threshold"]
    IOU = cfg["iou_threshold"]
    device = cfg.get("device", "cpu")
    smooth_window = cfg.get("smooth_window", 5)
    empty_th = cfg.get("empty_threshold", 5)
    moderate_th = cfg.get("moderate_threshold", 30)
    send_interval = cfg.get("send_interval", 5)

    # Initialize model
    print(f"Loading YOLO model {model_path} on device {device} ...")
    model = YOLO(model_path)
    # Optionally set model to device (ultralytics chooses automatically but you can pass device in call)
    print("Model loaded. Class names:", model.names)

    # Tracker and smoothing deque
    tracker = CentroidTracker(maxDisappeared=cfg.get("max_disappeared", 30),
                              maxDistance=cfg.get("min_track_distance", 50))
    counts_history = deque(maxlen=smooth_window)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video:", VIDEO_PATH)
        return
    native_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video opened: {VIDEO_PATH}, native FPS={native_fps}, total_frames={total_frames}")

    frame_idx = 0
    processed = 0
    last_send_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🏁 End of video")
            break

        # FPS throttling
        skip = max(1, int(native_fps / TARGET_FPS))
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        # Preprocess: resize and convert to RGB (YOLO prefers RGB)
        frame_resized = cv2.resize(frame, RES)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Run detection
        # pass device via model(...) call if needed: model(frame, device=device, imgsz=RES[0], conf=CONF, iou=IOU)
        results = model(frame_rgb, imgsz=RES[0], conf=CONF, iou=IOU)[0]

        # Parse detections robustly
        rects = []  # boxes for person -> [(x1,y1,x2,y2), ...]
        try:
            # try to get tensors then to numpy
            xyxy = results.boxes.xyxy.cpu().numpy()    # shape (n,4)
            confs = results.boxes.conf.cpu().numpy()
            clsids = results.boxes.cls.cpu().numpy().astype(int)
            for (x1,y1,x2,y2), conf, clsid in zip(xyxy, confs, clsids):
                label = model.names.get(int(clsid), str(clsid))
                if label != "person" and label != "Person" and clsid != 0:
                    continue
                rects.append((int(x1), int(y1), int(x2), int(y2)))
        except Exception:
            # fallback: iterate object boxes
            for box in results.boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    clsid = int(box.cls[0].cpu().numpy())
                    label = model.names.get(clsid, str(clsid))
                    if label != "person" and clsid != 0:
                        continue
                    rects.append((int(x1), int(y1), int(x2), int(y2)))
                except Exception:
                    continue

        # Update tracker
        objects = tracker.update(rects)
        current_count = len(objects)
        counts_history.append(current_count)
        # compute smoothed count (median is more robust to spikes)
        smoothed = int(np.median(list(counts_history))) if len(counts_history) > 0 else current_count
        level = occupancy_level(smoothed, empty_th, moderate_th)

        # Debug prints
        processed += 1
        if processed % 5 == 0:
            print(f"[Frame {frame_idx}] Detections: {len(rects)}, Tracked objects: {current_count}, Smoothed: {smoothed}, Level: {level}")

        # Draw detections and IDs on frame_resized for quick visualization
        vis = frame_resized.copy()
        for oid, bbox in objects.items():
            (x1,y1,x2,y2) = bbox
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cx = int((x1+x2)/2); cy = int((y1+y2)/2)
            cv2.putText(vis, f"ID {oid}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Overlay count text
        cv2.putText(vis, f"Count: {smoothed} | {level}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow("Detection+Tracking", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 User stopped")
            break

        # Prepare JSON payload
        payload = {
            "bus_id": cam_cfg.get("bus_id", "TEST_BUS"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "count": int(smoothed),
            "occupancy": level
        }

        # Optionally: send to cloud at intervals (placeholder function)
        now = time.time()
        if now - last_send_time > send_interval:
            # CALL YOUR send_update(bus_id, count, occupancy) HERE
            # Example:
            # send_update(payload["bus_id"], payload["count"], payload["occupancy"])
            # For now we just print the JSON
            print(">>> OUT:", json.dumps(payload))
            last_send_time = now

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Finished. Total processed frames:", processed)


if __name__ == "__main__":
    main()
