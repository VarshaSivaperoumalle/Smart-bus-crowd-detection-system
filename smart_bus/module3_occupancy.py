import cv2
import yaml
import json
import math
import numpy as np
from collections import deque
from ultralytics import YOLO
from datetime import datetime
import os

# ✅ Firebase
import firebase_admin
from firebase_admin import credentials, firestore


# ----------------------------
# Centroid Tracker
# ----------------------------
class CentroidTracker:
    def __init__(self, maxDisappeared=30, maxDistance=50):
        self.nextObjectID = 0
        self.objects = {}
        self.bboxes = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid, bbox):
        oid = self.nextObjectID
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.nextObjectID += 1

    def deregister(self, oid):
        del self.objects[oid], self.bboxes[oid], self.disappeared[oid]

    def update(self, rects):
        if len(rects) == 0:
            remove = []
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    remove.append(oid)
            for oid in remove:
                self.deregister(oid)
            return self.bboxes.copy()

        input_centroids = [(int((x1+x2)/2), int((y1+y2)/2)) for (x1,y1,x2,y2) in rects]

        if not self.objects:
            for c, r in zip(input_centroids, rects):
                self.register(c, r)
            return self.bboxes.copy()

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())
        D = np.zeros((len(objectCentroids), len(input_centroids)), dtype="float")

        for i, oc in enumerate(objectCentroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = math.hypot(oc[0]-ic[0], oc[1]-ic[1])

        assignedRows, assignedCols = set(), set()
        while True:
            idx = np.unravel_index(np.argmin(D), D.shape)
            if np.isinf(D[idx]): break
            r, c = idx
            if r in assignedRows or c in assignedCols:
                D[r, c] = np.inf
                continue
            if D[r, c] > self.maxDistance: break
            oid = objectIDs[r]
            self.objects[oid] = input_centroids[c]
            self.bboxes[oid] = rects[c]
            self.disappeared[oid] = 0
            assignedRows.add(r); assignedCols.add(c)
            D[r, :] = np.inf; D[:, c] = np.inf

        for i, oid in enumerate(objectIDs):
            if i not in assignedRows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)

        for j, c in enumerate(input_centroids):
            if j not in assignedCols:
                self.register(c, rects[j])

        return self.bboxes.copy()


# ----------------------------
# Helpers
# ----------------------------
def occupancy_level(count, empty_th, moderate_th):
    if count <= empty_th: return "Empty"
    if count <= moderate_th: return "Moderate"
    return "Overcrowded"

def conclusion(percent, seats_left):
    if seats_left <= 0 or percent >= 90:
        return "Overcrowded — no seats available."
    if percent >= 60:
        return "Crowded — few seats remaining."
    if percent >= 20:
        return "Moderate — seats available."
    return "Light — many seats available."


# ----------------------------
# Main
# ----------------------------
def main():
    # Load configs
    with open("configs/camera.yaml") as f: cam_cfg = yaml.safe_load(f)
    with open("configs/detection.yaml") as f: det_cfg = yaml.safe_load(f)

    VIDEO_PATH = cam_cfg["video_path"]
    RES = tuple(cam_cfg["resolution"])
    TARGET_FPS = cam_cfg["fps"]
    BUS_ID = cam_cfg["bus_id"]
    CAPACITY = cam_cfg["bus_capacity"]

    model = YOLO(det_cfg["model"])
    CONF, IOU = det_cfg["conf_threshold"], det_cfg["iou_threshold"]
    tracker = CentroidTracker(det_cfg["max_disappeared"], det_cfg["min_track_distance"])
    counts_history = deque(maxlen=det_cfg["smooth_window"])

    os.makedirs("outputs", exist_ok=True)
    log_path = os.path.join("outputs", "occupancy_log.jsonl")

    # ✅ Firebase Init
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    cap = cv2.VideoCapture(VIDEO_PATH)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS

    print(f"🎥 Video: {VIDEO_PATH}, Bus: {BUS_ID}, Capacity: {CAPACITY}")

    win = "Occupancy Monitor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_idx, processed = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % int(native_fps/TARGET_FPS) != 0:
            frame_idx += 1; continue

        resized = cv2.resize(frame, RES)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = model(rgb, imgsz=RES[0], conf=CONF, iou=IOU)[0]

        rects = []
        for box in results.boxes:
            clsid = int(box.cls[0])
            if model.names[clsid] != "person": continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            rects.append((x1,y1,x2,y2))

        objs = tracker.update(rects)
        count = len(objs)
        counts_history.append(count)
        smoothed = int(np.median(list(counts_history)))
        seats_left = max(0, CAPACITY - smoothed)
        percent = min(100, int((smoothed/CAPACITY)*100))

        occ_level = occupancy_level(smoothed, det_cfg["empty_threshold"], det_cfg["moderate_threshold"])
        concl = conclusion(percent, seats_left)

        vis = resized.copy()
        for oid, (x1,y1,x2,y2) in objs.items():
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"ID {oid}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        cv2.putText(vis, f"Count:{smoothed} Seats:{seats_left}/{CAPACITY}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        cv2.putText(vis, f"Occupancy:{percent}% ({occ_level})", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
        cv2.putText(vis, f"Conclusion:{concl}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)

        cv2.imshow(win, vis)

        # Payload
        payload = {
            "bus_id": BUS_ID,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "count": smoothed,
            "seats_left": seats_left,
            "capacity": CAPACITY,
            "occupancy_percent": percent,
            "occupancy_level": occ_level,
            "conclusion": concl,
            "last_update": datetime.now().isoformat()
        }

        # Save locally
        with open(log_path, "a") as f:
            f.write(json.dumps(payload) + "\n")

        # ✅ Save to Firebase
        bus_ref = db.collection("buses").document(BUS_ID)
        bus_ref.set(payload, merge=True)

        processed += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Finished. Total processed frames:", processed)
    print("Logs saved at:", log_path)


if __name__ == "__main__":
    main()
