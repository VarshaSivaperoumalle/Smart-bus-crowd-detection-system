import json
import time
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
cred = credentials.Certificate("configs/firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
def send_to_firebase(payload):
    bus_id = payload["bus_id"]
    timestamp = datetime.utcnow().isoformat() + "Z"
    bus_ref = db.collection("buses").document(bus_id)
    bus_ref.set({
        "last_count": payload["count"],
        "seats_left": payload["seats_left"],
        "capacity": payload["capacity"],
        "occupancy_percent": payload["occupancy_percent"],
        "occupancy_level": payload["occupancy_level"],
        "conclusion": payload["conclusion"],
        "last_update": timestamp
    }, merge=True)

    # History collection
    bus_ref.collection("history").add({
        **payload,
        "recorded_at": firestore.SERVER_TIMESTAMP
    })

    print(f"✅ Sent to Firebase: Bus {bus_id} | {payload['occupancy_level']} | {payload['count']} passengers")

# ----------------------------
# Replay from log file
# ----------------------------
if __name__ == "__main__":
    with open("outputs/occupancy_log.jsonl", "r") as f:
        for line in f:
            payload = json.loads(line.strip())
            send_to_firebase(payload)
            time.sleep(1)  # simulate real-time streaming
