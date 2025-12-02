import cv2
import yaml

# Load config
with open("configs/camera.yaml", "r") as f:
    config = yaml.safe_load(f)

VIDEO_PATH = config["video_path"]
FPS_LIMIT = config["fps"]
RES = tuple(config["resolution"])

print(f" Opening video: {VIDEO_PATH}")
print(f"Settings -> Target FPS: {FPS_LIMIT}, Resolution: {RES}")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error opening video")
    exit()

native_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video loaded. Native FPS: {native_fps}, Total frames: {total_frames}")

frame_count = 0
processed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    # FPS throttling
    if frame_count % int(native_fps / FPS_LIMIT) != 0:
        frame_count += 1
        continue

    # Preprocessing
    frame_resized = cv2.resize(frame, RES)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb / 255.0  # Normalize 0–1

    processed_frames += 1
    if processed_frames % 10 == 0:  # Print progress every 10 processed frames
        print(f"Processed frame {frame_count}/{total_frames} "
              f"({(frame_count/total_frames)*100:.2f}%)")

    # Display
    cv2.imshow("Bus Video (Preprocessed)", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User stopped the video.")
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Finished. Total processed frames: {processed_frames}")
