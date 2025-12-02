import cv2
import sys
def probe_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames / fps if fps > 0 else 0
    print(f"✅ Video Info: {width}x{height} @ {fps:.2f} FPS, {frames} frames, {duration:.2f} sec")
    cap.release()
if __name__ == "__main__":
    probe_video("C:/Users/Dell/Desktop/MICRO PROJECT/smart_bus/videos/bus_sample.mp4")
