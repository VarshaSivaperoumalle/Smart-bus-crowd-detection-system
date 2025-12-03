# Smart Bus Crowd Detection System

An AI-powered real-time crowd monitoring system for buses using YOLO-based object detection, camera feed processing, and automated occupancy analysis. This project helps ensure safety, crowd management, and operational efficiency in public transport systems.

## Project Overview

The Smart Bus Crowd Detection System captures live video from onboard cameras, detects passengers, counts occupancy, and sends updates to a backend/server or Firebase. It is designed for:

- Preventing overcrowding  
- Real-time monitoring by transport authorities  
- Automated passenger count analytics  
- Safety and surveillance

## Setup Instructions

1. Clone the Repository
git clone https://github.com/VarshaSivaperoumalle/Smart-bus-crowd-detection-system.git
cd Smart-bus-crowd-detection-system

2. Create Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS / Linux

3. Install Dependencies
pip install -r requirements.txt

4. Configure Environment Variables
Create a .env file in the root directory:
CAMERA_SOURCE=0
YOLO_MODEL_PATH=smart_bus/yolov8n.pt

# Firebase config 
FIREBASE_API_KEY=YOUR_API_KEY
FIREBASE_PROJECT_ID=YOUR_PROJECT_ID

5. Run the Person Detection System
python smart_bus/main_capture.py

This will:
Load YOLO model
Capture video frames
Detect persons
Display bounding boxes
Update real-time occupancy count
Save logs in smart_bus/outputs/

6. Test With Sample Video
Modify the .env file:
CAMERA_SOURCE=smart_bus/videos/bus_sample.mp4
Run detection:
python smart_bus/main_capture.py

## Project Structure
smart_bus/
│── main_capture.py
│── module2_detection.py
│── module3_occupancy.py
│── module4_transmission.py
│── configs/
│── outputs/
│── videos/
│── tools/
