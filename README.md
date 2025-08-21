# Real-Time Object Detection & Attendance Tracking System

A powerful, flexible real-time object detection system built with YOLOv8 ( just for example) and FastAPI that automatically tracks attendance by detecting and logging unique objects/people in video streams. Perfect for attendance monitoring, security applications, and object tracking scenarios.

## 🌟 Key Features

- **Real-Time Object Detection**: Uses YOLOv8 for accurate, fast object detection demo
- **Automatic Attendance Tracking**: Logs each detected class only once with timestamps
- **Live Web Interface**: Beautiful, responsive web UI with real-time video streaming
- **Multi-Client Support**: Handles up to 4 concurrent video streams
- **Flexible Architecture**: Easy to swap models and customize processing logic
- **Cloud & Local Support**: Works with ngrok for public access or locally
- **Performance Optimized**: GPU acceleration, batch processing, and efficient WebSocket communication

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │◄──►│   FastAPI Server │◄──►│  YOLO Processor │
│  (HTML/JS/CSS)  │    │  (WebSocket API) │    │   (Detection)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Attendance      │
                       │ Tracker (CSV)   │
                       └─────────────────┘
                       
                     
```


# 🚀 How to Execute the Code

## 1. Create Virtual Environment
python -m venv venv

### On Windows
venv\Scripts\activate

### On Linux/Mac
source venv/bin/activate

## 2. Install Dependencies
pip install -r requirements.txt

## 3. Run the Application
cd src
python video_streaming.py

## 4. Access the Web Interface
http://127.0.0.1:8080/static/index.html
