# AI-Based Autonomous Surveillance System

## Overview
This project implements an AI-powered autonomous surveillance system using advanced object detection models (YOLOv8, Faster R-CNN) and a custom I3D two-stream model for anomaly detection. The system is designed to process video streams in real-time, detect specific actions, and flag anomalous events, enhancing traditional surveillance for public safety, traffic, and critical infrastructure monitoring.

## Features
- Real-time object detection (YOLOv8, Faster R-CNN)
- Anomaly detection using the I3D two-stream model
- Support for SPHAR and UCF-Crime datasets
- Modular, scalable, and cost-effective

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/VedPawar1410/ai-autonomous-surveillance-system.git
    ```

## Usage

- **YOLOv8 Inference:**
    ```
    from ultralytics import YOLO
    model = YOLO('yolov8x')
    results = model.predict('input_videos/neutral1.mp4', save=True)
    ```
- **Faster R-CNN Inference:**
    ```
    # See rcnn/rcnnInference.py for full example
    ```
- **I3D Two-Stream Model:**
    ```
    # See i3d_two_stream/main.py for training and evaluation
    ```

## Results

- **YOLOv8:** mAP@0.5 = 82–85%, 45–60 FPS
- **Faster R-CNN:** mAP@0.5 = 88–92%, 7–8 FPS
- **I3D Two-Stream:** AUC = 84.45% (UCF-Crime dataset)


## License
No License
