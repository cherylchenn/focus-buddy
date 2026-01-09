# Focus Buddy

## Overview
Focus Buddy is a real-time head pose monitoring system that detects whether a user is looking away from their screen and emits an alert after prolonged distraction. It uses a multi-layer perceptron (MLP) trained on facial landmarks extracted from the BIWI Kinect Head Pose dataset, with MediaPipe and OpenCV for live webcam inference.

## Features
- Multi-class head pose classification: center, left, right, up, down
- Trained on the BIWI Kinect Head Pose dataset (93% validation accuracy)
- Facial landmark extraction using MediaPipe Face Mesh (468 landmarks)
- Real-time webcam inference with face bounding box and pose label
- Automated audio alert when the user looks away for longer than a configurable threshold
- Lightweight MLP architecture suitable for real-time use

## Tech Stack
- PyTorch: model training and inference
- MediaPipe: facial landmark detection
- OpenCV: webcam input and visualization
- NumPy: data preprocessing

## Dataset
- [BIWI Kinect Head Pose dataset](https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database)
- Facial landmarks are normalized relative to the nose and yaw/pitch angles are discretized into pose classes

## Installation + Usage
1. Install dependencies: ```pip install -r requirements.txt```
2. Preprocess the dataset: ```python preprocess_biwi.py```  
This will save the extracted landmarks to ```data/biwi_landmarks.npz```.
3. Train the model with the extracted landmarks: ```python train_mlp.py```   
This will save the trained model to ```models/mlp.pth```.
5. Run the real-time webcam: ```python main.py```.
6. To exit, press the ```q``` key.
