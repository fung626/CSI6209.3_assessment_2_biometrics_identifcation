# Real-Time Driver Drowsiness / Micro-Sleep Detection: Project Structure

This document outlines the architectural layout, modules, datasets, and script references for the CSI6209.3 Biometrics Identification project.

## 1. Directory Tree

```text
CSI6209.3_assessment_2_biometrics_identifcation/
│
├── assets/
│   ├── fonts/
│   ├── icons/
│   ├── images/
│   ├── sounds/
│
├── data/
│   ├── Driver-Drowsiness-Dataset/
│   │   ├── test/
│   │   │   ├── Drowsy/
│   │   │   └── Non Drowsy/
│   │   ├── train/
│   │   │   ├── Drowsy/
│   │   │   └── Non Drowsy/
│   │   └── val/
│   │       ├── Drowsy/
│   │       └── Non Drowsy/
│   └── MRL-Eye-Dataset/
│       ├── get_info.py
│       ├── labels.txt
│       ├── readme.md
│       ├── split_data.py
│       ├── test/
│       │   ├── awake/
│       │   └── sleepy/
│       ├── train/
│       │   ├── awake/
│       │   └── sleepy/
│       └── val/
│           ├── awake/
│           └── sleepy/
│
├── computer_vision/
│   ├── eye_detection/
│   │   └── eye_detector.py
│   └── facial_landmarks/
│       ├── facial_landmark_detector.py
│       └── models/
│           └── face_landmarker.task
│
├── deep_learning/
│   ├── cnn/
│   └── mobilenetv2/
│       ├── mobilenetv2.py
│       ├── driver_drowsiness/
│       │   ├── train.py
│       │   └── model/
│       │       └── mobilenetv2_driver_drowsiness.keras
│       └── mrl_eye/
│           ├── eye_predictor.py
│           ├── train.py
│           └── model/
│               └── mobilenetv2_mrl_eye.keras
│
├── time_series/
│   ├── blink_detection/
│   ├── closure_duration/
│   ├── eye_state_tracking/
│   │   └── eye_state_tracker.py
│   └── perclos/
│
├── drowsiness_detection/
│   ├── drowsiness_logic/
│   │   └── drowsiness_evaluator.py
│   └── eye_closure_detection/
│   │   └── eye_closure_detector.py
│   └── yawn_detection/
│       └── yawn_detector.py
│
├── realtime/
│   ├── alert_system/
│   ├── buzzer_alert.py
│   └── webcam/
│       └── webcam.py
│
├── evaluation/
│   ├── confusion_matrix/
│   │   └── confusion_matrix.py
│   ├── metrics/
│   │   ├── evaluate_driver_drowsiness.py
│   │   ├── evaluate_mrl_eye.py
│   │   └── evaluator.py
│   ├── model_comparison/
│   │   └── model_comparison.py
│   ├── results/
│   │   ├── driver_drowsiness_results.json
│   │   └── mrl_eye_results.json
│   └── roc_curve/
│       └── roc_curve.py
│
├── training_results/
│   └── mobilenetv2/
│       ├── driver_drowsiness/
│       │   └── training_history.csv
│       └── mrl_eye/
│           └── training_history.csv
│
├── notebooks/
│
├── documentation/
│   └── PROJECT_STRUCTURE.md
│
├── main.py
├── config.py
├── README.md
├── requirements.txt
└── .venv/
```

## Core Component Breakdown

#### A. Data (data/)

- Driver-Drowsiness-Dataset: Binary classification dataset categorized into Drowsy and Non Drowsy.
- MRL-Eye-Dataset: Binary eye state dataset categorized into awake and sleepy.

#### B. Preprocessing (preprocessing/) — On Hold

- Handles data cleaning, image resizing, exploratory data analysis, and train/validation/test data partitioning scripts.

#### C. Computer Vision (computer_vision/)

- Facial Landmarks (facial_landmark_detector.py): Utilizes MediaPipe Face Landmarker to extract real-time facial mesh points, calculating Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).

- Eye Detection (eye_detector.py): Isolates and crops eye regions based on landmark coordinates with dynamic padding.

#### D. Classical Machine Learning (classical_ml/) — On Hold

- Preserved in the repository for baseline architecture comparisons (includes SVM, Random Forest, and EAR feature extractors).

#### E. Deep Learning (deep_learning/)

- MobileNetV2 (MRL-Eye): Binary classification model trained on eye crops (mobilenetv2_mrl_eye.keras).

- MobileNetV2 (Driver Drowsiness): Binary classification model trained on overall driver states (mobilenetv2_driver_drowsiness.keras).

#### F. Time Series Analysis (time_series/) — On Hold

- Evaluates sliding window queues across recent frames to calculate PERCLOS (Percentage of Eye Closure), closure duration, and blink frequency, bridging single-frame predictions with sustained fatigue detection.

#### G. Drowsiness Detection (drowsiness_detection/)

- Drowsiness Logic (drowsiness_evaluator.py): Fuses deep learning MobileNetV2 confidence scores with geometric metrics (EAR/MAR). Features a Yawn Override where a high MAR percentage immediately flags a drowsy state to bypass static CNN misclassifications.

#### H. Real-Time Processing (realtime/)

- Webcam App (webcam.py): Runs the live video processing loop displaying face/mouth boundary indicators, telemetry overlays, and triggering asynchronous buzzer alerts with cooldown protection.

#### I. Evaluation & Training Results (evaluation/, training_results/)

- Computes performance metrics (Accuracy, Precision, Recall, F1-Score, FAR, FRR), confusion matrices, ROC curves, and stores CSV training history logs.

## Key Script Reference

| Script                                                        | Purpose                                                  |
| ------------------------------------------------------------- | -------------------------------------------------------- |
| main.py                                                       | Main application entry point                             |
| realtime/webcam/webcam.py                                     | Real-time drowsiness detection from webcam feed          |
| computer_vision/facial_landmarks/facial_landmark_detector.py  | Extracts MediaPipe landmarks, computes EAR and MAR       |
| drowsiness_detection/drowsiness_logic/drowsiness_evaluator.py | Manages hybrid score fusion and hard geometric overrides |
| evaluation/metrics/evaluate_driver_drowsiness.py              | Evaluates Driver Drowsiness model on test split          |
| evaluation/metrics/evaluate_mrl_eye.py                        | Evaluates MRL Eye model on test split                    |
| deep_learning/mobilenetv2/driver_drowsiness/train.py          | Training script for Driver Drowsiness MobileNetV2        |
| deep_learning/mobilenetv2/mrl_eye/train.py                    | Training script for MRL Eye MobileNetV2                  |

## 4. Model Performance Summary

MobileNetV2 (MRL-Eye)

- Accuracy: 98.53%

- Precision: 99.36%

- Recall: 97.65%

- F1-Score: 98.50%

- FAR / FRR: 0.62% / 2.35%

MobileNetV2 (Driver Drowsiness)

- Accuracy: 99.94%

- Precision: 99.97%

- Recall: 99.90%

- F1-Score: 99.93%

- FAR / FRR: 0.03% / 0.10%

## 5. Course Module Mapping

- Module 3 (ML & Deep Learning): CNN, MobileNetV2, Transfer Learning, SVM, Random Forest.

- Module 4 (Computer Vision): Face/eye/mouth detection, facial landmarks, image processing.

- Module 5 (Time Series Analysis): Blink patterns, closure duration, sliding windows, PERCLOS telemetry.

- Module 8 (Evolutionary Algorithms): Optional threshold optimization extensions.
