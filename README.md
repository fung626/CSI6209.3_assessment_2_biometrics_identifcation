# Real-Time Driver Drowsiness / Micro-Sleep Detection

## 1. Project Overview

This project develops an AI-powered real-time driver drowsiness detection system designed to monitor a driver’s alertness via live webcam feed.

The system continuously tracks facial features, evaluates eye and mouth states through a hybrid approach combining geometric metrics (such as EAR and MAR) and trained deep learning models, and performs temporal analysis to distinguish normal blinking from dangerous driver fatigue.

```text
Live Camera Feed
       ↓
Face, Eye & Mouth Landmark Detection (MediaPipe)
       ↓
Hybrid Feature & Image Processing (EAR/MAR & Eye Crops)
       ↓
Unified Decision Pipeline (Trained Models + Threshold Logic)
       ↓
Time-Series Analysis (PERCLOS & Closure Duration)
       ↓
Awake / Drowsy Status & Real-Time Alert
```

---

# 3. Main Components

## A. Dataset

### [MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset): Used to train and evaluate eye-state classification models (Open vs. Closed eyes).

### [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd): Used for broader driver fatigue state validation (Drowsy vs. Non-Drowsy).

## B. Computer Vision & Feature Extraction

### [MediaPipe](https://github.com/google-ai-edge/mediapipe): Powers high-performance face mesh tracking to extract precise coordinates for eyes and mouth in real-time.

### Geometric Metrics: Calculates Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to capture instantaneous blink and yawn behaviors.

## C. Deep Learning & Unified Models

### [MobileNetV2](https://www.sciencedirect.com/topics/computer-science/mobilenetv2) / CNN: Lightweight convolutional neural networks optimized for fast image classification on eye regions.

### Unified Decision Pipeline: Combines deterministic geometric thresholds with trained model inferences into a single, robust scoring pipeline to minimize false positives caused by head movement or lighting variations.

## D. Time-Series & Temporal Analysis

### PERCLOS (Percentage of Eye Closure): Measures the proportion of time eyes remain closed over a sliding time window.

### Consecutive Closed Frames: Tracks prolonged micro-sleeps.

### Yawn Frequency: Integrates mouth state tracking via MAR to reinforce fatigue detection.

---

# 3. System Architecture

```text
                      ┌───────────────────┐
                      │    Live Camera    │
                      └─────────┬─────────┘
                                ↓
                      ┌───────────────────┐
                      │ MediaPipe Face    │
                      │ & Landmark Mesh   │
                      └─────────┬─────────┘
                                │
         ┌──────────────────────┴──────────────────────┐
         ↓                                             ↓
┌─────────────────┐                           ┌─────────────────┐
│ EAR & MAR       │                           │ Eye Image Crops │
│ Calculations    │                           │ (MobileNetV2)   │
└────────┬────────┘                           └────────┬────────┘
         │                                             │
         └──────────────────────┬──────────────────────┘
                                ↓
                      ┌───────────────────┐
                      │ Unified Decision  │
                      │ Pipeline          │
                      └─────────┬─────────┘
                                ↓
                      ┌───────────────────┐
                      │ Time-Series       │
                      │ Temporal Analysis │
                      │ (PERCLOS)         │
                      └─────────┬─────────┘
                                ↓
                      ┌───────────────────┐
                      │ Drowsiness Level  │
                      └─────────┬─────────┘
                                ↓
                     ┌──────────┴──────────┐
                     ↓                     ↓
                   Awake                 Drowsy
                                           ↓
                                    🚨 Audio/Visual 🚨
                                         WARNING
```

---

# 4. Key Features & Pipeline Stages

### 1. Robust Feature Extraction: Computes real-time EAR and MAR from MediaPipe face landmarks with built-in safety checks for boundary conditions.

### 2. Hybrid Inference: Merges geometric calculations with trained model predictions to ensure high accuracy across diverse lighting conditions.

### 3. Temporal Sliding Window: Evaluates fatigue dynamically over time rather than reacting to single-frame blinks.

### 4. Real-Time Warning System: Triggers immediate alerts when PERCLOS or closure duration exceeds safe driving thresholds.

### 5. Analyse eye behaviour over time using measures such as:

---

# 5. Expected System Output

=== DRIVER MONITORING ACTIVE ===

| Item        | Result |
| ----------- | ------ |
| Eye State   | OPEN   |
| Mouth State | CLOSED |
| PERCLOS     | 8.4%   |
| Drowsiness  | LOW    |
| Status      | AWAKE  |

When prolonged closure or high PERCLOS is detected:

=== DRIVER MONITORING ACTIVE ===

| Item        | Result   |
| ----------- | -------- |
| Eye State   | CLOSED   |
| Mouth State | YAWNING  |
| PERCLOS     | 52.1%    |
| Drowsiness  | CRITICAL |
| Status      | DROWSY   |

⚠️ WARNING ⚠️
DRIVER FATIGUE DETECTED!

---
