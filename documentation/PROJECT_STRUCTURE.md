# Real-Time Driver Drowsiness / Micro-Sleep Detection

## Project Structure

```text
CSI6209.3_assessment_2_biometrics_identifcation/
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
├── preprocessing/
│   ├── data_analysis/
│   ├── data_cleaning/
│   └── data_split/
│
├── computer_vision/
│   ├── eye_detection/
│   │   └── eye_detector.py
│   └── facial_landmarks/
│       ├── facial_landmark_detector.py
│       └── models/
│           └── face_landmarker.task
│
├── classical_ml/
│   ├── ear_features/
│   ├── random_forest/
│   └── svm/
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
│   └── micro_sleep_detection/
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

## Main Components

### 1. Data

Contains two datasets:

- **Driver-Drowsiness-Dataset**: Binary classification of driver drowsiness states
  - Drowsy
  - Non Drowsy

- **MRL-Eye-Dataset**: Eye state classification
  - awake
  - sleepy

### 2. Preprocessing

Responsible for:

- Data cleaning
- Image preprocessing
- Dataset analysis
- Training/validation/testing split

### 3. Computer Vision

Responsible for identifying and extracting the driver's eyes and facial features from video frames:

- **Facial Landmark Detection** (`facial_landmark_detector.py`): Uses MediaPipe to detect facial landmarks and compute real-time geometric ratios:
  - **Eye Aspect Ratio (EAR)**: Tracks eye openness to detect squinting or heavy eyelids.
  - **Mouth Aspect Ratio (MAR)**: Tracks lip distance to compute percentage-based yawn telemetry.
- **Eye Detection** (`eye_detector.py`): Extracts eye regions based on facial landmarks using MediaPipe Face Landmarker indices with configurable padding.

---

### 4. Classical Machine Learning (Currently on Hold)

_(Note: While the module structure is preserved in the repository for future evaluation, classical ML algorithms like SVM and Random Forest have been put on hold in favor of the more robust Deep Learning + Geometric Hybrid architecture.)_

- **EAR Features**: Designed to extract Eye Aspect Ratio metrics.
- **Support Vector Machine (SVM)**: Intended for classical feature-based eye state classification.
- **Random Forest**: Intended for ensemble-based fatigue detection.

### 5. Deep Learning

Neural network models for eye state classification:

- **MobileNetV2 for MRL-Eye**: Binary classification (awake/sleepy)
  - Model: `mobilenetv2_mrl_eye.keras`
  - Input: 224×224 RGB images
  - Output: Sigmoid probability (P(sleepy))
  - Architecture: MobileNetV2 base + GlobalAveragePooling2D + Dropout + Dense(1)

- **MobileNetV2 for Driver Drowsiness**: Binary classification (Drowsy/Non Drowsy)
  - Model: `mobilenetv2_driver_drowsiness.keras`
  - Input: 224×224 RGB images

### 6. Time Series Analysis

Analyzes temporal patterns in eye state:

- **Eye State Tracking** (`eye_state_tracker.py`): Tracks eye state over time
- **Blink Detection**: Detects and counts blinks
- **Closure Duration**: Measures how long eyes stay closed
- **PERCLOS**: Percentage of eye closure (standard drowsiness metric)

While frame-by-frame classifiers and geometric ratios (EAR/MAR) can determine eye and mouth states at a single millisecond, they cannot differentiate between a normal, healthy blink and a dangerous micro-sleep on their own. The **Time-Series Module** (`time_series/`) bridges this gap by evaluating behavioral patterns across a rolling historical window.

### Core Architecture & Sliding Window

The system maintains a rolling buffer (queue) of recent frame predictions and geometric states (typically tracking the last 30 to 60 frames). Instead of reacting instantly to a single closed-frame anomaly, the temporal engine analyzes sequences of data to detect sustained fatigue.

```text
Incoming Frame Predictions
      ↓
[ Rolling Frame Buffer (Queue) ]
      ├──> Closure Duration Calculation
      ├──> Blink Frequency Counter
      └──> PERCLOS Evaluation
      ↓
Temporal Drowsiness Flag
```

### 7. Drowsiness Detection

Main logic for detecting drowsiness and micro-sleeps:

- **Drowsiness Logic** (`drowsiness_evaluator.py`): Aggregates deep learning MobileNetV2 scores and geometric signals (EAR/MAR) using weighted fusion.
- **Yawn Override**: Integrates a hard geometric override where a high MAR percentage triggers an immediate "Drowsy" warning state, bypassing static CNN misclassifications.
- **Micro-Sleep Detection**: Detects sudden, prolonged eye closure and fatigue episodes.

---

### 8. Real-Time Processing

Live webcam monitoring application (webcam.py):

- Hybrid Pipeline: Captures video frames, extracts face landmarks via MediaPipe, and evaluates combined deep learning (MobileNetV2) and geometric signals (Eye Aspect Ratio EAR/ Mouth Aspect Ratio MAR).
- Dynamic Multi-Object Indicators:
- Face Indicator: Draws a color-coded bounding box and status label (DROWSY / AWAKE) based on the final aggregated score.
- Mouth Indicator: Isolates lip landmarks to compute percentage-based MAR telemetry, displaying live feedback (e.g., MAR: 84.5% (YAWNING) vs NOT YAWNING).
- Overall Status HUD & Warning Banner: Renders a clean top-left dashboard for system telemetry and triggers a bold red warning banner across the screen when a micro-sleep or yawn state breaches safety thresholds.
- Alert System: Triggers asynchronous buzzer alerts with built-in cooldown timers to prevent audio spam.

---

### 9. Evaluation

Model evaluation and validation:

- **Evaluator** (`evaluator.py`): Computes metrics (accuracy, precision, recall, F1)
- **Confusion Matrix** (`confusion_matrix.py`): Visualizes prediction errors
- **ROC Curve** (`roc_curve.py`): ROC curve analysis
- **Model Comparison** (`model_comparison.py`): Compares different models
- **Results**: JSON files with evaluation metrics

### 10. Training Results

Stores training history and artifacts:

- Training curves (accuracy, loss)
- Training history CSV files
- Trained model weights

## Key Scripts

| Script                                                          | Purpose                                                  |
| --------------------------------------------------------------- | -------------------------------------------------------- |
| `main.py`                                                       | Main application entry point                             |
| `realtime/webcam/webcam.py`                                     | Real-time drowsiness detection from webcam               |
| `computer_vision/facial_landmarks/facial_landmark_detector.py`  | Extracts MediaPipe face landmarks, computes EAR and MAR  |
| `drowsiness_detection/drowsiness_logic/drowsiness_evaluator.py` | Handles hybrid score fusion and hard geometric overrides |
| `evaluation/metrics/evaluate_driver_drowsiness.py`              | Evaluate Driver-Drowsiness model on test set             |
| `evaluation/metrics/evaluate_mrl_eye.py`                        | Evaluate MRL-Eye model on test set                       |
| `deep_learning/mobilenetv2/driver_drowsiness/train.py`          | Train MobileNetV2 on Driver-Drowsiness-Dataset dataset   |
| `deep_learning/mobilenetv2/mrl_eye/train.py`                    | Train MobileNetV2 on MRL-Eye dataset                     |

## Model Details

### MobileNetV2 (MRL-Eye)

**Training Dataset**: MRL-Eye-Dataset

- Train: ~7,000 images
- Validation: ~1,000 images
- Test: ~16,500 images

**Performance** (on test set):

- Accuracy: 98.53%
- Precision: 99.36%
- Recall: 97.65%
- F1-Score: 98.50%
- FAR (False Alarm Rate): 0.62%
- FRR (False Rejection Rate): 2.35%

**Input**: 224×224 RGB images
**Output**: Binary classification (awake/sleepy)

### MobileNetV2 (Driver-Drowsiness)

**Training Dataset**: Driver-Drowsiness-Dataset

- Train: ~30,000 images
- Validation: ~7,000 images
- Test: ~16,500 images
- Test: Evaluated on test split (`data/Driver-Drowsiness-Dataset/test`)

**Performance** (on test set):

- Accuracy: 99.94%
- Precision: 99.97%
- Recall: 99.90%
- F1-Score: 99.93%
- FAR (False Alarm Rate): 0.03%
- FRR (False Rejection Rate): 0.10%

**Confusion Matrix Breakdown**:

- True Negatives (TN): 3,352
- False Positives (FP): 1
- False Negatives (FN): 3
- True Positives (TP): 2,915

**Input**: 224×224 RGB images
**Output**: Binary classification (Drowsy / Non Drowsy)

## Dependencies

See `requirements.txt` for complete list. Key packages:

- TensorFlow/Keras
- OpenCV
- MediaPipe
- scikit-learn
- pandas
- numpy
- Random Forest

### 5. Deep Learning

Uses eye images directly to learn visual features.

Models:

- CNN
- MobileNetV2

### 6. Time-Series Analysis

Analyses eye-state predictions over time rather than relying on a single frame.

Features include:

- Blink frequency
- Eye closure duration
- Consecutive closed-eye frames
- PERCLOS

While frame-by-frame classifiers and geometric ratios (EAR/MAR) can determine eye and mouth states at a single millisecond, they cannot differentiate between a normal, healthy blink and a dangerous micro-sleep on their own. The **Time-Series Module** (`time_series/`) bridges this gap by evaluating behavioral patterns across a rolling historical window.

### Core Architecture & Sliding Window

The system maintains a rolling buffer (queue) of recent frame predictions and geometric states (typically tracking the last 30 to 60 frames). Instead of reacting instantly to a single closed-frame anomaly, the temporal engine analyzes sequences of data to detect sustained fatigue.

````text
Incoming Frame Predictions
      ↓
[ Rolling Frame Buffer (Queue) ]
      ├──> Closure Duration Calculation
      ├──> Blink Frequency Counter
      └──> PERCLOS Evaluation
      ↓
Temporal Drowsiness Flag

### 7. Drowsiness Detection

Combines temporal information to distinguish between:

```text
Normal blinking
      vs.
Prolonged/repeated eye closure
      ↓
Potential drowsiness / micro-sleep
````

### 8. Real-Time System

The final application processes a webcam stream:

```text
Webcam
   ↓
Face Detection
   ↓
Eye Detection
   ↓
Eye-State Prediction
   ↓
Temporal Analysis
   ↓
Drowsiness Detection
   ↓
Warning / Alarm
```

### 9. Evaluation

Compare the different approaches using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Inference time

The project can compare:

| Approach      | Input                | Model             | Task       |
| ------------- | -------------------- | ----------------- | ---------- |
| Classical ML  | EAR/features         | SVM               | Eye state  |
| Classical ML  | EAR/features         | Random Forest     | Eye state  |
| Deep Learning | Eye images           | CNN               | Eye state  |
| Deep Learning | Eye images           | MobileNetV2       | Eye state  |
| Deep Learning | Face images          | MobileNetV2       | Face state |
| Final system  | Eye states over time | Temporal analysis | Drowsiness |

## Course Module Mapping

| Course Module                      | Project Application                             |
| ---------------------------------- | ----------------------------------------------- |
| Module 3 — ML & Deep Learning      | SVM, Random Forest, CNN, MobileNetV2            |
| Module 4 — Computer Vision         | Face/eye detection, landmarks, image processing |
| Module 5 — Time Series             | Blink patterns, closure duration, PERCLOS       |
| Module 8 — Evolutionary Algorithms | Optional threshold/model optimization           |
| Module 6 — NLP                     | Not required                                    |
| Module 7 — Reinforcement Learning  | Not required                                    |

## Core Project Idea

> **Classify the driver's eye state using classical machine learning and deep learning, then analyse eye behaviour over time to distinguish normal blinking from prolonged eye closure associated with drowsiness or micro-sleep.**

## Final System

The final system should be able to display information such as:

```text
--------------------------------
     DRIVER MONITORING SYSTEM
--------------------------------

Eye State:       CLOSED
PERCLOS:         45%
Closure Time:    2.1 seconds

Drowsiness:      HIGH
Status:          DROWSY

       ⚠ WARNING ⚠
          WAKE UP!
--------------------------------
```

### Optional Future Extensions

_(Note: Yawning detection has been promoted to a core feature)_

- Head-pose estimation
- Facial expression analysis
- Genetic Algorithm for threshold optimization
- Mobile/Raspberry Pi deployment
- Additional driver datasets
- Multi-modal drowsiness detection

## Development Order

```text
1. Dataseta
      ↓
2. Preprocessing
      ↓
3. Computer Vision
      ↓
4. Classical ML
      ↓
5. Deep Learning
      ↓
6. Model Evaluation & Comparison
      ↓
7. Time-Series Analysis
      ↓
8. Drowsiness Detection
      ↓
9. Real-Time Webcam System
      ↓
10. Final Evaluation & Results
      ↓
11. Documentation
```
