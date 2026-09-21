# Real-Time Driver Drowsiness / Micro-Sleep Detection

## 1. Project Idea

The project aims to develop an **AI-based real-time driver drowsiness detection system**.

A camera monitors the driver's face and eyes. The system identifies whether the driver's eyes are **open or closed**, and then analyses the eye behaviour over a period of time.

The key idea is:

> **A single closed-eye frame may indicate a normal blink, but prolonged or repeated eye closure may indicate drowsiness.**

When the system detects a high level of drowsiness, it can provide an **audio/visual warning** to the driver.

---

# 2. Overall Concept

```text
Camera
   ↓
Face & Eye Detection
   ↓
Eye State Detection
(Open / Closed)
   ↓
Temporal Analysis
   ↓
Drowsiness Detection
(Awake / Drowsy)
   ↓
Warning / Alarm
```

---

# 3. Main Components

## A. Dataset

Use the **MRL Eye Dataset** from Kaggle.

The dataset will be used mainly to train a model that recognises:

- Open eyes
- Closed eyes

Dataset:

https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset

Use the **Driver Drowsiness Dataset** from Kaggle.

The dataset will be used mainly to train a model that recognises:

- Drowsy
- Non Drowsy

Dataset:

https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd

---

## B. Computer Vision

Use computer vision to identify the driver's:

- Face
- Eyes
- Eye region

Tools:

- MediaPipe

The purpose is to extract the relevant eye and face information from the live camera feed.

---

## C. Classical Machine Learning

Use **Eye Aspect Ratio (EAR)** as a traditional computer-vision feature.

Concept:

```text
Face
 ↓
Facial landmarks
 ↓
Eye landmarks
 ↓
EAR
 ↓
Machine Learning Classifier
 ↓
Open / Closed
```

Possible classifiers:

- SVM
- Random Forest

This gives you the **classical ML component** of the project.

---

## D. Deep Learning

Use a lightweight CNN such as **MobileNetV2** to classify eye images.

Concept:

```text
Eye Image
   ↓
CNN / MobileNetV2
   ↓
Open / Closed
```

The CNN learns visual features automatically instead of relying only on manually calculated EAR features.

---

## E. Time-Series Analysis

This is the component that converts **eye-state detection into drowsiness detection**.

The system continuously records predictions:

```text
Open
Open
Open
Closed
Closed
Closed
Closed
Open
Open
```

It then analyses:

- How long the eyes remain closed
- How frequently the eyes close
- Percentage of time the eyes are closed
- Repeated/prolonged eye closures

One useful measure is **PERCLOS**.

```text
Eye-state predictions
        ↓
Time-series data
        ↓
PERCLOS / closure duration
        ↓
Drowsiness level
```

---

# 4. Final System Architecture

```text
                  ┌─────────────┐
                  │   Camera    │
                  └──────┬──────┘
                         ↓
                ┌─────────────────┐
                │ Face/Eye        │
                │ Detection       │
                └────────┬────────┘
                         ↓
                   ┌────────────┐
                   │ Eye Region │
                   └─────┬──────┘
                         ↓
             ┌───────────┴───────────┐
             ↓                       ↓
       EAR Features             CNN Model
             ↓                       ↓
        SVM / RF                Open/Closed
             ↓                       ↓
             └───────────┬───────────┘
                         ↓
                  Time-Series
                    Analysis
                         ↓
              ┌──────────────────┐
              │ Drowsiness Level │
              └────────┬─────────┘
                       ↓
                ┌──────┴──────┐
                ↓             ↓
              Awake         Drowsy
                              ↓
                           Alarm
```

---

# 5. Main Project Stages

### Stage 1 — Dataset Preparation

Understand and prepare the MRL dataset.

### Stage 2 — Classical ML

Use EAR features with:

- SVM
- Random Forest

### Stage 3 — Deep Learning

Train a CNN/MobileNetV2 for eye-state classification.

### Stage 4 — Model Comparison

Compare classical ML and deep learning performance.

### Stage 5 — Real-Time Detection

Connect the selected model to a webcam.

### Stage 6 — Time-Series Analysis

Analyse eye behaviour over time using measures such as:

- Eye closure duration
- PERCLOS
- Consecutive closed frames

### Stage 7 — Drowsiness Alert

If the system detects prolonged/repeated eye closure, trigger an alert.

---

# 6. What You Will Compare

Your project can have three main approaches:

| Approach      | Input                     | Method               | Output       |
| ------------- | ------------------------- | -------------------- | ------------ |
| Classical ML  | EAR features              | SVM / Random Forest  | Open/Closed  |
| Deep Learning | Eye images                | CNN / MobileNetV2    | Open/Closed  |
| Final System  | Eye predictions over time | Time-Series Analysis | Awake/Drowsy |

This comparison will make the project more meaningful academically.

---

# 7. How It Relates to Your Modules

### Module 3 — Machine Learning / ANN / Deep Learning

- SVM
- Random Forest
- Neural Networks
- CNN
- Transfer Learning

### Module 4 — Computer Vision

- Face detection
- Eye detection
- Facial landmarks
- Eye image processing
- CNN-based image classification

### Module 5 — Time Series Analysis

- Continuous eye-state predictions
- Sliding windows
- Eye closure duration
- PERCLOS
- Temporal drowsiness detection

### Module 8 — Evolutionary Algorithms

This can be an **optional extension**, where you use a Genetic Algorithm to optimise things such as:

- Drowsiness threshold
- PERCLOS threshold
- Time window
- EAR threshold

---

# 8. Expected Final Demo

The final demonstration could look like:

```text
        LIVE CAMERA FEED

Eye State:       OPEN
PERCLOS:         12%
Drowsiness:      LOW
Status:          AWAKE
```

When the person keeps their eyes closed:

```text
        LIVE CAMERA FEED

Eye State:       CLOSED
PERCLOS:         48%
Drowsiness:      HIGH
Status:          DROWSY

       ⚠ WARNING ⚠
       WAKE UP!
```

---

# 9. The Core Idea in One Sentence

> **Develop a real-time computer vision system that classifies eye states using classical ML and deep learning, then analyses eye behaviour over time to distinguish normal blinking from potential driver drowsiness and trigger an alert.**

---

# 10. Recommended Scope

For the **main project**, I would keep the scope to:

```text
MRL Dataset
     ↓
Eye Classification
     ↓
SVM / Random Forest
     ↓
CNN / MobileNetV2
     ↓
Model Comparison
     ↓
Webcam
     ↓
Eye-State Tracking
     ↓
PERCLOS / Temporal Analysis
     ↓
Drowsiness Alert
```

Then treat **Genetic Algorithm optimisation, yawning detection, head-pose detection, Raspberry Pi deployment, etc. as future work**, rather than making the initial project unnecessarily large.
