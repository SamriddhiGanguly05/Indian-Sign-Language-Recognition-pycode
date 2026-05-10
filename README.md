# SPARC: Smart Perception & Assistive Reality Companion
**(Indian Sign Language Recognition System)**

---

**Team:** NAMO NIRVANA (Team ID: 94943)  
**Problem Statement ID:** SIH25247  
**Theme:** Miscellaneous | **Category:** Hardware  
**Smart India Hackathon 2025**

---

## 👥 Team Members

* **Harsh Yadav** (Team Lead)
* Avishkar Jaiswal
* Samriddhi Ganguly
* Samyak Jain
* Harshit Singh
* Thakur Akshayakumar Raj

---

## 🚀 Project Overview

**SPARC** is a specialized **Indian Sign Language (ISL) Recognition System** designed to interpret complex and dynamic ISL gestures in real-time using **Computer Vision** and **Deep Learning**.

Unlike traditional sign language recognition systems focused only on static alphabets, SPARC captures the **temporal dynamics of gestures**, enabling recognition of complete words and sentence-level motion patterns.

### ❓ Why This Matters

* **5 Million+** deaf individuals in India.
* **< 250** certified ISL interpreters.
* Existing solutions mainly focus on **static alphabet classification**.
* **SPARC solves real-time dynamic word and sentence recognition**.

---

# 🏗️ SPARC Architecture

<p align="center">
  <img src="SPARC architecture.png" alt="SPARC Architecture Diagram" width="100%">
</p>

---

## 🧠 Advanced ISL Model Architecture

This repository implements a **multi-stage deep learning pipeline** evolving from standard LSTMs to advanced spatial-temporal architectures.

---

### 1️⃣ Data Processing Pipeline (`helper_functions.py`)

#### 🔹 Input Standardization
* 45 Frames per video sequence
* Fixed-length temporal input pipeline

#### 🔹 Feature Extraction using MediaPipe Holistic
Extracts **258 Keypoints per frame**

| Feature Type | Keypoints |
|---|---|
| Pose Landmarks | 132 |
| Left Hand Landmarks | 63 |
| Right Hand Landmarks | 63 |

#### 🔹 Data Augmentation
To improve robustness:
* Gaussian Noise Injection
* Spatial Scaling
* Temporal Warping

---

### 2️⃣ Model Evolution

We implemented three tiers of deep learning architectures.

---

#### 🟢 Tier 1: Baseline LSTM (`deploy-code.py`)

* 3 stacked LSTM layers
* Hidden Units: 64 → 128 → 256
* Lightweight low-latency inference
* Optimized for CPU deployment

---

#### 🔵 Tier 2: Regularized Deep LSTM (`train-improved-model.py`)

Enhancements:
* Batch Normalization
* Dropout (0.3)
* L2 Regularization
* `tanh` activation for stable gradients

Result:
* Better generalization on unseen users

---

#### 🔴 Tier 3: Ultra-Advanced Two-Stream Network (`train-ultra-advanced-model.py`)

Hybrid Spatial-Temporal architecture:

### Stream 1 — Spatial Modeling
* LSTM + Self-Attention
* Captures hand-face interaction

### Stream 2 — Temporal Modeling
* Temporal Convolutional Networks (TCN)
* Captures gesture motion dynamics

### Fusion Layer
* Attention-based stream fusion
* Final gesture classification

---

# ⚡ Real-Time Geometric Detection Engine

For ultra-fast recognition of static cultural gestures.

Implemented in:
```python
realtime-detection.py
```

Supported gestures:
* Namaste
* I am Indian
* Water
* Doctor
* Home

Detection uses:
* Wrist symmetry
* Hand orientation
* Geometric landmark relationships

---

# 📊 Dataset & Performance

| Metric | Value |
|---|---|
| Dataset | INCLUDE 50 + Custom Dataset |
| Vocabulary Size | 16 Classes |
| Training Videos | 1000+ |
| Data Augmentation | 5x |
| Validation Accuracy | 74.6% |
| Real-Time Accuracy | 84.0% |

---

# 📥 Installation

## Clone Repository
```bash
git clone <repo-url>
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

---

# 🖥️ Usage Guide

## ▶️ Run Main Dynamic ISL Recognition
```bash
python deploy-code.py
```

---

## ▶️ Run Real-Time Geometric Detection
```bash
python realtime-detection.py
```

Windows Shortcut:
```bash
RUN-REALTIME-DEMO.bat
```

---

## ▶️ Train Custom Model
```bash
python train-improved-model.py --epochs 100 --augment 5
```

---

# 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe Holistic
* NumPy
* LSTM
* Temporal CNN (TCN)
* Raspberry Pi

---

# 🌍 Applications

* Assistive Communication
* Accessibility Technology
* Human-Computer Interaction
* Smart Wearables
* Real-Time Gesture Interfaces

---

# 📌 Future Improvements

* Sentence-level ISL Translation
* Transformer-based Gesture Modeling
* Edge AI Optimization
* Mobile Deployment
* Multilingual Speech Output

---

Developed by **Team NAMO NIRVANA**.
