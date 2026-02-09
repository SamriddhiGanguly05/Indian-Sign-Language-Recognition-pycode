# SPARC: Smart Perception & Assistive Reality Companion
**(Indian Sign Language Recognition System)**

---

**Team:** NAMO NIRVANA (Team ID: 94943)  
**Problem Statement ID:** SIH25247  
**Theme:** Miscellaneous | **Category:** Hardware  
**Smart India Hackathon 2025**

---

## 👥 Team Members

*   **Harsh Yadav** (Team Lead)
*   Avishkar Jaiswal
*   Samriddhi Ganguly
*   Samyak Jain
*   Harshit Singh
*   Thakur Akshayakumar Raj

---

## 🚀 Project Overview

**SPARC** is a specialized **Indian Sign Language (ISL) Recognition System**. It is designed to interpret the complex, dynamic, and bimanual gestures unique to ISL, translating them into text/speech in real-time.

Unlike generic sign language models, SPARC focuses specifically on the **temporal dynamics** of ISL—where the *movement* is just as important as the *pose*.

### ❓ Why This Matters
*   **5 Million+** deaf individuals in India.
*   **< 250** certified ISL interpreters.
*   **The Gap:** Most existing solutions focus on static alphabets (A-Z). ISL is a full language with grammar and continuous motion. **SPARC solves for words and sentences.**

---

## 🧠 Advanced ISL Model Architecture

This repository implements a multi-stage deep learning pipeline, evolving from standard LSTMs to State-of-the-Art (SOTA) architectures.

### 1. Data Processing Pipeline (`helper_functions.py`)
*   **Input normalization:** 45 Frames per video (Fixed Size).
*   **Feature Extraction:** MediaPipe Holistic extracts **258 Keypoints** per frame:
    *   **Pose (132):** Body orientation & arm movement.
    *   **Left Hand (63) + Right Hand (63):** Fine-grained finger articulation.
*   **Augmentation Strategy:** To ensure robustness, we implement:
    *   Gaussian Noise Injection (Simulating sensor noise).
    *   Spatial Scaling (Handling different body sizes).
    *   Temporal Warping (Handling different signing speeds).

### 2. Model Evolution
We researched and implemented three distinct tiers of models:

#### 🟢 Tier 1: Baseline LSTM (`deploy-code.py`)
*   **Structure:** 3 stacked LSTM layers (64-128-256 units) + Dense classification head.
*   **Use Case:** Fast, lightweight recognition for basic vocabulary.
*   **Current Deployment:** Optimized for low-latency CPU inference.

#### 🔵 Tier 2: Regularized Deep LSTM (`train-improved-model.py`)
*   **Improvements:** Added Batch Normalization, Dropout (0.3), and L2 Regularization.
*   **Activation:** Switched to `tanh` for stable gradient flow.
*   **Result:** Higher accuracy on unseen test subjects.

#### 🔴 Tier 3: Ultra-Advanced Two-Stream Network (`train-ultra-advanced-model.py`)
*   **SOTA Architecture:** A hybrid **Spatial-Temporal** design.
*   **Stream 1 (Spatial):** LSTM with Self-Attention mechanisms to focus on hand-face interaction.
*   **Stream 2 (Temporal):** Temporal Convolutional Networks (TCN) to capture fast motion dynamics.
*   **Fusion:** Attention-based fusion layer combines both streams for the final prediction.

---

## ⚡ Real-Time Geometric Detector
*For instant feedback on static cultural signs.*

Alongside the AI model, we engineered a **Rule-Based Heuristic Engine** (`realtime-detection.py`) for specific geometric ISL gestures:
*   **Namaste:** Calculates wrist-to-wrist distance and palm symmetry.
*   **I am Indian:** Triangulates Hand-Eyebrow-Shoulder positions.
*   **Water/Doctor/Home:** Custom geometric signatures.

---

## 📊 Dataset & Performance

*   **Dataset:** [INCLUDE 50](https://zenodo.org/record/4010759) + **Custom NAMO NIRVANA Dataset**.
*   **Vocabulary:** 16 Classes (Hello, Thank you, Please, Good Morning, etc.).
*   **Training Scale:** 1000+ Videos with 5x Augmentation.
*   **Accuracy:**
    *   Validation: **74.6%**
    *   Real-Time Test: **84.0%**

---

## 📥 Installation

1.  **Clone the repository**
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🖥️ Usage Guide

### 1. Run the Main ISL Model (Recommended)
This uses the LSTM network to recognize dynamic words ("Hello", "How are you").
```bash
python deploy-code.py
```

### 2. Run Geometric Detection Demo
For checking specific static signs (Namaste, Indian, etc.).
```bash
python realtime-detection.py
```
*(Or use `RUN-REALTIME-DEMO.bat` on Windows)*

### 3. Train Your Own Model
If you want to add new words to the ISL dictionary:
```bash
# Prepare data in 'training-data/' folder
python train-improved-model.py --epochs 100 --augment 5
```

---

*Developed by Team NAMO NIRVANA for Smart India Hackathon 2025.*
