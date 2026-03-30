# 🎙 Audio Deepfake Detection using CNN + LSTM

## 📌 Overview

This project focuses on detecting whether an audio sample is **real or deepfake** using a hybrid deep learning model that combines **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks.

The system processes short-duration (2-second) audio clips and classifies them into:

* ✅ Real (Bonafide)
* 🚨 Fake (Deepfake)

---

## 🚀 Features

* 🎯 End-to-end deepfake detection pipeline
* 🧠 Hybrid CNN + LSTM architecture (spectral + temporal learning)
* ⚡ Lightweight and CPU-friendly model
* 📊 Train, validate, and test workflow
* 🎙 Real-time detection using Streamlit UI
* 📁 Structured dataset handling (train/validation/test split)

---

## 📂 Dataset Structure

```
dataset/
│
├── training/
│   ├── real/
│   └── fake/
│
├── validation/
│   ├── real/
│   └── fake/
│
├── testing/
│   ├── real/
│   └── fake/
```

Each folder contains **2-second WAV audio clips**.

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/Deepfake_Audio.git
cd Deepfake_Audio
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Run the training script:

```
python src/train.py
```

This will:

* Load dataset
* Train CNN + LSTM model
* Save model in `models/` directory

---

## 📊 Model Evaluation

Run:

```
python src/evaluate.py
```

Output:

```
Test Accuracy: 74 %
```

---

## 🎯 Run the Application (UI)

Start the Streamlit app:

```
streamlit run app.py
```

Then:

* Upload a `.wav` file
* Get prediction:

  * ✅ Real Audio
  * 🚨 Fake Audio

---

## 🧠 Model Architecture

```
Audio Input
   ↓
Mel Spectrogram
   ↓
CNN (Feature Extraction)
   ↓
LSTM (Temporal Modeling)
   ↓
Fully Connected Layer
   ↓
Output (Real / Fake)
```

---


## 🛠 Technologies Used

* Python
* PyTorch
* Librosa
* NumPy
* Scikit-learn
* Streamlit

