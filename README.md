# 🎙️ Speech Emotion Recognition AI

> A deep learning-based system that detects human emotions from voice and audio signals using **Audio Signal Processing**, **Machine Learning**, and **Deep Learning**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-Educational-green)

### 🌐 [Live Demo → algonive-speech-emotion-detection.onrender.com](https://algonive-speech-emotion-detection.onrender.com)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Emotions Detected](#-emotions-detected)
- [Dataset](#-dataset)
- [Tech Stack](#️-tech-stack)
- [Audio Features](#-audio-features-used)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation & Setup](#️-installation--setup)
- [How to Use](#-how-to-use)
- [Prediction Pipeline](#-prediction-pipeline)
- [Deployment](#-deployment)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🧾 Overview

This project takes speech input from an uploaded `.wav` file or real-time microphone recording, extracts important audio features, and predicts the emotion expressed in the speech.

It provides an interactive **Streamlit dashboard** featuring:
- Audio upload and live microphone recording
- Emotion prediction with confidence score and emoji reaction
- Waveform and mel spectrogram visualization

---

## 📌 Features

| Feature | Description |
|--------|-------------|
| 🎤 Real-time Recording | Detect emotions from live microphone input |
| 📂 File Upload | Support for `.wav` audio file uploads |
| 😊 Emoji Reactions | Visual emotion indicators |
| 📊 Waveform Plot | Visual representation of the audio signal |
| 🎼 Mel Spectrogram | Frequency-time visualization of speech |
| 🤖 Deep Learning | DNN-based emotion classification |
| ⚡ Streamlit UI | Fast, interactive web application |
| 📈 High Accuracy | ~92–93% validation accuracy on RAVDESS |

---

## 🧠 Emotions Detected

The model classifies speech into **8 emotion categories**:

| # | Emotion | # | Emotion |
|---|---------|---|---------|
| 1 | 😐 Neutral | 5 | 😡 Angry |
| 2 | 😌 Calm | 6 | 😨 Fear |
| 3 | 😊 Happy | 7 | 🤢 Disgust |
| 4 | 😢 Sad | 8 | 😲 Surprise |

---

## 📂 Dataset

This project uses the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

- 📎 **Dataset Link:** [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)
- 🎭 Contains emotional speech recordings from multiple professional actors
- 🔬 Widely used benchmark for Speech Emotion Recognition (SER) research

### Emotion Code Mapping

| Code | Emotion  | Code | Emotion  |
|------|----------|------|----------|
| 01   | Neutral  | 05   | Angry    |
| 02   | Calm     | 06   | Fear     |
| 03   | Happy    | 07   | Disgust  |
| 04   | Sad      | 08   | Surprise |

---

## 🛠️ Tech Stack

| Category | Libraries / Tools |
|----------|-------------------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow / Keras |
| Audio Processing | Librosa, SoundDevice, SoundFile |
| Data & ML | NumPy, Scikit-learn, Joblib |
| Visualization | Matplotlib |
| Web App | Streamlit |

---

## 🧪 Audio Features Used

The model was trained on a combination of audio features for robust performance:

| Feature | Purpose |
|---------|---------|
| **MFCC** | Mel Frequency Cepstral Coefficients — captures timbre and speech texture |
| **Chroma Features** | Pitch class distribution across the chromatic scale |
| **Mel Spectrogram** | Frequency-time representation of energy |
| **Spectral Contrast** | Difference in energy between peaks and valleys |
| **Zero Crossing Rate** | Rate at which the signal changes sign — reflects speech energy |

---

## 🚀 Model Architecture

The model is a **Deep Neural Network (DNN)** built with TensorFlow/Keras.

```
Input (188 features)
        │
        ▼
Dense(512) → BatchNormalization → Dropout
        │
        ▼
Dense(256) → BatchNormalization → Dropout
        │
        ▼
Dense(128) → Dropout
        │
        ▼
Dense(64)
        │
        ▼
Dense(8, Softmax)  ← Output: 8 Emotion Classes
```

**Key design choices:**
- Batch Normalization for stable, faster training
- Dropout layers to prevent overfitting
- Softmax activation for multi-class probability output

---

## 📈 Training Strategy

| Technique | Details |
|-----------|---------|
| **Data Augmentation** | Noise injection, audio time-shifting |
| **Feature Normalization** | StandardScaler applied to all features |
| **Early Stopping** | Monitors validation loss to prevent overfitting |
| **Train/Test Split** | Standard split for unbiased evaluation |
| **Validation Monitoring** | Tracks accuracy and loss per epoch |

---

## ✅ Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | ~92% – 96% |
| Validation / Test Accuracy | ~92% – 93% |

> These results are strong for a Speech Emotion Recognition system trained on RAVDESS.

---

## 📁 Project Structure

```
speech-emotion-recognition-ai/
│
├── app/
│   └── dashboard.py          # Streamlit web application
│
├── models/
│   ├── emotion_saved_model/
│   │   ├── saved_model.pb    # Trained TensorFlow model
│   │   └── variables/        # Model weights
│   └── scaler.pkl            # Fitted StandardScaler
│
├── src/
│   └── predict.py            # Feature extraction & prediction logic
│
├── requirements.txt          # Python dependencies
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/karthik-0211/speech-emotion-recognition-ai.git
cd speech-emotion-recognition-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
python -m streamlit run app/dashboard.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 💡 How to Use

### Option 1: Upload an Audio File

1. Navigate to the **Upload** section in the dashboard
2. Upload a `.wav` audio file
3. View the predicted emotion, confidence score, waveform, and spectrogram

### Option 2: Record Live Audio

1. Navigate to the **Record** section
2. Click the microphone button and speak for a few seconds
3. The system will record your voice and display the emotion prediction

---

## 🧠 Prediction Pipeline

```
Audio Input (.wav / Microphone)
        │
        ▼
  Feature Extraction
  (MFCC, Chroma, Mel, ZCR, Contrast)
        │
        ▼
  Feature Scaling (StandardScaler)
        │
        ▼
  Deep Learning Model (DNN)
        │
        ▼
  Emotion Prediction + Confidence Score
        │
        ▼
  Dashboard Visualization
  (Waveform, Spectrogram, Emoji)
```

---

## 📓 Training Workflow (Google Colab)

1. Mount Google Drive and load the RAVDESS dataset
2. Extract audio features (MFCC, Chroma, Mel, etc.)
3. Apply data augmentation (noise, shifting)
4. Normalize features with `StandardScaler`
5. Train the DNN model with early stopping
6. Save the model in TensorFlow `SavedModel` format
7. Save the scaler using `joblib`
8. Use saved model + scaler in the local Streamlit app

---

## 🌐 Deployment

The app can be deployed on any of the following platforms:

| Platform | Notes |
|----------|-------|
| **Streamlit Cloud** | Set entry file as `app/dashboard.py` |
| **Render** | Easy Python web app hosting |
| **Hugging Face Spaces** | Great for ML demo projects |
| **GitHub Pages** | Static hosting (frontend only) |

---

## 🔮 Future Improvements

- [ ] Real-time streaming emotion detection
- [ ] Noise reduction for more robust predictions in noisy environments
- [ ] Support for additional datasets (CREMA-D, TESS)
- [ ] CNN / LSTM / Transformer-based model architectures
- [ ] Multilingual speech emotion recognition
- [ ] Cloud-hosted AI deployment with REST API

---

## 🎯 Use Cases

- Human–computer interaction with emotion awareness
- Emotion-aware virtual assistants
- AI-powered interview analysis tools
- Audio signal processing research
- Internship and portfolio showcase project

---

## 📜 License

This project is intended for **educational and portfolio use only**.

---

## 👨‍💻 Author

**Karthik**  
🔗 GitHub: [https://github.com/karthik-0211](https://github.com/karthik-0211)

---

> ⭐ **If you found this project useful, consider giving it a star on GitHub!**
