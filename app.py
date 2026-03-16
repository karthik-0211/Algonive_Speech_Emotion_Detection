import streamlit as st
from src.predict import predict_emotion
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

st.set_page_config(page_title="Speech Emotion Recognition AI", layout="wide")

st.title("🎙 Speech Emotion Recognition AI")
st.write("Detect human emotions from voice using Deep Learning")

# -------- Emotion Emoji Mapping --------

emotion_emojis = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😊",
    "sad": "😢",
    "angry": "😡",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😲"
}

# -------- Cache Prediction (Faster App) --------

@st.cache_resource
def load_model():
    return predict_emotion

predict = load_model()

# -------- Upload Audio --------

audio = st.file_uploader("Upload a .wav audio file", type=["wav"])

# -------- Prediction --------

if audio is not None:

    # Save uploaded file
    with open("temp.wav", "wb") as f:
        f.write(audio.read())

    st.audio("temp.wav")

    # Predict emotion
    emotion, conf = predict("temp.wav")

    emoji = emotion_emojis.get(emotion, "")

    st.subheader("Prediction Result")

    st.markdown(
        f"""
        ## Emotion Detected: **{emotion.upper()}** {emoji}
        """
    )

    st.progress(int(conf * 100))

    st.write("Confidence:", round(conf * 100, 2), "%")

    # -------- Load Audio --------

    y, sr = librosa.load("temp.wav", sr=22050)

    # -------- Waveform --------

    st.subheader("Audio Waveform")

    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")

    st.pyplot(fig)

    # -------- Spectrogram --------

    st.subheader("Mel Spectrogram")

    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig2, ax2 = plt.subplots()

    img = librosa.display.specshow(
        mel_db,
        x_axis="time",
        y_axis="mel",
        sr=sr
    )

    plt.colorbar(img, ax=ax2)

    ax2.set_title("Mel Spectrogram")

    st.pyplot(fig2)

else:

    st.info("Upload a .wav file to detect emotion.")

st.divider()

st.caption("AI Project: Speech Emotion Recognition using Deep Learning")
