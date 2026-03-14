import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from src.predict import predict_emotion
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
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

# -------- Upload Audio --------

audio = st.file_uploader("Upload a .wav audio file", type=["wav"])

# -------- Microphone Recording --------

st.subheader("🎤 Or Record Voice")

if st.button("Start Recording (3 sec)"):

    duration = 3
    samplerate = 22050

    recording = sd.rec(int(duration * samplerate),
                       samplerate=samplerate,
                       channels=1)

    sd.wait()

    sf.write("temp.wav", recording, samplerate)

    st.success("Recording completed!")

    audio = "temp.wav"

# -------- Prediction --------

if audio is not None:

    # Save uploaded file
    if type(audio) != str:
        with open("temp.wav", "wb") as f:
            f.write(audio.read())

    st.audio("temp.wav")

    emotion, conf = predict_emotion("temp.wav")

    emoji = emotion_emojis.get(emotion, "")

    st.subheader("Prediction Result")

    st.markdown(
        f"""
        ## Emotion Detected: **{emotion.upper()}** {emoji}
        """
    )

    st.progress(int(conf * 100))

    st.write("Confidence:", round(conf * 100, 2), "%")

    # -------- Waveform --------

    st.subheader("Audio Waveform")

    audio_data, sr = librosa.load("temp.wav")

    fig, ax = plt.subplots()

    librosa.display.waveshow(audio_data, sr=sr)

    ax.set_title("Waveform")

    st.pyplot(fig)

    # -------- Spectrogram --------

    st.subheader("Mel Spectrogram")

    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)

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

    st.info("Upload or record audio to detect emotion.")

st.divider()

st.caption("AI Project: Speech Emotion Recognition using Deep Learning")
