import tensorflow as tf
import librosa
import numpy as np
import joblib

model = tf.saved_model.load("models/emotion_saved_model")
infer = model.signatures["serving_default"]

scaler = joblib.load("models/scaler.pkl")

emotion_labels = [
    "neutral","calm","happy","sad",
    "angry","fear","disgust","surprise"
]

def extract_features(file):

    audio, sr = librosa.load(file, duration=2.5, offset=0.6)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma = np.mean(chroma.T, axis=0)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel = np.mean(mel.T, axis=0)

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast = np.mean(contrast.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr = np.mean(zcr)

    features = np.hstack([mfcc, chroma, mel, contrast, zcr])

    return features


def predict_emotion(file):

    feature = extract_features(file)

    feature = feature.reshape(1,-1)

    feature = scaler.transform(feature)

    prediction = infer(tf.constant(feature.astype(np.float32)))

    prediction = list(prediction.values())[0].numpy()

    emotion = emotion_labels[np.argmax(prediction)]

    confidence = float(np.max(prediction))

    return emotion, confidence