import streamlit as st
import torch
import librosa
from src.model import CNN_LSTM_Model

st.title("🎙 Audio Deepfake Detector")

model = CNN_LSTM_Model()
model.load_state_dict(torch.load("models/cnn_lstm_model.pth"))
model.eval()

file = st.file_uploader("Upload WAV file", type=["wav"])

if file:
    audio, sr = librosa.load(file, sr=16000)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel = librosa.power_to_db(mel)

    x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()

    output = model(x)
    pred = torch.argmax(output, dim=1).item()
    conf = torch.softmax(output, dim=1)[0][pred].item()

    if pred == 0:
        st.success(f"✅ Real Audio ({conf*100:.2f}%)")
    else:
        st.error(f"🚨 Fake Audio ({conf*100:.2f}%)")