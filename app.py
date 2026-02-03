import streamlit as st
import cv2
import joblib
import numpy as np
import webbrowser
import random
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="AI Emotion Music Player", layout="wide")
st.title("🎵 AI Hybrid Emotion Music System")

# Load Models
@st.cache_resource
def get_resources():
    rf = joblib.load("rf_model.pkl")
    svm = joblib.load("svm_model.pkl")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return rf, svm, cascade

try:
    rf_model, svm_model, face_cascade = get_resources()
except:
    st.error("Models not found! Please run train_app.py first.")
    st.stop()

music_map = {
    "happy": "https://www.youtube.com/results?search_query=happy+songs",
    "sad": "https://www.youtube.com/results?search_query=sad+lofi+songs",
    "angry": "https://www.youtube.com/results?search_query=calm+music",
    "neutral": "https://www.youtube.com/results?search_query=focus+music"
}

class EmotionTransformer(VideoTransformerBase):
    def __init__(self): self.current_emotion = "neutral"
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            try:
                face_img = img[y:y+h, x:x+w]
                emb = DeepFace.represent(face_img, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
                self.current_emotion = rf_model.predict([emb])[0]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, self.current_emotion.upper(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except: pass
        return img

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📷 Live Feed")
    ctx = webrtc_streamer(key="emo", video_transformer_factory=EmotionTransformer)

with col2:
    st.subheader("📊 Analysis")
    if ctx.video_transformer:
        emo = ctx.video_transformer.current_emotion
        st.info(f"Detected: **{emo.upper()}**")
        if st.button("🎧 Play Music"):
            webbrowser.open(music_map.get(emo, music_map["neutral"]))
