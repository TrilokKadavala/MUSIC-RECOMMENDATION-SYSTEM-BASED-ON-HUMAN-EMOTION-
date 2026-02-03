import streamlit as st
import cv2
import joblib
import numpy as np
import webbrowser
import random
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Page Configuration
st.set_page_config(page_title="AI Emotion Music Player", layout="wide")

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    rf = joblib.load("rf_model.pkl")
    svm = joblib.load("svm_model.pkl")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return rf, svm, face_cascade

rf_model, svm_model, face_cascade = load_models()

# Mapping Data
music_map = {
    "happy": "https://www.youtube.com/results?search_query=happy+songs",
    "sad": "https://www.youtube.com/results?search_query=sad+lofi+songs",
    "angry": "https://www.youtube.com/results?search_query=calm+music",
    "neutral": "https://www.youtube.com/results?search_query=focus+music"
}

activity_map = {
    "happy": ["Smile and enjoy your day.", "Share happiness!"],
    "sad": ["Take rest and relax.", "Listen to calm music."],
    "angry": ["Take deep breaths.", "Go for a short walk."],
    "neutral": ["Keep focusing on your work.", "Drink water."]
}

# ==========================
# Video Processing Class
# ==========================
class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.current_emotion = "Detecting..."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            try:
                # Feature Extraction
                embedding = DeepFace.represent(face_img, model_name="VGG-Face", enforce_detection=False)
                features = embedding[0]["embedding"]

                # Prediction
                rf_pred = rf_model.predict([features])[0]
                self.current_emotion = rf_pred
                
                # Draw on frame
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, self.current_emotion.upper(), (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except:
                pass
        return img

# ==========================
# Streamlit UI
# ==========================
st.title("🎵 AI Hybrid Emotion Music System")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Live Feed")
    ctx = webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionTransformer)

with col2:
    st.subheader("📊 Analysis & Suggestions")
    if ctx.video_transformer:
        emo = ctx.video_transformer.current_emotion
        st.info(f"Detected Emotion: **{emo.upper()}**")
        
        if emo in activity_map:
            st.write(f"💡 Suggestion: {random.choice(activity_map[emo])}")
            
            if st.button("🎧 Play Recommended Music"):
                webbrowser.open(music_map[emo])
                st.success("Opening YouTube...")
    else:
        st.write("Please start the camera to see analysis.")

st.sidebar.title("About Project")
st.sidebar.info("આ મોડલ CNN, Random Forest અને SVM ના હાઇબ્રિડ કોમ્બિનેશનથી ચાલે છે.")
