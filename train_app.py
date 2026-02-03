import os
import streamlit as st
import cv2
import numpy as np
import joblib
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Model Trainer", layout="wide")
st.title("🧠 Hybrid Emotion Model Trainer")

DATASET_PATH = "dataset"
EMOTIONS = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust", "sleepy"]

if st.button("🚀 Start Model Training"):
    X, y = [], []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 1. Feature Extraction
    st.subheader("Step 1: Extracting Features from Images")
    for idx, emotion in enumerate(EMOTIONS):
        folder = os.path.join(DATASET_PATH, emotion)
        if not os.path.exists(folder): continue
        
        status_text.text(f"Processing: {emotion}")
        for img_name in os.listdir(folder):
            try:
                img = cv2.imread(os.path.join(folder, img_name))
                embedding = DeepFace.represent(img, model_name="VGG-Face", enforce_detection=False)
                X.append(embedding[0]["embedding"])
                y.append(emotion)
            except: pass
        progress_bar.progress((idx + 1) / len(EMOTIONS))

    X, y = np.array(X), np.array(y)
    
    # 2. Training
    st.subheader("Step 2: Training Models")
    with st.spinner("Training Random Forest & SVM..."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # RF
        rf = RandomForestClassifier(n_estimators=150, random_state=42)
        rf.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        
        # SVM
        svm = SVC(kernel="rbf", probability=True)
        svm.fit(X_train, y_train)
        svm_acc = accuracy_score(y_test, svm.predict(X_test))
        
        # Save
        joblib.dump(rf, "rf_model.pkl")
        joblib.dump(svm, "svm_model.pkl")

    # 3. Results
    c1, c2 = st.columns(2)
    c1.metric("Random Forest Accuracy", f"{rf_acc*100:.2f}%")
    c2.metric("SVM Accuracy", f"{svm_acc*100:.2f}%")
    st.success("✅ Models Saved: rf_model.pkl & svm_model.pkl")
    st.balloons()
