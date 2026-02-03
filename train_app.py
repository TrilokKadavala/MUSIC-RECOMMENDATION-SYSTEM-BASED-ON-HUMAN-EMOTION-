import os
import cv2
import numpy as np
import joblib
import streamlit as st
import time
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Streamlit Page Setup
st.set_page_config(page_title="AI Model Trainer", layout="wide")
st.title("🧠 Hybrid Emotion Model Trainer")
st.markdown("CNN (VGG-Face) + Random Forest + SVM")

# Configuration
DATASET_PATH = "dataset"
EMOTIONS = ["happy","sad","angry","neutral","surprise","fear","disgust","sleepy"]

# Sidebar for Hyperparameters
st.sidebar.header("Model Settings")
n_trees = st.sidebar.slider("RF Estimators", 50, 500, 150)
test_size = st.sidebar.slider("Test Data Size (%)", 10, 50, 20) / 100

if st.button("🚀 Start Hybrid Training"):
    X = []
    y = []

    # 1. Feature Extraction Phase
    st.header("1. Feature Extraction (CNN)")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_emotions = len(EMOTIONS)
    
    for idx, emotion in enumerate(EMOTIONS):
        emotion_folder = os.path.join(DATASET_PATH, emotion)
        if not os.path.exists(emotion_folder):
            st.warning(f"Folder not found: {emotion}")
            continue
            
        status_text.text(f"Processing Emotion: {emotion}...")
        img_list = os.listdir(emotion_folder)
        
        for img_name in img_list:
            img_path = os.path.join(emotion_folder, img_name)
            try:
                img = cv2.imread(img_path)
                embedding = DeepFace.represent(img, model_name="VGG-Face", enforce_detection=False)
                X.append(embedding[0]["embedding"])
                y.append(emotion)
            except:
                pass
        
        # Update Progress
        progress_bar.progress((idx + 1) / total_emotions)

    X = np.array(X)
    y = np.array(y)
    st.success(f"✅ Feature Extraction Completed! Total Samples: {len(X)}")

    # 2. Training Phase
    st.header("2. Model Training")
    with st.spinner("Training Random Forest and SVM..."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)

        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)

        # SVM
        svm_model = SVC(kernel="rbf", probability=True)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_pred)

        # Hybrid Logic
        hybrid_pred = [rf_pred[i] if rf_pred[i] == svm_pred[i] else rf_pred[i] for i in range(len(rf_pred))]
        hybrid_acc = accuracy_score(y_test, hybrid_pred)

    # 3. Results Display
    st.header("3. Training Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest Accuracy", f"{rf_acc*100:.2f}%")
    c2.metric("SVM Accuracy", f"{svm_acc*100:.2f}%")
    c3.metric("Hybrid Model Accuracy", f"{hybrid_acc*100:.2f}%", delta="Final")

    # Save Models
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(svm_model, "svm_model.pkl")
    
    st.balloons()
    st.success("🎉 Models Saved Successfully as .pkl files!")
