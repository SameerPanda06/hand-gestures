import streamlit as st
import numpy as np
import joblib
import cv2
import os
from skimage.feature import hog
from PIL import Image

st.set_page_config(page_title="Hand Gesture Recognizer", page_icon="✋", layout="wide")
st.title("✋ Hand Gesture Recognition — SVM + Random Forest")

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    svm = joblib.load(os.path.join(BASE, 'models', 'svm.pkl'))
    rf  = joblib.load(os.path.join(BASE, 'models', 'random_forest.pkl'))
    sc  = joblib.load(os.path.join(BASE, 'models', 'scaler.pkl'))
    le  = joblib.load(os.path.join(BASE, 'models', 'label_encoder.pkl'))
    return svm, rf, sc, le

svm, rf, scaler, le = load_models()

def predict(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    feat = hog(gray, orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               visualize=False)
    feat_scaled = scaler.transform([feat])
    svm_pred  = le.inverse_transform(svm.predict(feat_scaled))[0]
    svm_proba = svm.predict_proba(feat_scaled)[0].max() * 100
    rf_pred   = le.inverse_transform(rf.predict(feat_scaled))[0]
    rf_proba  = rf.predict_proba(feat_scaled)[0].max() * 100
    return svm_pred, svm_proba, rf_pred, rf_proba

# Upload
st.sidebar.header("Upload Gesture Image")
uploaded = st.sidebar.file_uploader("JPG / PNG", type=['jpg', 'jpeg', 'png'])

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.sidebar.image(image, caption='Uploaded', use_container_width=True)

    svm_pred, svm_conf, rf_pred, rf_conf = predict(img_array)

    st.subheader("Predictions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🤖 SVM")
        st.metric("Gesture", svm_pred)
        st.metric("Confidence", f"{svm_conf:.1f}%")
        st.progress(int(svm_conf))
    with col2:
        st.markdown("### 🌲 Random Forest")
        st.metric("Gesture", rf_pred)
        st.metric("Confidence", f"{rf_conf:.1f}%")
        st.progress(int(rf_conf))

    if svm_pred == rf_pred:
        st.success(f"Both models agree: **{svm_pred}**")
    else:
        st.warning(f"Models disagree — SVM: {svm_pred} | RF: {rf_pred}")

# Plots
st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)
with col1:
    p = os.path.join(BASE, 'plots', 'gesture_samples.png')
    if os.path.exists(p):
        st.image(p, caption='Gesture Classes', use_container_width=True)
with col2:
    p = os.path.join(BASE, 'plots', 'confusion_svm.png')
    if os.path.exists(p):
        st.image(p, caption='SVM Confusion Matrix', use_container_width=True)
with col3:
    p = os.path.join(BASE, 'plots', 'confusion_random_forest.png')
    if os.path.exists(p):
        st.image(p, caption='RF Confusion Matrix', use_container_width=True)