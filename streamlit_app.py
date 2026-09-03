"""
Streamlit web demo for the hand sign detector.

Captures a single photo from the browser's camera, runs MediaPipe hand
landmark detection, then classifies the sign with the trained sklearn
model. Deploy as-is on Streamlit Community Cloud (entry point:
streamlit_app.py).
"""

import os
import pickle

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "model", "sign_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "src", "model", "label_encoder.pkl")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le


@st.cache_resource
def load_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6,
    )


st.set_page_config(page_title="Hand Sign Detector", page_icon="🤟")
st.title("Hand Sign Detector")
st.write(
    "Real-time ASL letter (A-Z) recognition from hand landmarks. "
    "Take a photo showing a hand sign and get an instant prediction."
)

model, le = load_model()
hands = load_hands()
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

photo = st.camera_input("Show a hand sign to the camera")

if photo is not None:
    image = np.array(Image.open(photo).convert("RGB"))
    result = hands.process(image)

    annotated = image.copy()

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(annotated, hand_lms, mp_hands.HAND_CONNECTIONS)

            features = []
            for lm in hand_lms.landmark:
                features.extend([lm.x, lm.y, lm.z])
            features = np.array(features).reshape(1, -1)

            pred_idx = model.predict(features)[0]
            pred_prob = model.predict_proba(features)[0]
            confidence = pred_prob[pred_idx]
            predicted_letter = le.inverse_transform([pred_idx])[0]

        st.image(annotated, caption="Detected hand landmarks")
        st.success(f"Predicted sign: **{predicted_letter}** ({confidence * 100:.1f}% confidence)")
    else:
        st.image(annotated, caption="No hand detected")
        st.warning("No hand detected — try a clearer, well-lit shot with your hand fully in frame.")
