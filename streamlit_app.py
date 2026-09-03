"""
Streamlit web demo for the hand sign detector.

Captures a single photo from the browser's camera, runs MediaPipe's
HandLandmarker (Tasks API) to find hand landmarks, then classifies the
sign with the trained sklearn model. Deploy as-is on Streamlit Community
Cloud (entry point: streamlit_app.py).

Uses the Tasks API instead of the legacy `mp.solutions.hands` because
newer mediapipe releases (needed for newer Python versions on hosted
platforms) dropped `mp.solutions` entirely.
"""

import os
import pickle

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "model", "sign_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "src", "model", "label_encoder.pkl")
HAND_LANDMARKER_PATH = os.path.join(BASE_DIR, "assets", "hand_landmarker.task")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le


@st.cache_resource
def load_hand_landmarker():
    base_options = mp_tasks.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


st.set_page_config(page_title="Hand Sign Detector", page_icon="🤟")
st.title("Hand Sign Detector")
st.write(
    "Real-time ASL letter (A-Z) recognition from hand landmarks. "
    "Take a photo showing a hand sign and get an instant prediction."
)

model, le = load_model()
landmarker = load_hand_landmarker()

photo = st.camera_input("Show a hand sign to the camera")

if photo is not None:
    image = np.array(Image.open(photo).convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = landmarker.detect(mp_image)

    annotated = image.copy()
    h, w, _ = annotated.shape

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        for connection in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS:
            start = hand_landmarks[connection.start]
            end = hand_landmarks[connection.end]
            cv2.line(
                annotated,
                (int(start.x * w), int(start.y * h)),
                (int(end.x * w), int(end.y * h)),
                (0, 200, 0),
                2,
            )
        for lm in hand_landmarks:
            cv2.circle(annotated, (int(lm.x * w), int(lm.y * h)), 4, (0, 140, 255), -1)

        features = []
        for lm in hand_landmarks:
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
