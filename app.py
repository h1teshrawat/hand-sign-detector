"""
Gradio web demo for the hand sign detector.

Loads the trained sklearn model + MediaPipe hand landmarker and exposes
a live webcam interface. Run locally with `python app.py`, or deploy
as-is to a Hugging Face Space (Gradio SDK).
"""

import os
import pickle

import gradio as gr
import mediapipe as mp
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "model", "sign_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "src", "model", "label_encoder.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)


def predict_sign(frame):
    if frame is None:
        return None, "Waiting for camera..."

    frame = np.array(frame)
    result = hands.process(frame)

    annotated = frame.copy()
    label_text = "No hand detected"

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

            label_text = f"{predicted_letter}  ({confidence * 100:.1f}%)"

    return annotated, label_text


demo = gr.Interface(
    fn=predict_sign,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Webcam"),
    outputs=[gr.Image(label="Annotated"), gr.Textbox(label="Predicted Sign")],
    live=True,
    title="Hand Sign Detector",
    description=(
        "Real-time ASL letter (A-Z) recognition from hand landmarks. "
        "Allow camera access and show a sign."
    ),
)

if __name__ == "__main__":
    demo.launch()
