import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Load model
with open("model/sign_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_letter = "—"
    confidence       = 0.0

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Build feature vector: 63 values
            features = []
            for lm in hand_lms.landmark:
                features.extend([lm.x, lm.y, lm.z])
            features = np.array(features).reshape(1, -1)

            # Predict
            pred_idx  = model.predict(features)[0]
            pred_prob = model.predict_proba(features)[0]
            confidence = pred_prob[pred_idx]
            predicted_letter = le.inverse_transform([pred_idx])[0]

    # Overlay
    color = (0, 200, 0) if confidence > 0.7 else (0, 140, 255)
    cv2.putText(frame, f"{predicted_letter}  {confidence*100:.1f}%",
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
    cv2.putText(frame, "Press Q to quit",
                (20, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

    cv2.imshow("Hand Sign Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()