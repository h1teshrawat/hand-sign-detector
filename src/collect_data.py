import cv2
import mediapipe as mp
import os
import csv


mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)


os.makedirs("data", exist_ok=True)
output_file = "data/hand_signs.csv"

# 26 letters
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

cap = cv2.VideoCapture(0)
current_label = None
sample_count  = 0
SAMPLES_PER_LABEL = 200

print("Press a letter key (A-Z) to start recording that sign.")
print("Press 'q' to quit.")

with open(output_file, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror so it feels natural
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                if current_label is not None and sample_count < SAMPLES_PER_LABEL:
                    # Extract all 21 landmarks → 63 values
                    row = [current_label]
                    for lm in hand_lms.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    writer.writerow(row)
                    sample_count += 1

                    if sample_count == SAMPLES_PER_LABEL:
                        print(f"  Done collecting '{current_label}' ({SAMPLES_PER_LABEL} samples)")
                        current_label = None
                        sample_count  = 0

        # HUD
        status = f"Recording: {current_label} ({sample_count}/{SAMPLES_PER_LABEL})" if current_label else "Press A-Z to record"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect Hand Signs", frame)

        key = cv2.waitKey(1) & 0xFF
        if key ==  27:
            break
        elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
            letter = chr(key).upper()
            current_label = letter
            sample_count  = 0
            print(f"Recording '{letter}' — show the sign!")

cap.release()
cv2.destroyAllWindows()

