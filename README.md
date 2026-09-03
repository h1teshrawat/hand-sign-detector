# Hand Sign Detector — Minimal Demo

This repository contains a minimal real-time hand detection demo using MediaPipe and an offline text-to-speech engine (`pyttsx3`).

Requirements
- Python 3.8+
- See `requirements.txt` for packages.

Run
```bash
pip install -r requirements.txt
python src/hand_detection.py
```

Press `q` in the window to quit.

Next steps
- Integrate a classifier in `src/predict.py` to map landmarks to signs.
- Add data collection and training pipelines in `src/collect_data.py` and `src/train_model.py`.
