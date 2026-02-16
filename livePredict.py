import serial
import numpy as np
import joblib
from collections import deque

PORT = "COM5"
BAUD = 115200

WINDOW = 25
SMOOTHING = 15
VOTE_THRESHOLD = 0.8
CONF_THRESHOLD = 0.75

model = joblib.load("subject_model.pkl")
scaler = joblib.load("subject_scaler.pkl")
encoder = joblib.load("subject_encoder.pkl")

def extract_features(window):

    window = window - np.mean(window, axis=0)
    features = []

    for ch in range(window.shape[1]):
        signal = window[:, ch]

        mean = np.mean(signal)
        std = np.std(signal)
        rms = np.sqrt(np.mean(signal**2))
        mav = np.mean(np.abs(signal))
        wl = np.sum(np.abs(np.diff(signal)))
        zc = np.sum(np.diff(np.sign(signal)) != 0)

        features.extend([mean, std, rms, mav, wl, zc])

    energy = np.mean(np.abs(window), axis=0)
    ratio = energy / (np.sum(energy) + 1e-8)
    features.extend(ratio)

    return features

ser = serial.Serial(PORT, BAUD)

buffer = []
pred_buffer = deque(maxlen=SMOOTHING)

print("Binary Live System Running...")

while True:
    try:
        line = ser.readline().decode().strip()
        parts = line.split(",")

        if len(parts) < 4:
            continue

        s1, s2, s3, s4 = map(int, parts[:4])
        buffer.append([s1, s2, s3, s4])

        if len(buffer) >= WINDOW:

            window = np.array(buffer[-WINDOW:])
            features = extract_features(window)
            features_scaled = scaler.transform([features])

            probs = model.predict_proba(features_scaled)[0]
            max_prob = np.max(probs)

            if max_prob > CONF_THRESHOLD:

                pred = encoder.inverse_transform([np.argmax(probs)])[0]
                pred_buffer.append(pred)

                if len(pred_buffer) == SMOOTHING:

                    counts = {g: pred_buffer.count(g) for g in set(pred_buffer)}
                    best = max(counts, key=counts.get)

                    if counts[best] / SMOOTHING >= VOTE_THRESHOLD:
                        print(f"{best} ({max_prob:.2f})")

    except:
        continue
