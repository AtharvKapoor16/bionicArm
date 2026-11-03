import os
import numpy as np
import pandas as pd
import joblib
from collections import deque, Counter
from sklearn.metrics import accuracy_score, classification_report

WINDOW = 10
SMOOTHING_WINDOW = 5

data_folder = r"C:\Users\kapoo\src\bioarm\bioarm"
script_dir = os.path.dirname(__file__)

model = joblib.load(os.path.join(script_dir, "emg_model.pkl"))
scaler = joblib.load(os.path.join(script_dir, "emg_scaler.pkl"))
label_encoder = joblib.load(os.path.join(script_dir, "emg_label_encoder.pkl"))

def extract_features(window):
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

    return features

X_all = []
y_all = []

for file in sorted(os.listdir(data_folder)):
    if not file.startswith("Papa") or not file.endswith(".csv"):
        continue

    file_path = os.path.join(data_folder, file)
    gesture = file.replace("Papa","").replace(".csv","")

    rows = []
    with open(file_path, "r", encoding="latin1") as f:
        for line in f:
            line = line.strip()
            if "sensor" in line.lower():
                continue

            parts = line.split(",")
            if len(parts) < 4:
                continue

            try:
                s1,s2,s3,s4 = map(int, parts[:4])
                if 0 in [s1,s2,s3,s4]:
                    continue
                rows.append([s1,s2,s3,s4])
            except:
                continue

    sensor_data = np.array(rows)

    for i in range(0, len(sensor_data) - WINDOW, WINDOW):
        window = sensor_data[i:i+WINDOW]
        features = extract_features(window)

        X_all.append(features)
        y_all.append(gesture)

X_test = pd.DataFrame(X_all)
X_test_scaled = scaler.transform(X_test)

# Raw predictions
y_pred_encoded = model.predict(X_test_scaled)
y_pred_raw = label_encoder.inverse_transform(y_pred_encoded)

# ----------------------------
# TEMPORAL SMOOTHING
# ----------------------------
buffer = deque(maxlen=SMOOTHING_WINDOW)
y_pred_smoothed = []

for pred in y_pred_raw:
    buffer.append(pred)
    most_common = Counter(buffer).most_common(1)[0][0]
    y_pred_smoothed.append(most_common)

# ----------------------------
# Evaluate Smoothed
# ----------------------------
print("\nPapa Accuracy AFTER Smoothing:",
      accuracy_score(y_all, y_pred_smoothed))
print(classification_report(y_all, y_pred_smoothed))
