import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

WINDOW = 10
data_folder = r"C:\Users\kapoo\src\bioarm\bioarm"

script_dir = os.path.dirname(__file__)

model = joblib.load(os.path.join(script_dir, "emg_model.pkl"))
scaler = joblib.load(os.path.join(script_dir, "emg_scaler.pkl"))

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
    if not file.lower().endswith(".csv"):
        continue
    if not file.startswith("Papa"):
        continue

    file_path = os.path.join(data_folder, file)
    file_name = file.replace(".csv", "")
    action_name = file_name.replace("Papa", "")

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
                s1, s2, s3, s4 = map(int, parts[:4])
                if 0 in [s1, s2, s3, s4]:
                    continue
                rows.append([s1, s2, s3, s4])
            except:
                continue

    sensor_data = np.array(rows)

    for i in range(0, len(sensor_data) - WINDOW, WINDOW):
        window = sensor_data[i:i+WINDOW]
        features = extract_features(window)

        X_all.append(features)
        y_all.append(action_name)

X_test = pd.DataFrame(X_all)
X_test_scaled = scaler.transform(X_test)
y_test = pd.Series(y_all)

y_pred = model.predict(X_test_scaled)

print("\nPapa Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
