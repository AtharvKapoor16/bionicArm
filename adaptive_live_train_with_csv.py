import serial
import numpy as np
import joblib
import time
import csv
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PORT = "COM3"
BAUD = 115200

WINDOW = 25
RECORD_SECONDS = 8
GESTURES = ["Open", "Close"]

# Save files in same folder as script
BASE_DIR = os.path.dirname(__file__)
RAW_FILE = os.path.join(BASE_DIR, "raw_emg_data.csv")
FEATURE_FILE = os.path.join(BASE_DIR, "feature_data.csv")

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
time.sleep(2)

X = []
y = []
raw_rows = []
feature_rows = []

print("=== TRAINING WITH CSV LOGGING ===")

for gesture in GESTURES:

    input(f"\nPerform {gesture}. Press Enter...")
    start = time.time()
    buffer = []

    while time.time() - start < RECORD_SECONDS:
        try:
            line = ser.readline().decode().strip()
            parts = line.split(",")

            if len(parts) < 4:
                continue

            s1, s2, s3, s4 = map(int, parts[:4])
            buffer.append([s1, s2, s3, s4])
            raw_rows.append([gesture, s1, s2, s3, s4])

        except:
            continue

    buffer = np.array(buffer)

    for i in range(0, len(buffer) - WINDOW):
        window = buffer[i:i+WINDOW]
        features = extract_features(window)

        X.append(features)
        y.append(gesture)
        feature_rows.append([gesture] + features)

    print(f"{gesture} collected.")

# ===== SAVE RAW CSV =====
with open(RAW_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Gesture", "S1", "S2", "S3", "S4"])
    writer.writerows(raw_rows)

# ===== SAVE FEATURE CSV =====
header = ["Gesture"]
for ch in range(1, 5):
    header += [
        f"Mean_{ch}",
        f"Std_{ch}",
        f"RMS_{ch}",
        f"MAV_{ch}",
        f"WL_{ch}",
        f"ZC_{ch}"
    ]
header += ["Ratio_1", "Ratio_2", "Ratio_3", "Ratio_4"]

with open(FEATURE_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(feature_rows)

print("CSV files saved.")

# ===== TRAIN MODEL =====
X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.3,
    stratify=y_encoded,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=800,
    max_depth=30,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Validation Accuracy: {acc:.3f}")

joblib.dump(model, os.path.join(BASE_DIR, "subject_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "subject_scaler.pkl"))
joblib.dump(encoder, os.path.join(BASE_DIR, "subject_encoder.pkl"))

print("Model saved.")
