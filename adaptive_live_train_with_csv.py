import serial
import numpy as np
import joblib
import time
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PORT = "COM5"
BAUD = 115200

WINDOW = 25
RECORD_SECONDS = 8
GESTURES = ["Open", "Close"]

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

print("=== BINARY TRAINING (Open vs Close) ===")

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
        except:
            continue

    buffer = np.array(buffer)

    # Sliding windows
    for i in range(0, len(buffer) - WINDOW):
        window = buffer[i:i+WINDOW]
        features = extract_features(window)

        X.append(features)
        y.append(gesture)

    print(f"{gesture} added.")

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
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
print(f"\nValidation Accuracy: {acc:.3f}")

joblib.dump(model, "subject_model.pkl")
joblib.dump(scaler, "subject_scaler.pkl")
joblib.dump(encoder, "subject_encoder.pkl")

print("Model saved.")
