import serial
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PORT = "COM5"   # change to your port
BAUD = 9600

WINDOW = 10
RECORD_SECONDS = 5
TARGET_ACCURACY = 0.90

GESTURES = ["Open", "Close"]

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

ser = serial.Serial(PORT, BAUD)
time.sleep(2)

X = []
y = []

print("=== ADAPTIVE LIVE TRAINING ===")

while True:

    for gesture in GESTURES:
        input(f"\nPerform {gesture}. Press Enter to record...")
        print(f"Recording {gesture} for {RECORD_SECONDS} seconds...")

        start_time = time.time()
        raw_buffer = []

        while time.time() - start_time < RECORD_SECONDS:
            try:
                line = ser.readline().decode().strip()
                parts = line.split(",")

                if len(parts) < 4:
                    continue

                s1, s2, s3, s4 = map(int, parts[:4])

                if 0 in [s1, s2, s3, s4]:
                    continue

                raw_buffer.append([s1, s2, s3, s4])

            except:
                continue

        raw_buffer = np.array(raw_buffer)

        for i in range(0, len(raw_buffer) - WINDOW, WINDOW):
            window = raw_buffer[i:i+WINDOW]
            features = extract_features(window)

            X.append(features)
            y.append(gesture)

        print(f"{gesture} data added.")

    print("\nTraining model...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.3,
        stratify=y_encoded,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=20,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.3f}")

    if acc >= TARGET_ACCURACY:
        print("Target accuracy reached.")
        break
    else:
        print("Accuracy below target. Collect more data and retrain.")

# Save final model
joblib.dump(model, "subject_model.pkl")
joblib.dump(scaler, "subject_scaler.pkl")
joblib.dump(label_encoder, "subject_label_encoder.pkl")

print("Model saved.")
