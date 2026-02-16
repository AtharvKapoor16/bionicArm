import serial
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ======================
# SETTINGS
# ======================
PORT = "COM5"     # change to your Arduino port
BAUD = 9600

WINDOW = 10
RECORD_SECONDS = 6   # record per gesture
GESTURES = ["Open", "Close"]

# ======================
# FEATURE EXTRACTION
# ======================
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

# ======================
# START SERIAL
# ======================
ser = serial.Serial(PORT, BAUD)
time.sleep(2)

X = []
y = []

print("=== LIVE SUBJECT CALIBRATION (Open vs Close) ===")

# ======================
# RECORD DATA
# ======================
for gesture in GESTURES:
    input(f"\nGet ready to perform {gesture}. Press Enter to start recording...")
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

    # Convert raw signal into windows
    for i in range(0, len(raw_buffer) - WINDOW, WINDOW):
        window = raw_buffer[i:i+WINDOW]
        features = extract_features(window)

        X.append(features)
        y.append(gesture)

    print(f"{gesture} data collected.")

# ======================
# TRAIN MODEL
# ======================
print("\nTraining subject-specific model...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    class_weight='balanced',
    random_state=42
)

model.fit(X_scaled, y_encoded)

# ======================
# SAVE MODEL
# ======================
joblib.dump(model, "subject_model.pkl")
joblib.dump(scaler, "subject_scaler.pkl")
joblib.dump(label_encoder, "subject_label_encoder.pkl")

print("\nModel trained successfully.")
print("Files saved:")
print(" - subject_model.pkl")
print(" - subject_scaler.pkl")
print(" - subject_label_encoder.pkl")
