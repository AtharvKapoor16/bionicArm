import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
print("Loading data...")
data = pd.read_csv(r"C:\Users\kapoo\src\bioarm\bioarm\merged_data.csv")

WINDOW = 10  # number of samples per window

# =========================
# EXTRACT SENSORS + LABELS
# =========================
sensor_data = data[['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']].values
gesture_columns = data.columns[4:]
labels_onehot = data[gesture_columns].values

X_new = []
y_new = []

# =========================
# WINDOW FEATURE EXTRACTION
# =========================
for i in range(0, len(sensor_data) - WINDOW, WINDOW):
    window = sensor_data[i:i+WINDOW]
    label_window = labels_onehot[i:i+WINDOW]

    # Features
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    rms = np.sqrt(np.mean(window**2, axis=0))

    features = np.concatenate([mean, std, rms])
    X_new.append(features)

    # Majority label in window
    label_counts = np.sum(label_window, axis=0)
    label = gesture_columns[np.argmax(label_counts)]
    y_new.append(label)

X = pd.DataFrame(X_new)
y = pd.Series(y_new)

print("Total windows:", len(X))

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================
# TRAIN MODEL
# =========================
print("Training model...")
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
print("Evaluating...")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {acc * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "emg_model.pkl")
joblib.dump(model, model_path)

print(f"Model saved to: {model_path}")
print("Done.")
