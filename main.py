import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

WINDOW = 10

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, "merged_data.csv")

print("Loading data...")
data = pd.read_csv(data_path)

sensor_cols = ['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']
sensor_data = data[sensor_cols].values

gesture_columns = data.columns[4:]
labels_onehot = data[gesture_columns].values

X_new = []
y_new = []

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

for i in range(0, len(sensor_data) - WINDOW, WINDOW):
    window = sensor_data[i:i+WINDOW]
    label_window = labels_onehot[i:i+WINDOW]

    features = extract_features(window)
    X_new.append(features)

    label_counts = np.sum(label_window, axis=0)
    label = gesture_columns[np.argmax(label_counts)]
    y_new.append(label)

X = pd.DataFrame(X_new)
y = pd.Series(y_new)

print("Total windows:", len(X))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nTraining Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, os.path.join(script_dir, "emg_model.pkl"))
joblib.dump(scaler, os.path.join(script_dir, "emg_scaler.pkl"))

print("\nModel + scaler saved.")
