import serial
import joblib
import pandas as pd
import os

# ======================
# LOAD MODEL
# ======================
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "emg_model.pkl")

model = joblib.load(model_path)
feature_names = model.feature_names_in_

print("Model loaded.")
print("Starting live prediction...\n")

# ======================
# SERIAL CONNECTION
# ======================
ser = serial.Serial("COM3", 115200)  # change COM port if needed

while True:
    try:
        line = ser.readline().decode().strip()

        parts = line.split(",")

        if len(parts) != 4:
            continue

        values = [float(p) for p in parts]

        input_df = pd.DataFrame([values], columns=feature_names)

        prediction = model.predict(input_df)[0]
        confidence = max(model.predict_proba(input_df)[0]) * 100

        print(f"Gesture: {prediction} | Confidence: {confidence:.1f}%")

    except:
        continue
