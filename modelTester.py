import joblib
import pandas as pd
import os

# =========================
# LOAD MODEL
# =========================
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "emg_model.pkl")

model = joblib.load(model_path)

print("Model loaded successfully.")
print(f"Model expects {model.n_features_in_} values.\n")

feature_names = model.feature_names_in_

# =========================
# INPUT LOOP
# =========================
while True:
    user_input = input("Enter 4 sensor values (ex: 51,99,25,50) or type 'exit': ")

    if user_input.lower() == "exit":
        break

    parts = user_input.split(",")

    if len(parts) != 4:
        print("Error: Enter exactly 4 values.\n")
        continue

    try:
        values = [float(p.strip()) for p in parts]
    except:
        print("Invalid input. Enter numbers only.\n")
        continue

    # Create DataFrame with correct column names
    input_df = pd.DataFrame([values], columns=feature_names)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities) * 100

    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%\n")
