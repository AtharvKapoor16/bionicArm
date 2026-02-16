import serial
import numpy as np
import torch
import torch.nn as nn
import joblib
from collections import deque

PORT = "COM3"
BAUD = 115200
WINDOW = 20
THRESHOLD = 0.7

mins, maxs = joblib.load("sensor_minmax.pkl")
encoder = joblib.load("gru_encoder.pkl")

def normalize(x):
    return (x - mins) / (maxs - mins + 1e-8)

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Load model
model = GRUModel()
model.load_state_dict(torch.load("gru_model.pt"))
model.eval()

ser = serial.Serial(PORT, BAUD)

print("Starting live prediction...")

buffer = deque(maxlen=WINDOW)

while True:
    try:
        line = ser.readline().decode().strip()
        s = np.array(list(map(int, line.split(","))))

        if len(s) < 4:
            continue

        s = normalize(s)
        buffer.append(s)

        if len(buffer) == WINDOW:
            seq = torch.tensor(np.array(buffer), dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                outputs = model(seq)
                probs = torch.softmax(outputs, dim=1)[0]

            conf, idx = torch.max(probs, 0)

            if conf.item() > THRESHOLD:
                gesture = encoder.inverse_transform([idx.item()])[0]
                print(f"Prediction: {gesture} | Confidence: {conf.item():.2f}")

    except:
        continue
