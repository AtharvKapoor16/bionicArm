import serial
import numpy as np
import time
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

PORT = "COM3"
BAUD = 115200
WINDOW = 20
RECORD_SECONDS = 4
REPS = 3
GESTURES = ["Open", "Close"]

mins, maxs = joblib.load("sensor_minmax.pkl")

def normalize(x):
    return (x - mins) / (maxs - mins + 1e-8)

def moving_average(x, k=10):
    return np.convolve(x, np.ones(k)/k, mode='same')

ser = serial.Serial(PORT, BAUD)
time.sleep(2)

X = []
y = []

for gesture in GESTURES:
    for r in range(REPS):
        input(f"Perform {gesture} rep {r+1}")
        start = time.time()
        buffer = []

        while time.time() - start < RECORD_SECONDS:
            try:
                line = ser.readline().decode().strip()
                s = np.array(list(map(int, line.split(","))))
                if len(s) < 4:
                    continue

                s = normalize(s)
                buffer.append(s)
            except:
                continue

        buffer = np.array(buffer)

        for i in range(4):
            buffer[:, i] = moving_average(buffer[:, i])

        for i in range(len(buffer) - WINDOW):
            seq = buffer[i:i+WINDOW]
            X.append(seq)
            y.append(gesture)

X = torch.tensor(np.array(X), dtype=torch.float32)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, len(GESTURES))

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = GRUModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "gru_model.pt")
joblib.dump(encoder, "gru_encoder.pkl")

print("GRU model saved.")
