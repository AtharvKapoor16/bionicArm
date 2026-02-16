import serial
import numpy as np
import time
from collections import deque

PORT = "COM3"
BAUD = 115200

WINDOW = 20
THRESHOLD_MULTIPLIER = 3
COOLDOWN = 0.6

ser = serial.Serial(PORT, BAUD)
time.sleep(2)

buffer = deque(maxlen=WINDOW)
baseline_samples = []

print("Collecting baseline... Relax arm.")
start = time.time()

while time.time() - start < 3:
    line = ser.readline().decode().strip()
    s = np.array(list(map(int, line.split(","))))
    baseline_samples.append(np.mean(np.abs(s)))

baseline = np.mean(baseline_samples)
std = np.std(baseline_samples)

threshold = baseline + THRESHOLD_MULTIPLIER * std

print("Baseline:", baseline)
print("Threshold:", threshold)

hand_open = True
last_trigger = 0

while True:
    try:
        line = ser.readline().decode().strip()
        s = np.array(list(map(int, line.split(","))))

        envelope = np.mean(np.abs(s))
        buffer.append(envelope)

        smoothed = np.mean(buffer)

        if smoothed > threshold:
            if time.time() - last_trigger > COOLDOWN:

                hand_open = not hand_open
                last_trigger = time.time()

                if hand_open:
                    print("OPEN")
                    ser.write(b'O')
                else:
                    print("CLOSE")
                    ser.write(b'C')

    except:
        continue
