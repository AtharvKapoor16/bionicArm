import serial
import time
import numpy as np
import joblib

PORT = "COM3"
BAUD = 115200
CALIBRATION_SECONDS = 8

ser = serial.Serial(PORT, BAUD)
time.sleep(2)

print("Relax arm completely...")
time.sleep(2)

mins = np.full(4, 9999)
maxs = np.zeros(4)

start = time.time()

while time.time() - start < CALIBRATION_SECONDS:
    try:
        line = ser.readline().decode().strip()
        s = np.array(list(map(int, line.split(","))))
        if len(s) < 4:
            continue

        mins = np.minimum(mins, s)
        maxs = np.maximum(maxs, s)
    except:
        continue

print("Now flex everything strongly...")
time.sleep(2)

start = time.time()
while time.time() - start < CALIBRATION_SECONDS:
    try:
        line = ser.readline().decode().strip()
        s = np.array(list(map(int, line.split(","))))
        if len(s) < 4:
            continue

        mins = np.minimum(mins, s)
        maxs = np.maximum(maxs, s)
    except:
        continue

joblib.dump((mins, maxs), "sensor_minmax.pkl")

print("Calibration saved.")
