import os
import pandas as pd

# =========================
# FOLDERS
# =========================

# Folder containing your original raw CSV files (IntelliJ folder)
data_folder = r"C:\Users\kapoo\src\bioarm\bioarm"

# Folder where this Python script lives (PyCharm project)
script_dir = os.path.dirname(__file__)

# Output file saved next to this script
output_file = os.path.join(script_dir, "merged_data.csv")

# =========================
# SETTINGS
# =========================

actions = [
    "Fist", "IndexClose", "IndexOpen",
    "MiddleClose", "MiddleOpen", "Open",
    "PinkyClose", "PinkyOpen", "Relaxed",
    "RingClose", "RingOpen",
    "ThumbClose", "ThumbOpen"
]

prefixes = ["Mumma", "UV", "Dadu", "Dadi"]

merged_rows = []

print("Scanning CSV files...")

for file in sorted(os.listdir(data_folder)):
    if file.lower().endswith(".csv") and file != "merged_data.csv":

        file_path = os.path.join(data_folder, file)
        file_name = file.replace(".csv", "")

        print(f"Processing: {file}")

        # Extract action name from filename
        action_name = file_name
        for prefix in prefixes:
            if file_name.startswith(prefix):
                action_name = file_name[len(prefix):]
                break

        # Read file line-by-line (like Java did)
        with open(file_path, "r", encoding="latin1") as f:
            for line in f:
                line = line.strip()

                # Skip header lines
                if "sensor" in line.lower():
                    continue

                parts = line.split(",")

                # Skip incomplete rows
                if len(parts) < 4:
                    continue

                try:
                    s1 = int(parts[0])
                    s2 = int(parts[1])
                    s3 = int(parts[2])
                    s4 = int(parts[3])

                    # Skip rows where any sensor is zero
                    if s1 == 0 or s2 == 0 or s3 == 0 or s4 == 0:
                        continue

                    row = [s1, s2, s3, s4]

                    # Add one-hot encoding
                    for act in actions:
                        row.append(1 if act.lower() == action_name.lower() else 0)

                    merged_rows.append(row)

                except:
                    continue

# Create DataFrame
merged_df = pd.DataFrame(
    merged_rows,
    columns=["Sensor1", "Sensor2", "Sensor3", "Sensor4"] + actions
)

# Save merged file in PyCharm project folder
merged_df.to_csv(output_file, index=False)

print("\n✅ All CSVs merged successfully.")
print(f"Total rows: {len(merged_df)}")
print(f"Saved to: {output_file}")
