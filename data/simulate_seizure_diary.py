import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Simulation settings
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 3, 1)
total_days = (end_date - start_date).days + 1

# Menstrual cycle: assume period starts every 28 days (starting on Jan 1, 2024)
cycle_length = 28
period_starts = [start_date + timedelta(days=i) for i in range(0, total_days, cycle_length)]

# Baseline event rates (per day)
focal_lambda = 1/7      # ~2 focal seizures per week
tonic_lambda = 0.5/28     # ~1 tonic clonic seizure per month
dizziness_lambda = 0.05  # dizziness events per day

def is_pre_period(current_day, period_starts):
    """Returns True if current_day is 1-3 days before a period start."""
    for ps in period_starts:
        if ps > current_day and 1 <= (ps - current_day).days <= 5:
            return True
    return False

data = []

for i in range(total_days):
    current_day = start_date + timedelta(days=i)
    
    # --- Start of Period Event ---
    if current_day in period_starts:
        # Assume period starts around 7:00 AM with a ±30-minute random variation.
        hour = 7 + random.randint(-1, 1)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        dt = current_day.replace(hour=hour, minute=minute, second=second)
        data.append({
            "Action": "Start of period",
            "UTC Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Note": ""
        })
    
    # --- Focal Seizures ---
    # Increase rate by 20% if within 1-3 days before a period
    focal_rate = focal_lambda * 1.5 if is_pre_period(current_day, period_starts) else focal_lambda
    num_focal = np.random.poisson(lam=focal_rate)
    for _ in range(num_focal):
        # Simulate a focal seizure during daytime: between 7:00 and 20:00
        hour = random.randint(7, 20)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        dt = current_day.replace(hour=hour, minute=minute, second=second)
        data.append({
            "Action": "Focal seizure",
            "UTC Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Note": ""
        })
    
    # --- Tonic Clonic Seizures ---
    num_tonic = np.random.poisson(lam=tonic_lambda)
    for _ in range(num_tonic):
        # Simulate tonic clonic seizures mostly during daytime (8:00-18:00)
        hour = random.randint(8, 18)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        dt = current_day.replace(hour=hour, minute=minute, second=second)
        data.append({
            "Action": "Tonic clonic seizure",
            "UTC Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Note": ""
        })
    
    # --- Dizziness Events ---
    num_dizziness = np.random.poisson(lam=dizziness_lambda)
    for _ in range(num_dizziness):
        # Dizziness may occur at any time; assign a random time over the full day.
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        dt = current_day.replace(hour=hour, minute=minute, second=second)
        data.append({
            "Action": "Dizziness",
            "UTC Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Note": ""
        })

# Create a DataFrame, convert the date column to datetime, and sort
df = pd.DataFrame(data)
df["UTC Time"] = pd.to_datetime(df["UTC Time"], format="%Y-%m-%d %H:%M:%S")
df.sort_values(by="UTC Time", inplace=True)

# Save to Excel
output_filename = "Simulated_Epilepsy_diary_2.xlsx"
df.to_excel(output_filename, index=False)
print(f"Simulated dataset saved to {output_filename}")