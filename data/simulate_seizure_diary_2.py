import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Simulation settings
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 3, 31)
total_days = (end_date - start_date).days + 1

# Define summer months (June, July, August)
summer_months = {6, 7, 8}

# Baseline rates:
# Baseline focal seizure rate: ~1 per week (lambda ~ 1/7)
baseline_lambda = 1/7  

# For cluster days: lambda increased (e.g., Poisson with lambda=10 for focal seizures)
cluster_lambda = 10

# Probability that a day in summer is a cluster day (e.g. 10% chance)
cluster_prob = 0.1

data = []

for i in range(total_days):
    current_day = start_date + timedelta(days=i)
    
    # Determine if this day is a cluster day (only consider summer days)
    if current_day.month in summer_months and random.random() < cluster_prob:
        # Cluster day: generate focal seizures at a higher rate.
        num_focal = np.random.poisson(lam=cluster_lambda)
        # Also add a tonic clonic seizure with 70% probability on a cluster day.
        add_tonic = random.random() < 0.7
    else:
        # Normal day: generate focal seizures with baseline rate.
        num_focal = np.random.poisson(lam=baseline_lambda)
        add_tonic = False  # rarely, we can even simulate tonic clonic on a normal day; here we skip it.
    
    # For each focal seizure, assign a random daytime time (say, between 8:00 and 20:00)
    for _ in range(num_focal):
        hour = random.randint(8, 20)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        dt = current_day.replace(hour=hour, minute=minute, second=second)
        data.append({
            "Action": "Focal seizure",
            "UTC date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Note": ""
        })
    
    # If this day has a tonic clonic seizure (for cluster days)
    if add_tonic:
        # Tonic clonic seizure typically during daytime
        hour = random.randint(9, 18)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        dt = current_day.replace(hour=hour, minute=minute, second=second)
        data.append({
            "Action": "Tonic clonic seizure",
            "UTC date": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Note": ""
        })
    
    # Optionally, you might add other events (like dizziness) here.

# Create DataFrame, sort, and output Excel
df = pd.DataFrame(data)
df['UTC date'] = pd.to_datetime(df['UTC date'], format="%Y-%m-%d %H:%M:%S")
df.sort_values(by="UTC date", inplace=True)

output_filename = "Simulated_Epilepsy_diary_2.xlsx"
df.to_excel(output_filename, index=False)

print(f"Simulated diary data saved to {output_filename}")
