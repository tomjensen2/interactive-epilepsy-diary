import pandas as pd

# List of allowed Action values
allowed_actions = [
    "Standard nighttime Seizure",
    "Need Rescue meds",
    "Reduce my lamotrogine last",
    "Have continuous focal SE",
    "Change Vimpat dose",
    "Seizure from Wakefulness",
    "Reduce Clobazam",
    "Change Cenobamate",
    "Change Perampanel "
]

# Read the CSV file (change "input.csv" to your actual filename)
df = pd.read_csv("When-Did-I.csv")

# Filter rows where the "Action" column is one of the allowed actions
filtered_df = df[df["Action"].isin(allowed_actions)]

# Write the filtered data to a new CSV file (change "output.csv" as desired)
filtered_df.to_csv("When-Did-I_2.csv", index=False)