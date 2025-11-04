import os
import pandas as pd
import numpy as np
from scipy.stats import zscore

cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')

payment_master_path = os.path.join(data_dir, 'payments_master.csv')
fraud_case_master_path = os.path.join(data_dir, 'fraud_cases_master.csv')
research_team_master_path = os.path.join(data_dir, 'research_team_master.csv')
research_team_member_master_path = os.path.join(data_dir, 'research_team_member_master.csv')

# Load the dataset
try:
    df = pd.read_csv('payments_master.csv')
except FileNotFoundError:
    print("Error: payments_master.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Convert date columns to datetime objects
date_cols = ['Date received', 'Date of payment']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Calculate 'time to payment' in days
df['time_to_payment'] = (df['Date of payment'] - df['Date received']).dt.days

# Drop rows where 'time_to_payment' is NaN
df.dropna(subset=['time_to_payment'], inplace=True)

# Calculate Z-scores for 'time_to_payment'
df['zscore_time_to_payment'] = np.abs(zscore(df['time_to_payment']))

# Define a threshold for outliers (e.g., Z-score > 3)
threshold = 3

# Identify outliers
outliers = df[df['zscore_time_to_payment'] > threshold]

print(f"Identified {len(outliers)} outliers in 'time to payment' using a Z-score threshold of {threshold}.")
print("\nOutliers detected (first 10 rows):")
print(outliers[['Invoice number', 'Date received', 'Date of payment', 'time_to_payment', 'zscore_time_to_payment']].head(10))
