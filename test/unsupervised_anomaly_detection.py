import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')

payment_master_path = os.path.join(data_dir, 'payments_master.csv')
fraud_case_master_path = os.path.join(data_dir, 'fraud_cases_master.csv')
research_team_master_path = os.path.join(data_dir, 'research_team_master.csv')
research_team_member_master_path = os.path.join(data_dir, 'research_team_member_master.csv')

# Load the dataset
try:
    df_payments = pd.read_csv(payment_master_path)
except FileNotFoundError:
    print("Error: payments_master.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Convert date columns to datetime objects for potential feature engineering
date_cols = ['Date received', 'Date of invoice', 'Date of authorisation', 'Payment due date', 'Date of payment']
for col in date_cols:
    if col in df_payments.columns:
        df_payments[col] = pd.to_datetime(df_payments[col], errors='coerce')

# Calculate 'time to payment' as a feature
df_payments['time_to_payment'] = (df_payments['Date of payment'] - df_payments['Date received']).dt.days

# Select features for anomaly detection
# We'll use a similar set of features that might indicate unusual patterns.
# 'Invoice value', 'Payment amount', 'time_to_payment' are numerical.
# 'Research team', 'Type of expense', 'Company', 'Payment Status', 'Submitted by', 'Authorised by', 'Payment authoriser' are categorical.

# Drop original date columns as we have 'time_to_payment'
df_payments = df_payments.drop(columns=date_cols, errors='ignore')

# Handle missing values: fill numerical with median, categorical with mode
for col in ['Invoice value', 'Payment amount', 'time_to_payment']:
    if col in df_payments.columns:
        df_payments[col].fillna(df_payments[col].median(), inplace=True)

for col in ['Research team', 'Type of expense', 'Company', 'Payment Status', 'Submitted by', 'Authorised by', 'Payment authoriser']:
    if col in df_payments.columns:
        df_payments[col].fillna(df_payments[col].mode()[0], inplace=True)

features = ['Invoice value', 'Payment amount', 'time_to_payment', 
            'Research team', 'Type of expense', 'Company', 
            'Payment Status', 'Submitted by', 'Authorised by', 'Payment authoriser']

# Ensure all features exist in df_payments, filter if not
features = [f for f in features if f in df_payments.columns]

X_unsupervised = df_payments[features]

# Identify categorical and numerical features for preprocessing
categorical_features = ['Research team', 'Type of expense', 'Company', 'Payment Status', 'Submitted by', 'Authorised by', 'Payment authoriser']
numerical_features = ['Invoice value', 'Payment amount', 'time_to_payment']

# Filter to only include features actually present in the dataframe
categorical_features = [f for f in categorical_features if f in X_unsupervised.columns]
numerical_features = [f for f in numerical_features if f in X_unsupervised.columns]

# Preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor_unsupervised = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the Isolation Forest model pipeline
# contamination is the proportion of outliers in the dataset (estimate)
model_isolation_forest = Pipeline(steps=[('preprocessor', preprocessor_unsupervised),
                                         ('detector', IsolationForest(random_state=42, contamination=0.05))]) # Assuming 5% outliers

# Fit the model and make predictions (anomaly scores and labels)
model_isolation_forest.fit(X_unsupervised)

anomaly_scores = model_isolation_forest.decision_function(X_unsupervised)
anomaly_predictions = model_isolation_forest.predict(X_unsupervised) # -1 for outliers, 1 for inliers

df_payments['anomaly_score'] = anomaly_scores
df_payments['is_anomaly'] = anomaly_predictions

outliers_unsupervised = df_payments[df_payments['is_anomaly'] == -1]

print(f"\n--- Unsupervised Anomaly Detection (Isolation Forest) ---")
print(f"Identified {len(outliers_unsupervised)} anomalies using Isolation Forest (assuming 5% contamination).")
print("\nTop 10 anomalies (lowest anomaly score):")
print(outliers_unsupervised.sort_values(by='anomaly_score').head(10)[['Invoice number', 'Invoice value', 'time_to_payment', 'anomaly_score']])
