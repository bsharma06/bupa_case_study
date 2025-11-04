
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
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
    df_fraud = pd.read_csv(fraud_case_master_path)
except FileNotFoundError:
    print("Error: fraud_cases_master.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Convert date columns to datetime objects if needed (for potential feature engineering later)
date_cols = ['Date received', 'Date of invoice', 'Date of authorisation', 'Payment due date', 'Date of payment']
for col in date_cols:
    if col in df_fraud.columns:
        df_fraud[col] = pd.to_datetime(df_fraud[col], errors='coerce')

# Calculate 'time to payment' as a potential feature
df_fraud['time_to_payment'] = (df_fraud['Date of payment'] - df_fraud['Date received']).dt.days

# Define features (X) and target (y)
# We'll use a similar set of features as in the regression analysis, plus any potentially relevant new ones.
# 'Invoice value', 'Payment amount', 'time_to_payment' are numerical.
# 'Research team', 'Type of expense', 'Company', 'Payment Status' are categorical.

# Drop original date columns as we have 'time_to_payment'
df_fraud = df_fraud.drop(columns=date_cols, errors='ignore')

# Handle potential missing values (e.g., for 'time_to_payment') before splitting
df_fraud.dropna(subset=['time_to_payment'], inplace=True)

X = df_fraud.drop('Fraud flag', axis=1)
y = df_fraud['Fraud flag']

# Identify categorical and numerical features for preprocessing
categorical_features = ['Research team', 'Type of expense', 'Company', 'Payment Status', 'Submitted by', 'Authorised by', 'Payment authoriser']
numerical_features = ['Invoice value', 'Payment amount', 'time_to_payment']

# Ensure all features exist in X, filter if not
categorical_features = [f for f in categorical_features if f in X.columns]
numerical_features = [f for f in numerical_features if f in X.columns]

X = X[categorical_features + numerical_features]

# Preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model pipeline
model_fraud = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier(random_state=42))])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify for imbalanced classes

# Train the model
model_fraud.fit(X_train, y_train)

# Make predictions
y_pred = model_fraud.predict(X_test)
y_pred_proba = model_fraud.predict_proba(X_test)[:, 1] # Probability for ROC-AUC

# Evaluate the model
print("\n--- Supervised Model Evaluation for Fraud Detection ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.2f}")
