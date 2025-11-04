import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    df = pd.read_csv(payment_master_path)
except FileNotFoundError:
    print("Error: payments_master.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Convert date columns to datetime objects
date_cols = ['Date received', 'Date of invoice', 'Date of authorisation', 'Payment due date', 'Date of payment']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Calculate 'time to payment' in days
df['time_to_payment'] = (df['Date of payment'] - df['Date received']).dt.days

# Drop rows where 'time_to_payment' is NaN (due to missing date of payment or date received)
df.dropna(subset=['time_to_payment'], inplace=True)

# Select features and target
# For initial analysis, we'll consider numerical and categorical features that might influence payment time.
# 'Invoice value', 'Payment amount' are numerical.
# 'Research team', 'Type of expense', 'Company' are categorical.
features = ['Invoice value', 'Payment amount', 'Research team', 'Type of expense', 'Company']
target = 'time_to_payment'

X = df[features]
y = df[target]

# Identify categorical and numerical features
categorical_features = ['Research team', 'Type of expense', 'Company']
numerical_features = ['Invoice value', 'Payment amount']

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
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Hypothesis testing (simple check based on MAE)
# The hypothesis is that all invoices take ~same time +/- 1 day.
# We can check if our MAE is close to or less than 1.
if mae <= 1:
    print("The model supports the hypothesis: Invoices take approximately the same time (within +/- 1 day).")
else:
    print(f"The model does not strongly support the hypothesis: MAE ({mae:.2f} days) is greater than 1 day.")
