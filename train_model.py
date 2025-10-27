import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# --- 1. Load Data ---
df = pd.read_csv("C:/Users/sival/Customer Churn Analysis/Telecom_Dataset.csv")

# --- 2. Preprocessing ---
# Drop customerID
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill NaN values in TotalCharges with the mean
mean_total_charges = df['TotalCharges'].mean()
df['TotalCharges'] = df['TotalCharges'].fillna(mean_total_charges)

# Remove rows where tenure is 0 (if any)
df = df[df['tenure'] != 0]

# Encode categorical variables
for column in df.columns:
    if df[column].dtype == 'object' and column != 'Churn': # Ensure Churn is not encoded here
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

# --- 3. Feature Scaling and Splitting ---
# Define features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0) # Ensure y is 0 or 1

# Split data before scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Model Training ---
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --- 5. Save the Model and Scaler ---
# Use pickle to save the trained model and the FITTED scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler have been saved successfully.")