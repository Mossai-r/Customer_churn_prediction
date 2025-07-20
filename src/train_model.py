# ===== train_model.py =====

import sys
import os

#Add this to include the parent directory in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Now this will work correctly
from src.preprocess import preprocess_data

# Load dataset
df = pd.read_csv(r"C:\Users\abhil\OneDrive\Desktop\customer_churn_prediction\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

#  Preprocess the data
X, y, scaler = preprocess_data(df)

#  Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

#  Save the model and scaler
joblib.dump(model, "model/churn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully!")
