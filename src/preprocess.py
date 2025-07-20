
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    # Drop unnecessary columns safely
    columns_to_drop = ['customerID']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Convert TotalCharges to numeric, fill missing values with median
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Label Encoding for all object (categorical) columns except 'Churn'
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        if col != 'Churn':
            df[col] = le.fit_transform(df[col])

    # Check if 'Churn' exists (for training vs prediction mode)
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
        y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    else:
        X = df
        y = None

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

