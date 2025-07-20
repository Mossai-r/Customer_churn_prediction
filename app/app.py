# app/app.py

import streamlit as st
import pandas as pd
import joblib
import os
import sys

# --- GLOBAL CUSTOM STYLE ---
with open("style.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import preprocess_data

model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.markdown("<h1 style='text-align:center;margin-bottom:-0.5em;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:1.10em; color:#cbd6e6; margin-top:0;'>"
    "Predict whether a customer will churn — in seconds. Enter details below."
    "</p>", unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# === Input Form ===
with st.form("churn_form"):
    st.markdown("##### Enter Customer Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ['Yes', 'No'])
        Dependents = st.selectbox("Dependents", ['Yes', 'No'])
        tenure = st.number_input("Tenure (months)", min_value=0)
        PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
        MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    with col2:
        InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
        TechSupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
        StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    Contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
    PaymentMethod = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    user_df = pd.DataFrame([input_data])
    base_df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    base_df = base_df.drop(columns=['customerID', 'Churn'], errors='ignore')
    combined_df = pd.concat([base_df, user_df], ignore_index=True)
    X_scaled, _, _ = preprocess_data(combined_df)
    final_input = X_scaled[-1].reshape(1, -1)
    prediction = model.predict(final_input)[0]
    proba = model.predict_proba(final_input)[0][1]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Prediction Result")
    if prediction == 1:
        st.markdown(
            f"<div style='background:#232b3a;padding:1.2em 1.6em;border-radius:1em;box-shadow:0 2px 16px #d74e5c43;margin-bottom:1em;'><span style='color:#ee4367;font-size:1.1em;font-weight:600;'>The customer is <u>likely to churn</u>.</span>"
            f"<br/><span style='font-size:1.04em;color:#f95d6a;'>Estimated Probability: <b>{proba:.2%}</b></span></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background:#222e37;padding:1.2em 1.6em;border-radius:1em;box-shadow:0 2px 16px #4eb57a33;margin-bottom:1em;'><span style='color:#3ddc97;font-size:1.1em;font-weight:600;'>The customer is <u>likely to stay</u>.</span>"
            f"<br/><span style='font-size:1.04em;color:#43eaa2;'>Estimated Probability: <b>{1-proba:.2%}</b></span></div>",
            unsafe_allow_html=True
        )

st.caption("Built for performance and clarity — powered by Streamlit, by [Your Name/Team].")
