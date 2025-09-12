import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title and description
st.title("Telecom Churn Prediction App")
st.write("""
This app predicts whether a telecom customer will churn using a logistic regression model.
Enter the customer details below. (Built using the churn-bigml dataset.)
""")

# Input fields for features (based on your dataset)
st.header("Customer Details")
account_length = st.number_input("Account Length (months)", min_value=1, max_value=300, value=100)
total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=400.0, value=180.0)
total_day_calls = st.number_input("Total Day Calls", min_value=0, max_value=200, value=100)
total_day_charge = st.number_input("Total Day Charge ($)", min_value=0.0, max_value=100.0, value=30.0)
total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, max_value=400.0, value=200.0)
total_eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=200, value=100)
total_eve_charge = st.number_input("Total Evening Charge ($)", min_value=0.0, max_value=100.0, value=17.0)
total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, max_value=400.0, value=200.0)
total_night_calls = st.number_input("Total Night Calls", min_value=0, max_value=200, value=100)
total_night_charge = st.number_input("Total Night Charge ($)", min_value=0.0, max_value=100.0, value=9.0)
total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, max_value=50.0, value=10.0)
total_intl_calls = st.number_input("Total International Calls", min_value=0, max_value=20, value=3)
total_intl_charge = st.number_input("Total International Charge ($)", min_value=0.0, max_value=20.0, value=2.7)
customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=10, value=1)
international_plan = st.selectbox("International Plan", ["No", "Yes"])
voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, max_value=100, value=0)

# Encode categorical variables
international_plan_encoded = 1 if international_plan == "Yes" else 0
voice_mail_plan_encoded = 1 if voice_mail_plan == "Yes" else 0

# Note: Excluding 'State' and 'Area Code' assuming they weren't used in the final model.
# If they were, add encoding (e.g., one-hot encoding) here.

# Prepare input data for prediction
input_data = np.array([[
    account_length, total_day_minutes, total_day_calls, total_day_charge,
    total_eve_minutes, total_eve_calls, total_eve_charge, total_night_minutes,
    total_night_calls, total_night_charge, total_intl_minutes, total_intl_calls,
    total_intl_charge, customer_service_calls, international_plan_encoded,
    voice_mail_plan_encoded, number_vmail_messages
]])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.error("[1] The customer is predicted to churn.")
    else:
        st.success("[0] The customer is predicted to not churn.")

# Display model performance (from notebook)
st.header("Model Performance")
st.write("The logistic regression model has an AUC-ROC score of approximately 0.834 (from the training notebook).")