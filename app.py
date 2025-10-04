import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
model_pipeline = joblib.load('churn_model.pkl')

st.title("Telecom Customer Churn Prediction")

st.write("""
Enter the customer details below to predict whether they are likely to churn.
""")

# --- User inputs ---
state = st.selectbox("State", ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Delhi", "Jammu & Kashmir", "Ladakh",
    "Puducherry", "Chandigarh", "Andaman & Nicobar Islands", "Lakshadweep"])  # Example states
account_length = st.number_input("Account Length", min_value=0, max_value=500, value=100)
area_code = st.selectbox("Area Code", [408, 415, 510])
international_plan = st.selectbox("International Plan", ["yes", "no"])
voice_mail_plan = st.selectbox("Voice Mail Plan", ["yes", "no"])
number_vmail_messages = st.number_input("Number of Voice Mail Messages", min_value=0, max_value=50, value=5)

# Minutes & Calls
total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=400.0, value=200.0)
total_day_calls = st.number_input("Total Day Calls", min_value=0, max_value=200, value=100)

total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, max_value=400.0, value=200.0)
total_eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=200, value=100)

total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, max_value=400.0, value=200.0)
total_night_calls = st.number_input("Total Night Calls", min_value=0, max_value=200, value=100)

total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, max_value=50.0, value=10.0)
total_intl_calls = st.number_input("Total International Calls", min_value=0, max_value=20, value=5)

customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=20, value=1)

# --- Auto-calculated charges ---
total_day_charge = round(total_day_minutes * 0.17, 2)
total_eve_charge = round(total_eve_minutes * 0.085, 2)
total_night_charge = round(total_night_minutes * 0.045, 2)
total_intl_charge = round(total_intl_minutes * 0.27, 2)

# Collect inputs into DataFrame with ALL required columns
user_input = pd.DataFrame({
    "State": [state],
    "Account length": [account_length],
    "Area code": [area_code],
    "International plan": [international_plan],
    "Voice mail plan": [voice_mail_plan],
    "Number vmail messages": [number_vmail_messages],
    "Total day minutes": [total_day_minutes],
    "Total day calls": [total_day_calls],
    "Total day charge": [total_day_charge],
    "Total eve minutes": [total_eve_minutes],
    "Total eve calls": [total_eve_calls],
    "Total eve charge": [total_eve_charge],
    "Total night minutes": [total_night_minutes],
    "Total night calls": [total_night_calls],
    "Total night charge": [total_night_charge],
    "Total intl minutes": [total_intl_minutes],
    "Total intl calls": [total_intl_calls],
    "Total intl charge": [total_intl_charge],
    "Customer service calls": [customer_service_calls]
})

# --- Prediction ---
if st.button("Predict Churn"):
    prediction = model_pipeline.predict(user_input)
    probability = model_pipeline.predict_proba(user_input)

    if prediction[0] == 1:
        st.error(f"The customer is likely to churn! (Probability: {probability[0][1]:.2f})")
    else:
        st.success(f"The customer is not likely to churn. (Probability: {probability[0][1]:.2f})")
