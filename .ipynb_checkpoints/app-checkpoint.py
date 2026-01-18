import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved components
model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="HCL Credit Guard AI", layout="wide")

st.title("🛡️ HCL Credit Guard: AI Loan Approval System")
st.markdown("Enter applicant details below to evaluate credit risk in real-time.")

# 2. Sidebar for User Inputs
with st.sidebar:
    st.header("Applicant Information")
    income = st.number_input("Annual Income (INR)", min_value=0, value=500000)
    loan_amount = st.number_input("Requested Loan Amount", min_value=0, value=200000)
    sanction_amount = st.number_input("Total Debt Sanctioned", min_value=0, value=100000)
    bank_balance = st.number_input("Bank Balance at Application", min_value=0, value=50000)
    util_ratio = st.slider("Current Credit Utilization (%)", 0, 100, 30) / 100
    open_acc = st.number_input("Number of Open Accounts", min_value=0, value=3)

# 3. Automatic Feature Engineering (The 'Secret Sauce' Logic)
dti_ratio = sanction_amount / (income + 1)
lti_ratio = loan_amount / (income + 1)
utilization_risk = util_ratio / (open_acc + 1)

# 4. Prepare data for prediction
input_data = pd.DataFrame([[income, loan_amount, sanction_amount, bank_balance, util_ratio, open_acc, dti_ratio, lti_ratio, utilization_risk]], 
                         columns=['income', 'loan_amount', 'sanction_amount', 'bank_balance_at_application', 
                                  'credit_utilization_ratio', 'number_of_open_accounts', 'dti_ratio', 'lti_ratio', 'utilization_risk'])

# Ensure all training features exist (filling missing ones with 0)
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[feature_names] # Reorder columns to match model training

# 5. Scale and Predict
input_scaled = scaler.transform(input_data)
risk_proba = model.predict_proba(input_scaled)[0][1]
risk_score = int(risk_proba * 100)

# 6. Display Results
st.subheader("📋 Credit Risk Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Risk Score", f"{risk_score}/100")

with col2:
    if risk_score < 30:
        st.success("Decision: LOW RISK (Approve)")
    elif risk_score < 60:
        st.warning("Decision: MODERATE RISK (Review)")
    else:
        st.error("Decision: HIGH RISK (Reject)")

with col3:
    st.info(f"DTI Ratio: {dti_ratio:.2f}")

st.progress(risk_score)