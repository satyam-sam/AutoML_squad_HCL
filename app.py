import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# --- 1. SECURE ASSET LOADING ---
@st.cache_resource
def load_assets():
    # These must exist in your GitHub repository
    model = joblib.load('credit_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

try:
    model, scaler, feature_names = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}. Ensure .pkl files are in the repository.")
    st.stop()

# --- 2. LAYOUT & STYLE ---
st.set_page_config(page_title="HCL Credit Guard Pro", layout="wide", page_icon="🏦")
st.title("🛡️ HCL Credit Guard: Advanced Risk Engine")
st.markdown("Automated AI decision support for high-volume loan applications.")
st.divider()

# --- 3. INPUT FORM (Ensures all data is captured before prediction) ---
with st.sidebar:
    st.header("Applicant Portal")
    with st.form("risk_input_form"):
        st.subheader("Financials")
        income = st.number_input("Annual Income (₹)", min_value=1, value=800000)
        bank_balance = st.number_input("Current Bank Balance (₹)", min_value=0, value=100000)
        
        st.subheader("Loan Details")
        loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, value=400000)
        sanction_amount = st.number_input("Existing Sanctioned Debt (₹)", min_value=0, value=200000)
        
        st.subheader("Credit Profile")
        util_ratio = st.slider("Credit Utilization (%)", 0, 100, 35) / 100
        open_acc = st.number_input("Number of Open Accounts", min_value=1, value=5)
        
        submit_button = st.form_submit_button(label="🚀 RUN RISK ANALYSIS")

# --- 4. THE ASSESSMENT LOGIC (Checked for Accuracy) ---
if submit_button:
    # A. Feature Engineering (Must match the training logic exactly)
    # We add 1 to denominators to prevent division by zero errors
    dti_ratio = sanction_amount / (income + 1)
    lti_ratio = loan_amount / (income + 1)
    utilization_risk = util_ratio / (open_acc + 1)

    # B. Create the input dictionary with exact names from training
    input_dict = {
        'income': income, 
        'loan_amount': loan_amount, 
        'sanction_amount': sanction_amount,
        'bank_balance_at_application': bank_balance, 
        'credit_utilization_ratio': util_ratio,
        'number_of_open_accounts': open_acc, 
        'dti_ratio': dti_ratio, 
        'lti_ratio': lti_ratio, 
        'utilization_risk': utilization_risk
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # C. Handle Missing Columns (Safety check)
    # If the model was trained on more columns (like one-hot encoded categories), we add them as 0
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # D. Reorder columns to match the model's training order exactly
    input_df = input_df[feature_names]

    # E. Scale the data using the saved Scaler
    # We only scale the numerical columns that the scaler was fitted on
    input_scaled = scaler.transform(input_df)

    # F. Generate Probability
    risk_proba = model.predict_proba(input_scaled)[0][1]
    risk_score = int(risk_proba * 100)

    # --- 5. VISUAL RESULTS ---
    col1, col2 = st.columns([1, 1.5])

    with col1:
        # Professional Risk Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            title = {'text': "Default Risk Score (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 35], 'color': "#2ecc71"},
                    {'range': [35, 70], 'color': "#f1c40f"},
                    {'range': [70, 100], 'color': "#e74c3c"}
                ],
                'threshold': {'line': {'color': "white", 'width': 4}, 'value': risk_score}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Official Recommendation")
        if risk_score < 35:
            st.success("✅ **APPROVED**: Applicant meets the safety threshold.")
        elif risk_score < 70:
            st.warning("⚠️ **MANUAL REVIEW**: Borderline risk detected.")
        else:
            st.error("❌ **REJECTED**: High probability of credit default.")
        
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("DTI Ratio", f"{dti_ratio:.2f}")
        m2.metric("LTI Ratio", f"{lti_ratio:.2f}")
        m3.metric("Util. Risk", f"{utilization_risk:.2f}")

else:
    st.info("👈 Enter applicant data in the sidebar and click the button to assess risk.")
