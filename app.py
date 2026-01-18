import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# --- 1. ASSET LOADING (OPTIMIZED) ---
@st.cache_resource
def load_assets():
    # Loading the artifacts saved during your ML workflow
    model = joblib.load('credit_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

model, scaler, feature_names = load_assets()

# --- 2. STYLING & CONFIGURATION ---
st.set_page_config(page_title="HCL Credit Guard Pro", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { background-color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_stdio=True)

# --- 3. HEADER ---
st.title("🛡️ HCL Credit Guard: Advanced Risk Engine")
st.markdown("Automated AI decision support for credit card and loan applications.")
st.divider()

# --- 4. SIDEBAR: APPLICANT INPUT FORM ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=80)
    st.header("Applicant Portal")
    
    # WRAPPING INPUTS IN A FORM TO ADD SUBMIT OPTION
    with st.form("credit_input_form"):
        st.subheader("Personal & Financials")
        income = st.number_input("Annual Income (₹)", min_value=0, value=800000, step=50000)
        bank_balance = st.number_input("Current Bank Balance (₹)", min_value=0, value=100000)
        
        st.subheader("Loan Details")
        loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, value=400000)
        sanction_amount = st.number_input("Total Debt Sanctioned (₹)", min_value=0, value=200000)
        
        st.subheader("Credit Profile")
        util_ratio = st.slider("Credit Utilization (%)", 0, 100, 35) / 100
        open_acc = st.number_input("Number of Open Accounts", min_value=0, value=5)
        
        # THE SUBMIT BUTTON
        submit_button = st.form_submit_button(label="🚀 RUN RISK ANALYSIS")

# --- 5. PREDICTION & DASHBOARD LOGIC ---
if submit_button:
    # A. Feature Engineering (Replicating the 3 Targeted Ratios)
    dti_ratio = sanction_amount / (income + 1)
    lti_ratio = loan_amount / (income + 1)
    utilization_risk = util_ratio / (open_acc + 1)

    # B. Data Preparation
    input_dict = {
        'income': income, 'loan_amount': loan_amount, 'sanction_amount': sanction_amount,
        'bank_balance_at_application': bank_balance, 'credit_utilization_ratio': util_ratio,
        'number_of_open_accounts': open_acc, 'dti_ratio': dti_ratio, 
        'lti_ratio': lti_ratio, 'utilization_risk': utilization_risk
    }
    input_df = pd.DataFrame([input_dict])
    
    # Aligning with training feature names and order
    for col in feature_names:
        if col not in input_df.columns: input_df[col] = 0
    input_df = input_df[feature_names]

    # C. Model Prediction
    input_scaled = scaler.transform(input_df)
    risk_proba = model.predict_proba(input_scaled)[0][1]
    risk_score = int(risk_proba * 100)

    # D. Display: Gauge vs Recommendation
    col1, col2 = st.columns([1.5, 2])

    with col1:
        # Gauge Chart for Risk Score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            title = {'text': "Default Probability (%)", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "#34495e"},
                'steps': [
                    {'range': [0, 35], 'color': "#2ecc71"}, # Safe
                    {'range': [35, 70], 'color': "#f1c40f"}, # Warning
                    {'range': [70, 100], 'color': "#e74c3c"} # Danger
                ],
                'threshold': {'line': {'color': "white", 'width': 4}, 'value': risk_score}
            }
        ))
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("System Recommendation")
        if risk_score < 35:
            st.success("✅ **APPROVED**: Applicant meets safety criteria.")
        elif risk_score < 70:
            st.warning("⚠️ **MANUAL REVIEW**: Moderate risk factors detected.")
        else:
            st.error("❌ **REJECTED**: High probability of default.")

        st.divider()
        
        # High-Level Metric Cards
        m1, m2, m3 = st.columns(3)
        m1.metric("DTI Ratio", f"{dti_ratio:.2f}", help="Debt-to-Income Ratio")
        m2.metric("LTI Ratio", f"{lti_ratio:.2f}", help="Loan-to-Income Ratio")
        m3.metric("Util. Risk", f"{utilization_risk:.2f}", help="Utilization spread per account")

    # E. Feature Insights (Visualizing why the model decided)
    st.subheader("💡 Analysis Insights")
    st.info("The values below are the primary drivers of your custom engineered risk features.")
    impact_df = pd.DataFrame({
        'Feature': ['Debt Burden', 'Loan Exposure', 'Utilization Risk'],
        'Value': [dti_ratio, lti_ratio, utilization_risk]
    }).set_index('Feature')
    st.bar_chart(impact_df)

else:
    # Initial landing view
    st.info("👈 Fill the applicant details in the sidebar and click 'Run Risk Analysis' to get started.")
    st.image("https://img.icons8.com/clouds/500/safe.png", width=300)
