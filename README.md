# 🛡️ HCL Credit Guard: AI-Driven Credit Risk Scoring

This repository contains the end-to-end Machine Learning pipeline developed for the **HCL Tech AutoML Hackathon**. The solution is designed to predict loan default probability using advanced Gradient Boosting techniques, ensuring the protection of bank capital while maintaining operational efficiency.

## 🚀 Project Overview
**HCL Credit Guard** is a "Production-Ready" decision support system. It transforms raw financial and behavioral data into actionable credit scores, meeting strict industry standards for model accuracy, class balance, and explainability.

---

## 🛠️ Technical Architecture

### 1. Data Preprocessing & Cleansing
* **Targeted Imputation**: Handled missing values using median strategies to maintain distribution integrity across the dataset.
* **Feature Scaling**: Implemented `StandardScaler` to normalize features like `income` and `loan_amount`, ensuring the model is not biased by raw numerical magnitudes.
* **Outlier Management**: Leveraged the robust binning capabilities of Gradient Boosting Machines to neutralize the impact of extreme financial values without losing data.

### 2. Advanced Feature Engineering ("The Secret Sauce")
To demonstrate technical maturity, we engineered three key financial ratios that serve as the primary risk drivers:
* **DTI (Debt-to-Income)**: Measures the applicant's total debt burden relative to their earnings.
* **LTI (Loan-to-Income)**: Evaluates the size of the requested loan against the applicant's annual income.
* **Utilization Risk**: Analyzes credit usage spread across all open accounts to detect potential over-leveraging.

### 3. Model Training & Class Imbalance
* **Algorithm Selection**: Utilized **LightGBM** (HCL Preferred) for its speed and efficiency with large-scale tabular data.
* **Class Imbalance Mitigation**: Applied **RandomOverSampler** to ensure the model accurately learns "Defaulter" patterns, fixing the common bias where models "approve everyone" to gain accuracy.
* **Benchmarking**: Validated results against an **XGBoost** challenger to ensure the most robust decision-making.



---

## 📊 Performance Metrics
The model was evaluated against HCL’s strict technical specifications:
* **AUC-ROC**: Achieved **> 0.80**, meeting the HCL requirement for predictive power.
* **Recall (Default Class)**: Optimized to ensure risky applicants are correctly identified and rejected, protecting the bank’s principal.
* **KS Statistic**: Calculated to confirm strong separation between "Good" and "Bad" customers.

---

## 💻 Live Deployment
The solution is deployed via an interactive **Streamlit Dashboard**:
* **Form-Based Input**: Uses `st.form` to capture all applicant data before triggering the AI engine.
* **Risk Gauge**: A professional Plotly-based "Speedometer" visualizing the probability of default.
* **Decision Logic**: Provides clear "Approve / Review / Reject" status based on custom risk thresholds.



---

## 📁 Repository Structure
* `app.py`: Final Streamlit Dashboard application.
* `requirements.txt`: List of dependencies (lightgbm, xgboost, plotly, etc.).
* `credit_risk_model.pkl`: The trained and optimized LightGBM model.
* `scaler.pkl`: Saved `StandardScaler` object for real-time data transformation.
* `feature_names.pkl`: List of features used during training to ensure data alignment.

## ⚙️ Setup & Usage
1.  **Clone the Repo**: `git clone <repo-url>`
2.  **Install Dependencies**: `pip install -r requirements.txt`
3.  **Run Locally**: `streamlit run app.py`

---

**Developed for HCL Tech Hackathon 2026**
