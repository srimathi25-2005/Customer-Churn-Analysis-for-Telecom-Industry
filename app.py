import streamlit as st
import pandas as pd
import pickle
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="ChurnVision AI",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Model and Scaler ---
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Please ensure 'model.pkl' and 'scaler.pkl' are in the correct directory.")
    st.stop()

# --- Custom CSS ---
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Orbitron:wght@400;700&display=swap');

@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes gradientAnimation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp {
    font-family: 'Roboto', sans-serif;
    color: #FAFAFA;
    background-color: #0B0F19;
}
[data-testid="stSidebar"] {
    background-color: rgba(17, 24, 39, 0.9);
    backdrop-filter: blur(8px);
    border-right: 1px solid rgba(255, 255, 255, 0.15);
    padding-top: 2rem;
}
[data-testid="stSidebar"] h1 {
    color: #FFFFFF;
    font-weight: 700;
    text-align: center;
    font-family: 'Orbitron', sans-serif;
}
.sidebar-logo-container {
    text-align: center;
    margin-bottom: 20px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.sidebar-logo-container img {
    max-width: 80px;
    border-radius: 50%;
    box-shadow: 0 0 15px rgba(100, 255, 255, 0.3);
}
h1#customer-churn-prediction {
    color: #00BFFF;
    font-family: 'Orbitron', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
    text-shadow: 0 0 15px rgba(0, 191, 255, 0.6);
}
p.subheading {
    font-size: 1.3rem;
    color: #BBDEFB;
    text-align: center;
    margin-bottom: 40px;
    font-weight: 300;
}
div.stButton {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 40px;
    padding-right: 5rem;
}
.stButton>button {
    color: #FFFFFF;
    border: none;
    padding: 18px 40px;
    border-radius: 50px;
    font-size: 20px;
    font-weight: 700;
    transition: all 0.4s ease-in-out;
    background: linear-gradient(-45deg, #00BFFF, #1E90FF, #00BFFF, #1E90FF);
    background-size: 400% 400%;
    animation: gradientAnimation 15s ease infinite;
    box-shadow: 0 8px 30px rgba(0, 191, 255, 0.5);
    cursor: pointer;
    min-width: 300px;
}
.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 12px 40px rgba(30, 144, 255, 0.6);
}
.result-card {
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 35px;
    text-align: center;
    margin-top: 30px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    animation: fadeIn 0.8s ease-out;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.stay {
    box-shadow: 0 0 35px 0 rgba(34, 197, 94, 0.7);
    border-left: 6px solid #22C55E;
}
.churn {
    box-shadow: 0 0 35px 0 rgba(239, 68, 68, 0.7);
    border-left: 6px solid #EF4444;
}
.result-icon {
    font-size: 60px;
    margin-bottom: 15px;
}
.result-text {
    font-size: 34px;
    font-weight: 700;
    margin-top: 10px;
    font-family: 'Orbitron', sans-serif;
}
.confidence-text {
    font-size: 24px;
    margin-top: 15px;
    opacity: 0.95;
    font-weight: 300;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo-container">
            <img src="https://i.imgur.com/v0gACsum.png">
            <h1>ChurnVision AI</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("### Enter customer details to predict churn risk.")
    
    gender = st.selectbox('Gender', ('Male', 'Female'))
    senior_citizen = st.selectbox('Senior Citizen', (0, 1))
    partner = st.selectbox('Partner', ('Yes', 'No'))
    dependents = st.selectbox('Dependents', ('Yes', 'No'))
    tenure = st.slider('Tenure (months)', 1, 72, 24)
    phone_service = st.selectbox('Phone Service', ('Yes', 'No'))
    multiple_lines = st.selectbox('Multiple Lines', ('No phone service', 'No', 'Yes'))
    internet_service = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    online_security = st.selectbox('Online Security', ('No internet service', 'No', 'Yes'))
    online_backup = st.selectbox('Online Backup', ('No internet service', 'No', 'Yes'))
    device_protection = st.selectbox('Device Protection', ('No internet service', 'No', 'Yes'))
    tech_support = st.selectbox('Tech Support', ('No internet service', 'No', 'Yes'))
    streaming_tv = st.selectbox('Streaming TV', ('No internet service', 'No', 'Yes'))
    streaming_movies = st.selectbox('Streaming Movies', ('No internet service', 'No', 'Yes'))
    contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.selectbox('Paperless Billing', ('Yes', 'No'))
    payment_method = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthly_charges = st.number_input('Monthly Charges', min_value=18.0, max_value=120.0, value=70.0, format="%.2f")
    total_charges = st.number_input('Total Charges', min_value=18.0, max_value=9000.0, value=1500.0, format="%.2f")

# --- Main Panel ---
st.markdown("<h1 id='customer-churn-prediction'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheading'>Leverage AI to accurately foresee customer departures and bolster retention strategies.</p>", unsafe_allow_html=True)

# --- Prediction Logic ---
if st.button('Predict Churn Risk'):
    data = {
        'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }

    input_df = pd.DataFrame(data, index=[0])

    # Encoding
    input_df_encoded = input_df.copy()
    input_df_encoded['gender'] = input_df_encoded['gender'].map({'Female': 0, 'Male': 1})
    input_df_encoded['Partner'] = input_df_encoded['Partner'].map({'No': 0, 'Yes': 1})
    input_df_encoded['Dependents'] = input_df_encoded['Dependents'].map({'No': 0, 'Yes': 1})
    input_df_encoded['PhoneService'] = input_df_encoded['PhoneService'].map({'No': 0, 'Yes': 1})
    input_df_encoded['MultipleLines'] = input_df_encoded['MultipleLines'].map({'No phone service': 0, 'No': 1, 'Yes': 2})
    input_df_encoded['InternetService'] = input_df_encoded['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    input_df_encoded['OnlineSecurity'] = input_df_encoded['OnlineSecurity'].map({'No internet service': 0, 'No': 1, 'Yes': 2})
    input_df_encoded['OnlineBackup'] = input_df_encoded['OnlineBackup'].map({'No internet service': 0, 'No': 1, 'Yes': 2})
    input_df_encoded['DeviceProtection'] = input_df_encoded['DeviceProtection'].map({'No internet service': 0, 'No': 1, 'Yes': 2})
    input_df_encoded['TechSupport'] = input_df_encoded['TechSupport'].map({'No internet service': 0, 'No': 1, 'Yes': 2})
    input_df_encoded['StreamingTV'] = input_df_encoded['StreamingTV'].map({'No internet service': 0, 'No': 1, 'Yes': 2})
    input_df_encoded['StreamingMovies'] = input_df_encoded['StreamingMovies'].map({'No internet service': 0, 'No': 1, 'Yes': 2})
    input_df_encoded['Contract'] = input_df_encoded['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    input_df_encoded['PaperlessBilling'] = input_df_encoded['PaperlessBilling'].map({'No': 0, 'Yes': 1})
    input_df_encoded['PaymentMethod'] = input_df_encoded['PaymentMethod'].map({
        'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3
    })

    with st.spinner('Running AI analysis...'):
        time.sleep(1.5)
        input_scaled = scaler.transform(input_df_encoded)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

    # --- Display Results ---
    if prediction[0] == 1:
        churn_probability = prediction_proba[0][1]
        st.markdown(f"""
        <div class="result-card churn">
            <p class="result-icon">⚠️</p>
            <p class="result-text">Prediction: Customer will CHURN</p>
            <p class="confidence-text">Confidence: {churn_probability*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # --- Strong Key Churn Drivers ---
        st.subheader("🔍 Key Churn Drivers (Root Causes)")
        churn_drivers = []
        if contract == "Month-to-month":
            churn_drivers.append("Unstable short-term contract (Month-to-month)")
        if tenure < 12:
            churn_drivers.append("Low customer loyalty (new user under 1 year)")
        if tech_support == "No":
            churn_drivers.append("Poor customer experience - No technical support available")
        if online_security == "No":
            churn_drivers.append("Lack of online security services")
        if monthly_charges > 85:
            churn_drivers.append("High monthly expenditure compared to average user")
        if internet_service == "Fiber optic":
            churn_drivers.append("High-cost Fiber Optic plan without added value")
        if payment_method == "Electronic check":
            churn_drivers.append("Outdated payment method leading to low satisfaction")
        if paperless_billing == "Yes":
            churn_drivers.append("Digital-only interaction – lower engagement with company")

        for d in churn_drivers:
            st.markdown(f"• {d}")

        # --- Strong Retention Offers ---
        st.subheader("💡 Strategic Retention Offers (AI-Recommended)")
        recommendations = []
        if contract == "Month-to-month":
            recommendations.append("🎁 Offer 15% discount for upgrading to annual contract.")
        if tenure < 12:
            recommendations.append("🎉 Provide loyalty bonus or cashback for customers under 1 year.")
        if tech_support == "No":
            recommendations.append("🛠️ Give 6-month free tech support to improve satisfaction.")
        if online_security == "No":
            recommendations.append("🛡️ Provide free Online Security for 3 months as retention incentive.")
        if monthly_charges > 85:
            recommendations.append("💰 Suggest downgrade plan or loyalty-based discount.")
        if internet_service == "Fiber optic":
            recommendations.append("📦 Offer bundle package (TV + Internet) for better value.")
        if payment_method == "Electronic check":
            recommendations.append("💳 Encourage switch to auto-pay using credit card or bank transfer.")
        if paperless_billing == "Yes":
            recommendations.append("📨 Send personalized digital engagement campaigns to increase retention.")

        for r in recommendations:
            st.markdown(f"✅ {r}")

    else:
        stay_probability = prediction_proba[0][0]
        st.markdown(f"""
        <div class="result-card stay">
            <p class="result-icon">✔️</p>
            <p class="result-text">Prediction: Customer will STAY</p>
            <p class="confidence-text">Confidence: {stay_probability*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
