import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# TITLE
# -----------------------------
st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("### AI-powered churn prediction system (Enhanced with Notebook Logic)")

# -----------------------------
# LOAD & PREPROCESS DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = df.drop('customerID', axis=1)
    
    # Convert categorical → numeric using get_dummies (matches notebook)
    df = pd.get_dummies(df, drop_first=True)
    
    return df

df = load_data()

# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(df):
    target = 'Churn_Yes'
    X = df.drop(target, axis=1)
    y = df[target]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X.columns

model, feature_columns = train_model(df)

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Enter Customer Details")

# Helper for binary inputs
def binary_to_val(val):
    return 1 if val == "Yes" else 0

# Numerical inputs
tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 10.0, 150.0, 50.0)
total_charges = st.sidebar.slider("Total Charges", 10.0, 10000.0, 1000.0)

# Categorical inputs based on dataset
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.checkbox("Senior Citizen", value=False)
partner = st.sidebar.checkbox("Partner", value=False)
dependents = st.sidebar.checkbox("Dependents", value=False)
phone_service = st.sidebar.checkbox("Phone Service", value=True)
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.checkbox("Paperless Billing", value=True)
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# -----------------------------
# PREDICTION
# -----------------------------
if st.sidebar.button("Predict"):

    # Map the inputs to a dictionary that matches the raw dataframe columns (pre-dummies)
    input_dict = {
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen else 0],
        'Partner': ['Yes' if partner else 'No'],
        'Dependents': ['Yes' if dependents else 'No'],
        'tenure': [tenure],
        'PhoneService': ['Yes' if phone_service else 'No'],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': ['Yes' if paperless_billing else 'No'],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    }

    # Create dummy dataframe for prediction
    input_df = pd.DataFrame(input_dict)
    
    # We need to perform the same dummy encoding as we did for training
    # But st.cache_data df is already encoded. We need to match feature_columns.
    
    # Method: Create a full dataframe with the single input row, then use get_dummies
    # To handle categorical values consistently, we can use CategoricalDtype or just manually set columns
    
    # A safer way: Start with a zeroed dataframe of feature_columns
    prediction_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Fill numericals
    prediction_df['tenure'] = tenure
    prediction_df['MonthlyCharges'] = monthly_charges
    prediction_df['TotalCharges'] = total_charges
    prediction_df['SeniorCitizen'] = 1 if senior_citizen else 0
    
    # Fill dummies
    def set_dummy(col_prefix, val):
        col_name = f"{col_prefix}_{val}"
        if col_name in prediction_df.columns:
            prediction_df[col_name] = 1

    set_dummy('gender', gender)
    set_dummy('Partner', 'Yes' if partner else 'No')
    set_dummy('Dependents', 'Yes' if dependents else 'No')
    set_dummy('PhoneService', 'Yes' if phone_service else 'No')
    set_dummy('MultipleLines', multiple_lines)
    set_dummy('InternetService', internet_service)
    set_dummy('OnlineSecurity', online_security)
    set_dummy('OnlineBackup', online_backup)
    set_dummy('DeviceProtection', device_protection)
    set_dummy('TechSupport', tech_support)
    set_dummy('StreamingTV', streaming_tv)
    set_dummy('StreamingMovies', streaming_movies)
    set_dummy('Contract', contract)
    set_dummy('PaperlessBilling', 'Yes' if paperless_billing else 'No')
    set_dummy('PaymentMethod', payment_method)

    # Reorder columns to match training
    prediction_df = prediction_df[feature_columns]

    # Prediction
    prediction = model.predict(prediction_df)[0]
    probability = model.predict_proba(prediction_df)[0][1]

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"❌ Customer is likely to churn\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer will stay\n\nProbability: {probability:.2f}")

# -----------------------------
# SHOW DATA (OPTIONAL)
# -----------------------------
if st.checkbox("Show Dataset (Training)"):
    st.write(df.head())