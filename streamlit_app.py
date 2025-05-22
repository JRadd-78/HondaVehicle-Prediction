
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("vehicle_model.pkl")

# Set page configuration
st.set_page_config(page_title="Car Model Prediction", layout="centered")

# App header
st.markdown("""
    <div style='text-align: center;'>
        <img src="https://img1.wsimg.com/blobby/go/72e115c3-a759-4da3-a901-ced48b652c56/downloads/f8492d45-cda7-4fcd-bba3-1c0a54a826d6/14%20MOH%20NEW%20LOGO%20H%20K%2B285.png?ver=1738832769439" height="100">
        <h1 style='color: #0073C2;'>Car Model Prediction</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
        .main-container {
            background-color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            margin-top: 30px;
        }
        .progress {
            height: 30px;
        }
        .progress-bar {
            font-size: 16px;
            font-weight: bold;
            line-height: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

# Load ZIP code and category options from training data
zip_codes = sorted(model.named_steps["encoder"].categories_[3].astype(str).tolist())
age_ranges = model.named_steps["encoder"].categories_[0].tolist()
income_levels = model.named_steps["encoder"].categories_[1].tolist()
genders = model.named_steps["encoder"].categories_[2].tolist()

# User input
with st.form("prediction_form"):
    st.subheader("Input Customer Details")
    selling_price = st.number_input("Customer Budget:", min_value=5000, max_value=100000, value=25000, step=1000)
    zip_code = st.selectbox("ZIP Code:", zip_codes)
    age_range = st.selectbox("Age Range:", age_ranges)
    income_level = st.selectbox("Income Level:", income_levels)
    gender = st.selectbox("Gender:", genders)
    submitted = st.form_submit_button("🚀 Predict Model")

if submitted:
    # Prepare input for model
    input_data = pd.DataFrame([[age_range, income_level, gender, zip_code, selling_price]],
                              columns=["AgeRange_Clean", "IncomeLevel_Clean", "Gender", "ADJZipCode", "SellingPrice"])
    # Predict probabilities
    try:
        proba = model.predict_proba(input_data)[0]
        labels = model.named_steps["classifier"].classes_
        predictions = sorted(zip(labels, proba), key=lambda x: x[1], reverse=True)[:3]

        st.subheader("Top 3 Predicted Vehicle Models")
        for model_name, prob in predictions:
            pct = round(prob * 100, 1)
            color = "success" if pct >= 80 else "warning" if pct >= 50 else "danger"
            st.markdown(f"<h4>{model_name}</h4>", unsafe_allow_html=True)
            st.progress(pct / 100)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p style='color: #7f8c8d;'>These predictions are based on customer demographics and preferences.</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
