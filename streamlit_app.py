
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("vehicle_model.pkl")

# App Title and Branding
st.set_page_config(page_title="Car Model Predictor", layout="centered")
st.markdown(
    "<div style='text-align:center;'>"
    "<img src='https://img1.wsimg.com/blobby/go/72e115c3-a759-4da3-a901-ced48b652c56/downloads/f8492d45-cda7-4fcd-bba3-1c0a54a826d6/14%20MOH%20NEW%20LOGO%20H%20K%2B285.png?ver=1738832769439' height='100'>"
    "<h1 style='color:#0073C2;'>Car Model Prediction</h1>"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div style='background-color: white; padding: 30px; border-radius: 15px; "
    "box-shadow: 0 10px 25px rgba(0,0,0,0.1);'>",
    unsafe_allow_html=True
)

st.header("Input Customer Details")
age_range = st.selectbox("Age Range", ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"])
income_level = st.selectbox("Income Level", ["Under $50,000", "$50,000–$74,999", "$75,000–$99,999", "$100,000–$149,999"])
gender = st.selectbox("Gender", ["Female", "Male"])
zip_code = st.text_input("ZIP Code", "12345")
budget = st.number_input("Customer Budget ($)", min_value=5000, max_value=100000, step=1000, value=25000)

if st.button("🚀 Predict Model"):
    input_df = pd.DataFrame([[age_range, income_level, gender, int(zip_code), float(budget)]],
                            columns=["AgeRange_Clean", "IncomeLevel_Clean", "Gender", "ADJZipCode", "SellingPrice"])

    probs = model.predict_proba(input_df)[0]
    labels = model.classes_
    prob_df = pd.DataFrame({"Model": labels, "Probability": probs})
    top_3 = prob_df.sort_values("Probability", ascending=False).head(3)

    st.subheader("Top 3 Predicted Vehicle Models:")
    for _, row in top_3.iterrows():
        pct = round(row["Probability"] * 100, 1)
        st.markdown(f"**{row['Model']}**")
        st.progress(pct / 100)

st.markdown("</div>", unsafe_allow_html=True)
