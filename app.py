import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# Title
st.title("💳 Credit Card Fraud Detection System")
st.markdown("### Detect whether a transaction is **Fraudulent or Genuine**")

st.write("---")

# Input fields
st.subheader("Enter Transaction Details")

time = st.number_input("Transaction Time", min_value=0.0, format="%.4f")
v1 = st.number_input("V1 Feature", format="%.4f")
v2 = st.number_input("V2 Feature", format="%.4f")
v3 = st.number_input("V3 Feature", format="%.4f")
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")

# Load trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

# Prediction button
if st.button("🔍 Detect Fraud"):

    features = np.array([[time, v1, v2, v3, amount]])
    prediction = model.predict(features)

    st.write("---")

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction")