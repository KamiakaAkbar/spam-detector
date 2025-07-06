# app.py

import streamlit as st
import joblib

# Load model
model = joblib.load('spam_model.pkl')

# App layout
st.title("📩 Spam Message Classifier")
st.markdown("Type a message and check if it's **Spam** or **Not Spam**.")

# Input
msg = st.text_input("Enter your message here:")

# Prediction
if st.button("Predict"):
    prediction = model.predict([msg])[0]
    if prediction == 'spam':
        st.error("⚠️ This message is classified as **SPAM**.")
    else:
        st.success("✅ This message is **NOT SPAM**.")
