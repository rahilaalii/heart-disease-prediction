import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")


@st.cache_resource
def load_files():
    model = joblib.load("heart_disease_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, encoders


model, encoders = load_files()

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below. (AIC354 – Machine Learning Fundamentals, Assignment 2)")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", list(encoders["sex"].classes_))
cp = st.selectbox("Chest Pain Type", list(encoders["cp"].classes_))
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)

if st.button("Predict"):
    row = pd.DataFrame(
        [[age, encoders["sex"].transform([sex])[0], encoders["cp"].transform([cp])[0], thalach]],
        columns=["age", "sex", "cp", "thalach"],
    )
    prediction = model.predict(row)[0]
    if prediction == 1:
        st.error("⚠️ Heart Disease Likely")
    else:
        st.success("✅ No Heart Disease Likely")

st.caption("This is a student project for learning only. Not for real medical decisions.")
