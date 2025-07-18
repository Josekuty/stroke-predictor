import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Stroke Risk Prediction", layout="centered")

# Load model and column structure
model = joblib.load("stroke_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Load dataset for encoding consistency
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_o.csv")
    df = df.dropna()
    df = df[df['smoking_status'] != 'Unknown']  # remove 'Unknown' values
    return df

df = load_data()

# App Title
st.title("🧠 Stroke Risk Predictor")
st.write("Enter patient details below to predict stroke risk.")

# Input fields in the main area (no sidebar)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
use_bmi_calc =st.checkbox("🧮 Calculate Your BMI (Optional)")
 
if use_bmi_calc:
    weight = st.number_input("Weight (kg) ⚖️", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    height = st.number_input("Height (cm) 📏", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
    if height > 0:
        bmi = round(weight / ((height / 100) ** 2), 1)
        st.success(f"Your calculated BMI is: **{bmi} kg/m²**")
    else:
        st.warning("Please enter a valid height.")
else:
    bmi = st.number_input("BMI (kg/m²) 🏋️", min_value=15.0, max_value=50.0, value=29.0, step=0.1)
glucose = st.number_input("Average Glucose Level (mg/dL)", min_value=60.0, max_value=250.0, value=106.0, step=1.0)
hypertension_input = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease_input = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ['WT -Private', 'WT -Self-employed', 'WT -Govt_job', 'WT -children', 'WT -Never_worked'])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes'])

# Encode inputs
le = LabelEncoder()
input_data = {
    'hypertension': 1 if hypertension_input == 'Yes' else 0,
    'heart_disease': 1 if heart_disease_input == 'Yes' else 0,
    'avg_glucose_level': glucose,
    'bmi': bmi,
    'smoking_status': le.fit(df['smoking_status']).transform([smoking_status])[0],
    'gender': le.fit(df['gender']).transform([gender])[0],
    'age': age,
    'ever_married': le.fit(df['ever_married']).transform([ever_married])[0],
    'work_type': le.fit(df['work_type']).transform([work_type])[0],
    'Residence_type': le.fit(df['Residence_type']).transform([residence_type])[0]
}

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict button
if st.button("🔍 Predict Stroke Risk"):
    result = model.predict(input_df)[0]

    # Custom Risk Score Calculation (not model-driven)
    age_risk = 2 if age >= 60 else (1 if age >= 45 else 0)
    heart_risk = 2 if heart_disease_input == 'Yes' else 0
    hyper_risk = 2 if hypertension_input == 'Yes' else 0
    smoke_risk = 2 if smoking_status == "smokes" else (1 if smoking_status == "formerly smoked" else 0)
    glucose_risk = 1 if glucose > 140 else 0
    bmi_risk = 1 if bmi > 30 else 0
    risk_score = age_risk + heart_risk + hyper_risk + smoke_risk + glucose_risk + bmi_risk

    # Risk interpretation
    risk_level = "Low Risk for Stroke"
    if risk_score >= 7:
        risk_level = "High Risk for Stroke"
    elif risk_score >= 4:
        risk_level = "Medium Risk for Stroke"

  
    st.info(f"Risk Score: {risk_score}/10 ({risk_level} Risk)")

    # Risk factor breakdown
    st.subheader("Risk Factors Breakdown:")
    st.write(f"- Age: {'2 (60+)' if age >= 60 else '1 (45-59)' if age >= 45 else '0 (<45)'}")
    st.write(f"- Heart Disease: {'2' if heart_disease_input == 'Yes' else '0'}")
    st.write(f"- Hypertension: {'2' if hypertension_input == 'Yes' else '0'}")
    st.write(f"- Smoking: {'2 (current)' if smoking_status == 'smokes' else '1 (former)' if smoking_status == 'formerly smoked' else '0 (never)'}")
    st.write(f"- Glucose: {'1 (>140)' if glucose > 140 else '0 (≤140)'}")
    st.write(f"- BMI: {'1 (>30)' if bmi > 30 else '0 (≤30)'}")

    st.subheader("Risk Score Interpretation:")
    st.write("0–3: Low risk")
    st.write("4–6: Medium risk")
    st.write("7–10: High risk")
