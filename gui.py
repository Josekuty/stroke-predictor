import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="🧠 Stroke Risk Predictor",
    page_icon="🩺",
    layout="wide",                # better for side-by-side columns
    initial_sidebar_state="collapsed"  # or "auto", or "expanded"
)

# --- Page Background Color ---
st.markdown("""
<style>
body {
    background-color: #34495e;  /* Alice Blue */
}
</style>
""", unsafe_allow_html=True)


# --- Global Styles ---
st.markdown("""
<style>
div.stButton > button:first-child {
    display: block;
    margin: 30px auto;
    background-color: #3498db;
    color: white;
    font-size: 22px;
    padding: 1rem 2rem;
    border-radius: 10px;
    transition: background-color 0.3s ease;
}
div.stButton > button:hover {
    background-color: #2c80b4;
}
</style>
""", unsafe_allow_html=True)

# --- Load model and data ---
model = joblib.load("stroke_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_o.csv").dropna()
    return df[df['smoking_status'] != 'Unknown']

df = load_data()

# --- Header ---
st.markdown("<h1 style='text-align: center; color:#0077b6;'>🧠 Stroke Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("### 📝 Enter the patient information below")

# --- Basic Info ---
st.divider()
st.markdown("#### 👤 Basic Info")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender ⚧️", ["Male", "Female", "Other"])
with col2:
    age = st.number_input("Age 🔢", 1, 100, 25)

# --- BMI Section ---
st.divider()
st.markdown("#### 📏 BMI")
use_bmi_calc = st.checkbox("🧮 Calculate BMI (Optional)")

if use_bmi_calc:
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg) ⚖️", 30.0, 200.0, 70.0, 0.5)
    with col2:
        height = st.number_input("Height (cm) 📏", 100.0, 250.0, 170.0, 0.5)
    if height > 0:
        bmi = round(weight / ((height / 100) ** 2), 1)
        st.success(f"✅ Your BMI is: **{bmi} kg/m²**")
    else:
        st.warning("⚠️ Please enter a valid height.")
else:
    bmi = st.number_input("BMI (kg/m²) 🏋️", 15.0, 50.0, 29.0, 0.1)

# --- Health & Lifestyle Section ---
st.markdown("""
<h2 style='text-align: center; color: #b85c00; font-size: 26px; margin-bottom: 20px;'>
    💉 Health & Lifestyle
</h2>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**🩺 Health Info**")
    glucose = st.number_input("Glucose Level 🩸", 60.0, 250.0, 106.0,1)
    hypertension_input = st.radio("Hypertension 💊", ["No", "Yes"], horizontal=True)
    heart_disease_input = st.radio("Heart Disease 🫀", ["No", "Yes"], horizontal=True)

with col2:
    st.markdown("**🏠 Lifestyle Info**")
    ever_married = st.radio("Ever Married 👨‍👩‍👧‍👦", ["No", "Yes"], horizontal=True)
    residence_type = st.selectbox("Residence Type 🏠", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status 🚬", ['formerly smoked', 'never smoked', 'smokes'])

work_type = st.selectbox("Work Type 💼", df["work_type"].unique())

# --- Encode inputs ---
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
input_df = pd.DataFrame([input_data]).reindex(columns=model_columns, fill_value=0)

# --- Track Predict Button Click ---
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

if st.button("🔍 Predict Stroke Risk"):
    st.session_state.predict_clicked = True

# --- Prediction Result Expander ---
if st.session_state.predict_clicked:
    with st.expander("🧠 Stroke Prediction Result", expanded=True):
        age_risk = 2 if age >= 60 else (1 if age >= 45 else 0)
        heart_risk = 2 if heart_disease_input == 'Yes' else 0
        hyper_risk = 2 if hypertension_input == 'Yes' else 0
        smoke_risk = 2 if smoking_status == "smokes" else (1 if smoking_status == "formerly smoked" else 0)
        glucose_risk = 1 if glucose > 140 else 0
        bmi_risk = 1 if bmi > 30 else 0
        risk_score = age_risk + heart_risk + hyper_risk + smoke_risk + glucose_risk + bmi_risk

        st.markdown("---")
        st.markdown(f"<h3 style='text-align:center;'>🧪 Risk Score: {risk_score}/10</h3>", unsafe_allow_html=True)

        if risk_score >= 7:
            st.error("⚠️ High Risk of Stroke!\n\nPlease consult a medical professional urgently.")
        elif risk_score >= 4:
            st.warning("🟠 Medium Risk of Stroke\n\nKeep monitoring health indicators.")
        else:
            st.success("🟢 Low Risk of Stroke\n\nStay healthy and active!")

    # --- Explanation shown on right side if checkbox selected ---
    if st.checkbox("📊 Show Explanation"):
            st.markdown("### 🔍 Risk Factor Breakdown")
            st.write(f"• Age: {'2 (60+)' if age >= 60 else '1 (45–59)' if age >= 45 else '0 (<45)'}")
            st.write(f"• Heart Disease: {'2' if heart_disease_input == 'Yes' else '0'}")
            st.write(f"• Hypertension: {'2' if hypertension_input == 'Yes' else '0'}")
            st.write(f"• Smoking: {'2 (current)' if smoking_status == 'smokes' else '1 (former)' if smoking_status == 'formerly smoked' else '0 (never)'}")
            st.write(f"• Glucose: {'1 (>140)' if glucose > 140 else '0 (≤140)'}")
            st.write(f"• BMI: {'1 (>30)' if bmi > 30 else '0 (≤30)'}")

            st.markdown("---")
            st.markdown("### 📘 Risk Score Guide")
            st.info("**0–3**: Low risk\n\n**4–6**: Medium risk\n\n**7–10**: High risk")
