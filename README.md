# 🧠 Stroke Prediction App

This is a **machine learning web app** that predicts the **risk of stroke** based on patient details such as age, hypertension, heart disease, BMI, glucose level, etc.

It is built using:
- 🔍 Random Forest Classifier (scikit-learn)
- 🎛️ Streamlit (for the web app interface)
- 🌐 Deployed publicly on [Streamlit Cloud]([https://stroke-predictor-project.streamlit.app/])

---

## 📌 Features
- Predict stroke risk using ML model
- Manual scoring based on health risk factors
- Clean user-friendly interface (Yes/No dropdowns instead of 0/1)
- Handles missing/unknown values

---

## 🚀 Try it Live

👉 [Click here to open the web app]((https://stroke-predictor-project.streamlit.app/))

---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/Josekuty/stroke-predictor.git
cd stroke-prediction-app
pip install -r requirements.txt
streamlit run app.py
