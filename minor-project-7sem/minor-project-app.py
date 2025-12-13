import streamlit as st
import pandas as pd
from joblib import load
import os

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🏢",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load(os.path.join(BASE_DIR, "attrition_model.pkl"))
data = pd.read_csv(os.path.join(BASE_DIR, "hr_manages.csv"))

EXPECTED_COLUMNS = model.named_steps["preprocessor"].feature_names_in_

st.title("🏢 Employee Attrition Prediction System")
st.markdown(
    "Predict whether an employee is **likely to leave** the organization using a trained machine learning model."
)

st.divider()

st.sidebar.header("🧾 Employee Information")

def input_block(title):
    st.sidebar.subheader(title)

def collect_inputs(columns):
    values = {}
    for col in columns:
        if data[col].dtype == "object" or data[col].nunique() < 10:
            values[col] = st.sidebar.selectbox(
                col.replace("_", " ").title(),
                sorted(data[col].unique().tolist())
            )
        else:
            values[col] = st.sidebar.slider(
                col.replace("_", " ").title(),
                int(data[col].min()),
                int(data[col].max()),
                int(data[col].mean())
            )
    return values

input_block("👤 Personal Details")
personal = collect_inputs([
    "age", "gender", "maritalstatus", "education",
    "educationfield", "distancefromhome", "numcompaniesworked"
])

input_block("💼 Job Details")
job = collect_inputs([
    "businesstravel", "department", "jobrole", "joblevel",
    "monthlyincome", "monthlyrate", "hourlyrate", "dailyrate",
    "overtime", "stockoptionlevel", "percentsalaryhike",
    "performancerating", "yearsatcompany", "yearsincurrentrole",
    "yearssincelastpromotion", "yearswithcurrmanager",
    "totalworkingyears"
])

input_block("😊 Satisfaction & Engagement")
satisfaction = collect_inputs([
    "environmentsatisfaction", "jobsatisfaction",
    "relationshipsatisfaction", "worklifebalance",
    "trainingtimeslastyear", "jobinvolvement"
])

employee_data = {**personal, **job, **satisfaction}

st.subheader("📋 Employee Profile Summary")
input_df = pd.DataFrame([employee_data])
st.dataframe(input_df, use_container_width=True)

st.divider()

if st.button("🔮 Predict Attrition Risk", use_container_width=True):
    try:
        input_df = input_df[EXPECTED_COLUMNS]
        prob = model.predict_proba(input_df)[0][1]
        prediction = 1 if prob > 0.30 else 0

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Attrition Risk Probability",
                value=f"{prob*100:.2f}%"
            )

        with col2:
            if prediction == 1:
                st.error("⚠️ Employee is **Likely to Leave**")
            else:
                st.success("✅ Employee is **Likely to Stay**")

        st.caption(
            "Threshold set at 30% to prioritize recall and reduce missed attrition cases."
        )

    except Exception as e:
        st.error("Prediction failed due to input mismatch.")
        st.exception(e)

st.divider()
st.markdown(
    "<center>Built by <b>Mayank | Mahima | Sachin</b> · Data Science Minor Project</center>",
    unsafe_allow_html=True
)
