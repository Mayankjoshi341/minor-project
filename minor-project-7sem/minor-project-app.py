import streamlit as st
import pandas as pd
from joblib import load

# Load model and data
model = load('attrition_model.pkl')
data = pd.read_csv('hr_manages.csv')

# --- Page setup ---
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🏢", layout="wide")


# --- Feature groups ---
personal = [
    "age", "gender", "maritalstatus", "education",
    "educationfield", "distancefromhome", "numcompaniesworked"
]
job_related = [
    "businesstravel", "department", "jobrole", "joblevel",
    "monthlyincome", "monthlyrate", "hourlyrate", "dailyrate",
    "overtime", "stockoptionlevel", "percentsalaryhike",
    "performancerating", "yearsatcompany", "yearsincurrentrole",
    "yearssincelastpromotion", "yearswithcurrmanager", "totalworkingyears"
]
satisfaction = [
    "environmentsatisfaction", "jobsatisfaction", "relationshipsatisfaction",
    "worklifebalance", "trainingtimeslastyear", "jobinvolvement"
]

# --- App Header ---
st.title("🏢 Employee Attrition Prediction Dashboard")
st.markdown("### Predict if an employee is likely to stay or leave the organization.")

# --- Sidebar Input ---
st.sidebar.header("🧾 Employee Details Input")
st.sidebar.markdown("Please fill out the following fields:")

perinfo = {}

def collect_inputs(group_title, columns):
    st.sidebar.markdown(f"### {group_title}")
    for col in columns:
        if data[col].dtype == 'object' or data[col].nunique() < 10:
            perinfo[col] = st.sidebar.selectbox(f"{col.replace('_',' ').title()}", data[col].unique().tolist())
        else:
            min_val = int(data[col].min())
            max_val = int(data[col].max())
            mean_val = int(data[col].mean())
            perinfo[col] = st.sidebar.slider(f"{col.replace('_',' ').title()}", min_val, max_val, mean_val)

collect_inputs("👤 Personal Information", personal)
collect_inputs("💼 Job Related", job_related)
collect_inputs("😊 Satisfaction & Engagement", satisfaction)

# --- Display Input Summary ---
st.subheader("🔍 Entered Employee Details")
st.dataframe(pd.DataFrame(perinfo, index=[0]), use_container_width=True)

# --- Prediction Section ---
st.markdown("---")
if st.button("🔮 Predict Attrition", use_container_width=True):
    input_df = pd.DataFrame([perinfo])
    try:
        result = model.predict(input_df)[0]
        if result == 1:
            st.error("⚠️ The employee is **likely to leave** the company.")
        else:
            st.success("✅ The employee is **likely to stay** with the company.")
    except Exception as e:
        st.warning("⚠️ Error during prediction. Please check your input format.")
        st.text(str(e))

# --- Footer ---
st.markdown("---")
st.markdown("<footer>Built by <b>Mayank</b> | Data Science Minor Project 👨‍💻</footer>", unsafe_allow_html=True)
