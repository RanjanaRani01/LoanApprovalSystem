
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.write("Predict whether a loan will be approved or not using Machine Learning.")

# -------------------------------
# LOAD DATASET
# -------------------------------
data = pd.read_csv("loan_approval.csv")

# -------------------------------
# DATA PREPROCESSING
# -------------------------------

# Drop column
data = data.drop('experience', axis=1)

# Mapping categorical values
data['home_ownership'] = data['home_ownership'].map({
    'RENT': 0,
    'OWN': 1,
    'MORTGAGE': 2,
    'OTHER': 3
})

data['loan_intent'] = data['loan_intent'].map({
    'PERSONAL': 0,
    'EDUCATION': 1,
    'MEDICAL': 2,
    'VENTURE': 3,
    'HOMEIMPROVEMENT': 4,
    'DEBTCONSOLIDATION': 5
})

data['gender'] = data['gender'].map({
    'male': 0,
    'female': 1
})

data['previous_loan_defaults'] = data['previous_loan_defaults'].map({
    'No': 0,
    'Yes': 1
})

# Encoding education column
education_order = [['High School', 'Bachelor', 'Master', 'Associate', 'Doctorate']]

encoder = OrdinalEncoder(categories=education_order)

data['education'] = encoder.fit_transform(data[['education']])

# -------------------------------
# FEATURES & TARGET
# -------------------------------
x = data[['age',
          'gender',
          'income',
          'home_ownership',
          'loan_amnt',
          'loan_intent',
          'loan_int_rate',
          'credit_history(years)',
          'credit_score',
          'previous_loan_defaults']]

y = data['loan_status']

# -------------------------------
# SPLIT DATA
# -------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = RandomForestClassifier()

model.fit(x_train, y_train)

# -------------------------------
# MODEL ACCURACY
# -------------------------------
y_predict = model.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)

st.success(f"Model Accuracy: {accuracy*100:.2f}%")

# -------------------------------
# USER INPUT SECTION
# -------------------------------

st.header("Enter Applicant Details")

age = st.number_input("Enter Age", min_value=18, max_value=100)

gender = st.selectbox(
    "Select Gender",
    ("Male", "Female")
)

income = st.number_input("Enter Income")

loan_amnt = st.number_input("Enter Loan Amount")

home_ownership = st.selectbox(
    "Home Ownership",
    ("RENT", "OWN", "MORTGAGE", "OTHER")
)

loan_intent = st.selectbox(
    "Loan Intent",
    (
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    )
)

loan_int_rate = st.number_input("Enter Interest Rate")

credit_history_years = st.number_input("Credit History (Years)")

credit_score = st.number_input("Enter Credit Score")

previous_loan_defaults = st.selectbox(
    "Previous Loan Defaults",
    ("No", "Yes")
)

# -------------------------------
# CONVERT INPUTS
# -------------------------------

gender = 0 if gender == "Male" else 1

home_map = {
    "RENT": 0,
    "OWN": 1,
    "MORTGAGE": 2,
    "OTHER": 3
}

loan_map = {
    "PERSONAL": 0,
    "EDUCATION": 1,
    "MEDICAL": 2,
    "VENTURE": 3,
    "HOMEIMPROVEMENT": 4,
    "DEBTCONSOLIDATION": 5
}

default_map = {
    "No": 0,
    "Yes": 1
}

home_ownership = home_map[home_ownership]
loan_intent = loan_map[loan_intent]
previous_loan_defaults = default_map[previous_loan_defaults]

# -------------------------------
# PREDICTION
# -------------------------------

if st.button("Predict Loan Status"):

    prediction = model.predict([[
        age,
        gender,
        income,
        home_ownership,
        loan_amnt,
        loan_intent,
        loan_int_rate,
        credit_history_years,
        credit_score,
        previous_loan_defaults
    ]])

    if prediction[0] == 1:
        st.success("✅ LOAN APPROVED")
    else:
        st.error("❌ LOAN NOT APPROVED")
