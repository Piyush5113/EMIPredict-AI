import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="EMI Eligibility",
    page_icon="💳",
    layout="wide"
)

st.title("💳 EMI Eligibility Prediction")

st.markdown("---")

# ==========================================================
# Load Model
# ==========================================================

model = joblib.load("models/xgboost_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# ==========================================================
# Personal Details
# ==========================================================

st.header("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        18,
        100,
        30
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married"
        ]
    )

    education = st.selectbox(
    "Education",
    [
        "High School",
        "Graduate",
        "Professional",
        "Post Graduate"
    ]
)

with col2:

    employment_type = st.selectbox(
    "Employment Type",
    [
        "Private",
        "Government",
        "Self-employed"
    ]
)

    years_of_employment = st.number_input(
        "Years of Employment",
        0.0,
        50.0,
        5.0
    )

    company_type = st.selectbox(
        "Company Type",
        [
            "Small",
            "Mid-size",
            "Large Indian",
            "MNC",
            "Startup"
        ]
    )

    house_type = st.selectbox(
        "House Type",
        [
            "Own",
            "Rented",
            "Family"
        ]
    )

st.markdown("---")

# ==========================================================
# Financial Details
# ==========================================================

st.header("💰 Financial Information")

col1, col2 = st.columns(2)

with col1:

    monthly_salary = st.number_input(
        "Monthly Salary",
        0.0,
        value=50000.0
    )

    monthly_rent = st.number_input(
        "Monthly Rent",
        0.0,
        value=10000.0
    )

    current_emi_amount = st.number_input(
        "Current EMI",
        0.0,
        value=5000.0
    )

    requested_amount = st.number_input(
        "Requested Loan Amount",
        0.0,
        value=500000.0
    )

    requested_tenure = st.number_input(
        "Loan Tenure (Months)",
        1,
        360,
        60
    )

with col2:

    credit_score = st.number_input(
        "Credit Score",
        300.0,
        900.0,
        750.0
    )

    bank_balance = st.number_input(
        "Bank Balance",
        0.0,
        value=150000.0
    )

    emergency_fund = st.number_input(
        "Emergency Fund",
        0.0,
        value=100000.0
    )

st.markdown("---")

# ==========================================================
# Family Expenses
# ==========================================================

st.header("👨‍👩‍👧 Family Details")

col1, col2 = st.columns(2)

with col1:

    family_size = st.number_input(
        "Family Size",
        1,
        20,
        4
    )

    dependents = st.number_input(
        "Dependents",
        0,
        20,
        2
    )

    school_fees = st.number_input(
        "School Fees",
        0.0,
        value=5000.0
    )

    college_fees = st.number_input(
        "College Fees",
        0.0,
        value=0.0
    )

with col2:

    travel_expenses = st.number_input(
        "Travel Expenses",
        0.0,
        value=3000.0
    )

    groceries_utilities = st.number_input(
        "Groceries & Utilities",
        0.0,
        value=10000.0
    )

    other_monthly_expenses = st.number_input(
        "Other Monthly Expenses",
        0.0,
        value=5000.0
    )

    existing_loans = st.selectbox(
        "Existing Loans",
        [
            "Yes",
            "No"
        ]
    )
    emi_scenario = st.selectbox(
    "Loan Purpose",
    [
        "Personal Loan EMI",
        "Vehicle EMI",
        "Education EMI",
        "Home Appliances EMI",
        "E-commerce Shopping EMI"
    ]
)

st.markdown("---")

# ==========================================================
# Feature Engineering
# ==========================================================

total_monthly_expenses = (
    monthly_rent
    + school_fees
    + college_fees
    + travel_expenses
    + groceries_utilities
    + other_monthly_expenses
    + current_emi_amount
)

debt_to_income_ratio = (
    current_emi_amount / monthly_salary
    if monthly_salary > 0 else 0
)

expense_to_income_ratio = (
    total_monthly_expenses / monthly_salary
    if monthly_salary > 0 else 0
)

savings_ratio = (
    bank_balance / monthly_salary
    if monthly_salary > 0 else 0
)

emergency_fund_ratio = (
    emergency_fund / monthly_salary
    if monthly_salary > 0 else 0
)

emi_affordability = (
    requested_amount /
    (monthly_salary * requested_tenure)
    if monthly_salary > 0 else 0
)

employment_stability = (
    1 if years_of_employment >= 5 else 0
)

financial_dependents = (
    family_size + dependents
)

st.markdown("---")

predict_btn = st.button(
    "Predict EMI Eligibility",
    use_container_width=True
)

if predict_btn:

    input_df = pd.DataFrame({

        "age": [age],
        "gender": [gender],
        "marital_status": [marital_status],
        "education": [education],
        "monthly_salary": [monthly_salary],
        "employment_type": [employment_type],
        "years_of_employment": [years_of_employment],
        "company_type": [company_type],
        "house_type": [house_type],
        "monthly_rent": [monthly_rent],
        "family_size": [family_size],
        "dependents": [dependents],
        "school_fees": [school_fees],
        "college_fees": [college_fees],
        "travel_expenses": [travel_expenses],
        "groceries_utilities": [groceries_utilities],
        "other_monthly_expenses": [other_monthly_expenses],
        "existing_loans": [existing_loans],
        "current_emi_amount": [current_emi_amount],
        "credit_score": [credit_score],
        "bank_balance": [bank_balance],
        "emergency_fund": [emergency_fund],
        "emi_scenario": [emi_scenario],
        "requested_amount": [requested_amount],
        "requested_tenure": [requested_tenure],
        "total_monthly_expenses": [total_monthly_expenses],
        "debt_to_income_ratio": [debt_to_income_ratio],
        "expense_to_income_ratio": [expense_to_income_ratio],
        "savings_ratio": [savings_ratio],
        "emergency_fund_ratio": [emergency_fund_ratio],
        "emi_affordability": [emi_affordability],
        "employment_stability": [employment_stability],
        "financial_dependents": [financial_dependents]

    })

    prediction = model.predict(input_df)[0]

    prediction_label = encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = (
        model.predict_proba(input_df).max()
        * 100
    )

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction_label == "Eligible":

        st.success("✅ Eligible")

    elif prediction_label == "High_Risk":

        st.warning("⚠️ High Risk")

    else:

        st.error("❌ Not Eligible")

    st.metric(
        "Prediction",
        prediction_label.replace("_", " ")
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.write("### Input Summary")

    st.dataframe(input_df)