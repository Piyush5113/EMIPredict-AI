import streamlit as st

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💳",
    layout="wide"
)

st.title("💳 EMIPredict AI")
st.subheader("Machine Learning Based Financial Risk Assessment System")

st.markdown("---")

st.markdown("""
## 📌 Project Overview

EMIPredict AI is a Machine Learning based financial risk assessment platform
that predicts loan eligibility and affordable monthly EMI using customer
financial information.

### Modules

- ✅ EMI Eligibility Prediction (Classification)
- ✅ Maximum EMI Prediction (Regression)
- ✅ Exploratory Data Analysis (EDA)
- ✅ Model Performance Comparison
- ✅ MLflow Experiment Tracking
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Dataset Size",
        "404,800 Records"
    )

    st.metric(
        "Classification Models",
        "3"
    )

    st.metric(
        "Regression Models",
        "3"
    )

with col2:

    st.metric(
        "Best Classification Model",
        "XGBoost"
    )

    st.metric(
        "Classification Accuracy",
        "97.78%"
    )

    st.metric(
        "Best Regression R²",
        "0.9920"
    )

st.markdown("---")

st.header("🚀 Features")

st.write("✅ EMI Eligibility Prediction")
st.write("✅ Maximum Safe EMI Prediction")
st.write("✅ Financial Risk Assessment")
st.write("✅ Interactive Dashboard")
st.write("✅ Exploratory Data Analysis")
st.write("✅ Model Comparison")
st.write("✅ MLflow Experiment Tracking")
st.write("✅ Machine Learning Powered Decision Support")

st.markdown("---")

st.header("📊 Models Used")

st.write("### Classification")
st.write("- Logistic Regression")
st.write("- Random Forest Classifier")
st.write("- XGBoost Classifier")

st.write("### Regression")
st.write("- Linear Regression")
st.write("- Random Forest Regressor")
st.write("- XGBoost Regressor")

st.markdown("---")

st.success(
    "🏆 Best Models Selected Automatically using Performance Metrics."
)

st.info(
    "👈 Use the left sidebar to access EMI Eligibility, Maximum EMI Prediction, EDA and Model Performance pages."
)