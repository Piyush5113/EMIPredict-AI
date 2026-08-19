import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Admin Dashboard")

st.markdown("---")

# =====================================================
# Dataset Information
# =====================================================

try:
    df = pd.read_csv(
        "data/processed/feature_engineered_dataset.csv",
        low_memory=False
    )

    total_records = len(df)
    total_features = len(df.columns)

except:
    total_records = "Not Found"
    total_features = "Not Found"

# =====================================================
# Dashboard Metrics
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Dataset Records",
        total_records
    )

with col2:

    st.metric(
        "Total Features",
        total_features
    )

with col3:

    st.metric(
        "Machine Learning Models",
        "2"
    )

st.markdown("---")

# =====================================================
# Files Status
# =====================================================

st.subheader("📂 Project Resources")

files = [

    "models/xgboost_model.pkl",

    "models/xgboost_regressor.pkl",

    "data/processed/feature_engineered_dataset.csv"

]

status = []

for file in files:

    status.append({

        "File": file,

        "Status": "Available" if os.path.exists(file) else "Missing"

    })

status_df = pd.DataFrame(status)

st.dataframe(
    status_df,
    use_container_width=True
)

st.markdown("---")

# =====================================================
# MLflow
# =====================================================

st.subheader("📈 MLflow")

if os.path.exists("mlruns"):

    st.success("MLflow Experiment Tracking Available")

else:

    st.warning("MLflow Folder Not Found")

st.markdown("---")

# =====================================================
# Refresh
# =====================================================

if st.button("🔄 Refresh Dashboard"):

    st.rerun()

st.success("Admin Dashboard Loaded Successfully.")