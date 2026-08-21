import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis")

st.markdown("---")

# ==========================================================
# Load Dataset
# ==========================================================

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    return pd.read_parquet(
        "data/processed/feature_engineered_dataset.parquet"
    )

df = load_data()

sample_df = df.sample(
    min(50000, len(df)),
    random_state=42
)
# ==========================================================
# Dataset Overview
# ==========================================================

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Total Features", len(df.columns))

with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

st.markdown("---")

# ==========================================================
# Preview
# ==========================================================

st.header("Dataset Preview")

ROWS_PER_PAGE = 25

total_rows = len(df)
total_pages = (total_rows - 1) // ROWS_PER_PAGE + 1

page = st.number_input(
    "Page",
    min_value=1,
    max_value=total_pages,
    value=1,
    step=1
)

start = (page - 1) * ROWS_PER_PAGE
end = start + ROWS_PER_PAGE

st.write(f"Showing rows {start + 1} - {min(end, total_rows)} of {total_rows}")

st.dataframe(
    df.iloc[start:end],
    use_container_width=True,
    height=500
)

st.markdown("---")

# ==========================================================
# Data Types
# ==========================================================

st.header("Feature Data Types")

dtype_df = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(
    dtype_df,
    use_container_width=True
)

st.markdown("---")

# ==========================================================
# Missing Values
# ==========================================================

st.header("Missing Values")

missing_df = pd.DataFrame({
    "Feature": df.columns,
    "Missing Values": df.isnull().sum()
})

st.dataframe(
    missing_df,
    use_container_width=True
)

st.markdown("---")

# ==========================================================
# Statistics
# ==========================================================

st.header("Statistical Summary")

st.dataframe(
    df.describe(include="number"),
    use_container_width=True
)

st.markdown("---")

# ==========================================================
# Salary Distribution
# ==========================================================

if "monthly_salary" in df.columns:

    st.header("Monthly Salary Distribution")


    fig = px.histogram(
        sample_df,
        x="monthly_salary",
        nbins=40,
        title="Monthly Salary"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
        )

# ==========================================================
# Credit Score
# ==========================================================

if "credit_score" in df.columns:

    st.header("Credit Score Distribution")

    fig = px.histogram(
        sample_df,
        x="credit_score",
        nbins=30,
        title="Credit Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# Maximum EMI
# ==========================================================

if "max_monthly_emi" in df.columns:

    st.header("Maximum EMI Distribution")

    fig = px.histogram(
        sample_df,
        x="max_monthly_emi",
        nbins=40,
        title="Maximum Monthly EMI"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# Eligibility Distribution
# ==========================================================

# ==========================================================
# EMI Eligibility Distribution
# ==========================================================

st.header("EMI Eligibility Distribution")

eligibility_count = (
    sample_df["emi_eligibility"]
    .value_counts()
    .reset_index()
)

eligibility_count.columns = [
    "EMI Eligibility",
    "Count"
]

fig = px.bar(
    eligibility_count,
    x="EMI Eligibility",
    y="Count",
    color="EMI Eligibility",
    title="EMI Eligibility Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)