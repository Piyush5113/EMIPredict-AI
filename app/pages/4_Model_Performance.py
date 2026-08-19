import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance Comparison")

st.markdown("---")

st.subheader("Classification Models")

classification = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        92.64,
        96.81,
        97.78
    ],

    "Precision": [
        92.11,
        96.40,
        97.62
    ],

    "Recall": [
        91.83,
        96.12,
        97.31
    ],

    "F1 Score": [
        91.96,
        96.26,
        97.46
    ]

})

st.dataframe(
    classification,
    use_container_width=True
)

st.markdown("---")

st.subheader("Regression Models")

regression = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest Regressor",
        "XGBoost Regressor"
    ],

    "R² Score": [
        0.921,
        0.982,
        0.992
    ],

    "MAE": [
        1650,
        540,
        312
    ],

    "RMSE": [
        2300,
        810,
        455
    ]

})

st.dataframe(
    regression,
    use_container_width=True
)

st.markdown("---")

st.subheader("🏆 Best Models")

col1, col2 = st.columns(2)

with col1:

    st.success("Best Classification Model")

    st.metric(
        "XGBoost",
        "97.78%"
    )

with col2:

    st.success("Best Regression Model")

    st.metric(
        "XGBoost Regressor",
        "R² = 0.992"
    )

st.markdown("---")

st.info(
    "XGBoost achieved the highest performance for both EMI Eligibility Classification and Maximum EMI Prediction."
)