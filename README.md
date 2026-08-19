# 💳 EMIPredict AI

An intelligent Machine Learning based Financial Risk Assessment System developed using Python, XGBoost, Scikit-Learn and Streamlit.

The application predicts:

- ✅ EMI Eligibility
- ✅ Maximum Safe Monthly EMI

using customer financial information.

---

# 📌 Project Overview

EMIPredict AI is a machine learning project that helps financial institutions evaluate whether a customer is eligible for a loan EMI and estimates the maximum EMI amount that can be safely paid every month.

The system performs complete data preprocessing, feature engineering, model training, model evaluation and deployment through an interactive Streamlit dashboard.

---

# 🚀 Features

- EMI Eligibility Prediction
- Maximum EMI Prediction
- Exploratory Data Analysis (EDA)
- Model Performance Comparison
- Interactive Streamlit Dashboard
- Admin Dashboard
- Feature Engineering
- MLflow Experiment Tracking
- Classification & Regression Models

---

# 🛠️ Technologies Used

## Programming

- Python 3.x

## Machine Learning

- Scikit-Learn
- XGBoost

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn
- Plotly

## Deployment

- Streamlit

## Experiment Tracking

- MLflow

---

# 📂 Project Structure

```
EMIPredict-AI
│
├── app
│   ├── Home.py
│   └── pages
│       ├── 1_EDA.py
│       ├── 2_EMI_Eligibility.py
│       ├── 3_Max_EMI_Prediction.py
│       ├── 4_Model_Performance.py
│       └── 5_Admin.py
│
├── data
│
├── models
│   ├── xgboost_model.pkl
│   ├── xgboost_regressor.pkl
│   └── label_encoder.pkl
│
├── notebooks
├── reports
├── src
├── mlruns
├── requirements.txt
├── README.md
└── main.py
```

---

# 📊 Dataset

Dataset Size:

**404,800 Customer Records**

Dataset contains customer financial information such as

- Age
- Gender
- Marital Status
- Education
- Employment Type
- Company Type
- Monthly Salary
- Existing Loans
- Credit Score
- Emergency Fund
- Bank Balance
- Requested Loan Amount
- Loan Tenure
- Monthly Expenses
- Family Details

Feature Engineering was performed to create additional financial indicators including:

- Debt to Income Ratio
- Expense to Income Ratio
- Savings Ratio
- Emergency Fund Ratio
- EMI Affordability
- Employment Stability
- Financial Dependents
- Total Monthly Expenses

---

# 🤖 Machine Learning Models

## Classification Models

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Best Model:

**XGBoost Classifier**

Accuracy:

**97.78%**

---

## Regression Models

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Best Model:

**XGBoost Regressor**

R² Score:

**0.992**

---

# 📈 Streamlit Pages

## Home

Project overview and statistics.

---

## EDA

- Dataset Preview
- Missing Values
- Correlation Heatmap
- Target Distribution
- Salary Distribution
- EMI Distribution

---

## EMI Eligibility

Predicts whether a customer is:

- Eligible
- High Risk
- Not Eligible

---

## Maximum EMI Prediction

Predicts:

- Maximum Monthly EMI
- Annual EMI Capacity
- Estimated Loan Capacity

---

## Model Performance

Comparison of all trained Machine Learning models using:

Classification

- Accuracy

Regression

- R² Score
- MAE
- RMSE

---

## Admin Dashboard

Displays

- Dataset Records
- Number of Features
- ML Models
- Project Resources

---

# 📷 Screenshots

Add screenshots of:

- Home Page
- EDA
- EMI Eligibility
- Maximum EMI Prediction
- Model Performance
- Admin Dashboard

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/EMIPredict-AI.git
```

Move into the project directory

```bash
cd EMIPredict-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/Home.py
```

---

# 📌 Future Improvements

- User Authentication
- Loan Recommendation System
- PDF Report Generation
- Cloud Deployment
- Database Integration
- Live Banking API Integration

---

