import pandas as pd
import numpy as np


# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_dataset.csv", low_memory=False)

numeric_columns = [
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure",
    "max_monthly_emi"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df[numeric_columns] = df[numeric_columns].fillna(0)

# ----------------------------------
# Financial Ratios
# ----------------------------------

# Total Monthly Expenses
df["total_monthly_expenses"] = (
    df["monthly_rent"]
    + df["school_fees"]
    + df["college_fees"]
    + df["travel_expenses"]
    + df["groceries_utilities"]
    + df["other_monthly_expenses"]
    + df["current_emi_amount"]
)

# Debt To Income Ratio
df["debt_to_income_ratio"] = (
    df["current_emi_amount"] / df["monthly_salary"]
)

# Expense To Income Ratio
df["expense_to_income_ratio"] = (
    df["total_monthly_expenses"] / df["monthly_salary"]
)

# Savings Ratio
df["savings_ratio"] = (
    df["bank_balance"] / df["monthly_salary"]
)

# Emergency Fund Ratio
df["emergency_fund_ratio"] = (
    df["emergency_fund"] / df["monthly_salary"]
)

# EMI Affordability
df["emi_affordability"] = (
    df["requested_amount"] / (df["monthly_salary"] * df["requested_tenure"])
)

# ----------------------------------
# Employment Stability
# ----------------------------------

df["employment_stability"] = np.where(
    df["years_of_employment"] >= 5,
    1,
    0
)

# ----------------------------------
# Family Financial Load
# ----------------------------------

df["financial_dependents"] = (
    df["dependents"] + df["family_size"]
)


# Replace Infinity with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN in numeric columns
numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)


# Save Dataset
df.to_csv(
    "data/processed/feature_engineered_dataset.csv",
    index=False
)

print("Feature Engineering Completed Successfully.")
print(df.head())
print(df.dtypes)

print(df.isin([np.inf, -np.inf]).sum())