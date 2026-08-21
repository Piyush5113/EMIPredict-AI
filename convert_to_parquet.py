import pandas as pd

df = pd.read_csv(
    "data/processed/feature_engineered_dataset.csv",
    low_memory=False
)

# Convert numeric columns
numeric_columns = [
    "age",
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
    "max_monthly_emi",
    "total_monthly_expenses",
    "debt_to_income_ratio",
    "expense_to_income_ratio",
    "savings_ratio",
    "emergency_fund_ratio",
    "emi_affordability",
    "employment_stability",
    "financial_dependents"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid rows
df = df.dropna()

df.to_parquet(
    "data/processed/feature_engineered_dataset.parquet",
    index=False
)

print("✅ Parquet created successfully!")