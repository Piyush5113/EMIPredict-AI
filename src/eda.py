import os
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("data/processed/cleaned_dataset.csv", low_memory=False)

# Create output folder
os.makedirs("reports/plots", exist_ok=True)

print("=" * 60)
print("EDA STARTED")
print("=" * 60)

# -----------------------------
# 1. EMI Eligibility Distribution
# -----------------------------
plt.figure(figsize=(7,5))
df["emi_eligibility"].value_counts().plot(kind="bar")
plt.title("EMI Eligibility Distribution")
plt.xlabel("Eligibility")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("reports/plots/emi_eligibility.png")
plt.close()

# -----------------------------
# 2. Salary Distribution
# -----------------------------
plt.figure(figsize=(7,5))
df["monthly_salary"].hist(bins=30)
plt.title("Monthly Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("reports/plots/monthly_salary.png")
plt.close()

# -----------------------------
# 3. Credit Score Distribution
# -----------------------------
plt.figure(figsize=(7,5))
df["credit_score"].hist(bins=30)
plt.title("Credit Score Distribution")
plt.tight_layout()
plt.savefig("reports/plots/credit_score.png")
plt.close()

# -----------------------------
# 4. Requested Amount Distribution
# -----------------------------
plt.figure(figsize=(7,5))
df["requested_amount"].hist(bins=30)
plt.title("Requested Loan Amount")
plt.tight_layout()
plt.savefig("reports/plots/requested_amount.png")
plt.close()

# -----------------------------
# 5. Correlation Heatmap
# -----------------------------
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(14,10))
plt.imshow(numeric_df.corr(), cmap="coolwarm", aspect="auto")
plt.colorbar()

plt.xticks(range(len(numeric_df.columns)), numeric_df.columns, rotation=90)
plt.yticks(range(len(numeric_df.columns)), numeric_df.columns)

plt.tight_layout()
plt.savefig("reports/plots/correlation_heatmap.png")
plt.close()

print("EDA Completed Successfully")