import pandas as pd

# Dataset Path
DATA_PATH = "data/raw/emi_prediction_dataset.csv"
OUTPUT_PATH = "data/processed/cleaned_dataset.csv"

# Load Dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"\nShape: {df.shape}")

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove Duplicate Rows
df.drop_duplicates(inplace=True)

# Remove Extra Spaces from Column Names
df.columns = df.columns.str.strip()

# Fill Missing Values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Save Cleaned Dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaning Completed Successfully.")
print(f"Cleaned Dataset Saved At: {OUTPUT_PATH}")
print(f"Final Shape: {df.shape}")