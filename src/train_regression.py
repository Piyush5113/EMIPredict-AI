import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv(
    "data/processed/feature_engineered_dataset.csv",
    low_memory=False
)

mlflow.set_experiment("EMI Regression")

# ============================================
# Features & Target
# ============================================

X = df.drop(
    columns=[
        "emi_eligibility",
        "max_monthly_emi"
    ]
)

y = df["max_monthly_emi"]

# ============================================
# Detect Columns
# ============================================

categorical_columns = X.select_dtypes(include="object").columns
numerical_columns = X.select_dtypes(exclude="object").columns

# ============================================
# Preprocessing
# ============================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns),
    ]
)

# ============================================
# Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

os.makedirs("models", exist_ok=True)

# ============================================
# Evaluation Function
# ============================================

def evaluate_model(name, model):

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    filename = name.lower().replace(" ", "_") + ".pkl"

    with mlflow.start_run(run_name=name):

        # Train
        pipeline.fit(X_train, y_train)

        # Prediction
        pred = pipeline.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)

        # Console Output
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R²   : {r2:.4f}")

        # MLflow Parameters
        mlflow.log_param("Model", name)

        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)

        if hasattr(model, "max_depth") and model.max_depth is not None:
            mlflow.log_param("max_depth", model.max_depth)

        if hasattr(model, "learning_rate"):
            mlflow.log_param("learning_rate", model.learning_rate)

        # MLflow Metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Save Local Model
        joblib.dump(
            pipeline,
            f"models/{filename}"
        )

        # Save Model in MLflow
               # Save Model in MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            serialization_format="cloudpickle"
        )

    return r2
  

# ============================================
# Train Models
# ============================================

linear_r2 = evaluate_model(
    "Linear Regression",
    LinearRegression()
)

rf_r2 = evaluate_model(
    "Random Forest Regressor",
    RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
)

xgb_r2 = evaluate_model(
    "XGBoost Regressor",
    XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
)

# ============================================
# Model Comparison
# ============================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

scores = {
    "Linear Regression": linear_r2,
    "Random Forest": rf_r2,
    "XGBoost": xgb_r2
}

for model, score in scores.items():
    print(f"{model:<25} : {score:.4f}")

best_model = max(scores, key=scores.get)

print("\n" + "=" * 60)
print("BEST REGRESSION MODEL")
print("=" * 60)

print("Best Model :", best_model)
print("R² Score   :", round(scores[best_model], 4))

print("\nAll Regression Models Saved Successfully.")