import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

def log_to_mlflow(model_name, pipeline, accuracy):

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("Model", model_name)

        model = pipeline.named_steps["model"]

        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)

        if hasattr(model, "max_depth"):
            if model.max_depth is not None:
                mlflow.log_param("max_depth", model.max_depth)

        if hasattr(model, "learning_rate"):
            mlflow.log_param("learning_rate", model.learning_rate)

        mlflow.log_metric("Accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle"
        )

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv(
    "data/processed/feature_engineered_dataset.csv",
    low_memory=False
)
mlflow.set_experiment("EMI Classification")

# ===========================
# Features & Target
# ===========================

X = df.drop(
    columns=[
        "emi_eligibility",
        "max_monthly_emi"
    ]
)

y = df["emi_eligibility"]

# ===========================
# Detect Columns
# ===========================

categorical_columns = X.select_dtypes(include="object").columns
numerical_columns = X.select_dtypes(exclude="object").columns

# ===========================
# Preprocessing
# ===========================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_columns
        ),
        (
            "cat",
            categorical_transformer,
            categorical_columns
        )
    ]
)

# ===========================
# Train Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

os.makedirs("models", exist_ok=True)

# ============================================================
# Logistic Regression
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

logistic_pipeline.fit(X_train, y_train)

logistic_pred = logistic_pipeline.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

print("Accuracy :", logistic_accuracy)

print(
    classification_report(
        y_test,
        logistic_pred
    )
)

joblib.dump(
    logistic_pipeline,
    "models/logistic_model.pkl"
)
log_to_mlflow(
    "Logistic Regression",
    logistic_pipeline,
    logistic_accuracy
)
# ============================================================
# Random Forest
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

rf_pipeline.fit(X_train, y_train)

rf_pred = rf_pipeline.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("Accuracy :", rf_accuracy)

print(
    classification_report(
        y_test,
        rf_pred
    )
)

joblib.dump(
    rf_pipeline,
    "models/random_forest.pkl"
)

log_to_mlflow(
    "Random Forest",
    rf_pipeline,
    rf_accuracy
)
# ============================================================
# XGBoost
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST")
print("=" * 60)

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                objective="multi:softmax",
                num_class=3,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss"
            )
        )
    ]
)

xgb_pipeline.fit(
    X_train,
    y_train_encoded
)

xgb_pred = xgb_pipeline.predict(X_test)

xgb_pred = label_encoder.inverse_transform(xgb_pred)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)

print("Accuracy :", xgb_accuracy)

print(
    classification_report(
        y_test,
        xgb_pred
    )
)

joblib.dump(
    xgb_pipeline,
    "models/xgboost_model.pkl"
)
log_to_mlflow(
    "XGBoost",
    xgb_pipeline,
    xgb_accuracy
)
# ============================================================
# Model Comparison
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

accuracies = {
    "Logistic Regression": logistic_accuracy,
    "Random Forest": rf_accuracy,
    "XGBoost": xgb_accuracy
}

for model, score in accuracies.items():
    print(f"{model:<25} : {score:.4f}")

best_model = max(
    accuracies,
    key=accuracies.get
)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best Model :", best_model)
print("Accuracy   :", round(accuracies[best_model], 4))

print("\nAll Models Saved Successfully.")