"""
train_model.py
==============
Standalone training pipeline for the Banana Quality Classifier.

Run this script once before launching the Streamlit app:
    python train_model.py

It will:
  1. Load banana_quality_dataset.csv
  2. Engineer features (same logic as the research notebook)
  3. Train a Gradient Boosting classifier with balanced sample weights
  4. Evaluate on a held-out 20% test set
  5. Save the full sklearn Pipeline to  model/banana_pipeline.joblib
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

warnings.filterwarnings("ignore")

# ── 1. Configuration ──────────────────────────────────────────────────────────

DATA_PATH  = "banana_quality_dataset.csv"   # path to the raw CSV
MODEL_DIR  = "model"                        # folder where the pipeline is saved
MODEL_PATH = os.path.join(MODEL_DIR, "banana_pipeline.joblib")

TARGET_COL = "quality_category"

# Columns to drop before modelling (leakage / identifier / raw date)
DROP_COLS  = ["sample_id", "quality_score", "harvest_date", "ripeness_category"]

# ── 2. Load data ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load the CSV and perform a basic sanity check."""
    df = pd.read_csv(path)
    print(f"[data]  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[data]  Target distribution:\n{df[TARGET_COL].value_counts().to_string()}\n")
    return df


# ── 3. Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering used in the research notebook so that
    the saved pipeline produces identical inputs at inference time.

    New features:
      - harvest_month   : calendar month extracted from harvest_date
      - harvest_quarter : calendar quarter
      - sugar_ripeness_ratio : sugar_content_brix / ripeness_index
      - size_index           : weight_g / length_cm
    """
    df = df.copy()

    # Date features
    df["harvest_date"] = pd.to_datetime(df["harvest_date"])
    df["harvest_month"]   = df["harvest_date"].dt.month
    df["harvest_quarter"] = df["harvest_date"].dt.quarter

    # Interaction features
    df["sugar_ripeness_ratio"] = df["sugar_content_brix"] / (df["ripeness_index"] + 1e-6)
    df["size_index"]           = df["weight_g"] / (df["length_cm"] + 1e-6)

# Drop columns not used in modelling
    df = df.drop(columns=DROP_COLS)

    return df


# ── 4. Build sklearn pipeline ────────────────────────────────────────────────

def build_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Construct a full sklearn Pipeline:
      - Median imputation for numerics
      - Mode imputation + OHE for categoricals
      - GradientBoostingClassifier
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


# ── 5. Train and evaluate ────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame):
    """Full train / evaluate / save cycle."""

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    numeric_features     = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    print(f"[features]  Numeric  ({len(numeric_features)}): {numeric_features}")
    print(f"[features]  Categorical ({len(categorical_features)}): {categorical_features}\n")

    # Train / test split (stratified to preserve class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_features, categorical_features)

    # Balanced sample weights to handle class imbalance
    sw_train = compute_sample_weight("balanced", y_train)

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    cv_f1  = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")

    print(f"[cv]  Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"[cv]  Macro-F1 : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}\n")

    # Final fit on full training set with sample weights
    pipeline.fit(X_train, y_train, model__sample_weight=sw_train)

    # Test set evaluation
    preds = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, preds)
    test_f1  = f1_score(y_test, preds, average="macro")

    print(f"[test]  Accuracy : {test_acc:.4f}")
    print(f"[test]  Macro-F1 : {test_f1:.4f}\n")
    print("[test]  Classification report:")
    print(classification_report(y_test, preds))

    return pipeline, numeric_features, categorical_features


# ── 6. Save artefacts ────────────────────────────────────────────────────────

def save_pipeline(pipeline, numeric_features, categorical_features):
    """Save the fitted pipeline + feature metadata to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    artefact = {
        "pipeline":             pipeline,
        "numeric_features":     numeric_features,
        "categorical_features": categorical_features,
        "target_col":           TARGET_COL,
        "drop_cols":            DROP_COLS,
        "classes":              list(pipeline.classes_),
    }

    joblib.dump(artefact, MODEL_PATH)
    print(f"[save]  Pipeline saved to  {MODEL_PATH}")


# ── 7. Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_raw       = load_data(DATA_PATH)
    df_processed = engineer_features(df_raw)
    pipeline, num_feats, cat_feats = train_and_evaluate(df_processed)
    save_pipeline(pipeline, num_feats, cat_feats)
    print("\n✓  Training complete. You can now launch the Streamlit app.")
