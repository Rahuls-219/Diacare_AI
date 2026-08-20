"""
preprocessing.py - Sklearn preprocessing pipeline for DiaCare AI.

Defines the ColumnTransformer pipeline used for both training and prediction.
The pipeline is fitted ONLY on training data to prevent data leakage.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Feature schema - must match the dataset columns exactly
# ---------------------------------------------------------------------------

NUMERICAL_FEATURES = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
BINARY_FEATURES    = ['hypertension', 'heart_disease']
CATEGORICAL_FEATURES = ['gender', 'smoking_history']
TARGET = 'diabetes'

# Full feature order passed to the model (must be consistent between training
# and prediction - never change the order once the model is trained)
FEATURE_ORDER = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Known categories in the dataset (sorted alphabetically for OrdinalEncoder)
GENDER_CATEGORIES   = ['Female', 'Male', 'Other']
SMOKING_CATEGORIES  = ['No Info', 'current', 'ever', 'former', 'never', 'not current']


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build and return an *unfitted* sklearn ColumnTransformer.

    Numerical pipeline:
        - SimpleImputer(median)  - handles any unexpected NaNs
        - StandardScaler         - zero mean, unit variance

    Binary features (0/1 integers):
        - Passed through as-is (no scaling needed for tree models; scaled
          along with numerics for LR/SVM via the numerical pipeline)

    Categorical pipeline:
        - SimpleImputer(most_frequent)
        - OrdinalEncoder with explicit categories (stable encoding)

    Note: Binary features are passed through separately so that tree-based
    models are not affected by scaling, while LR / SVM use the combined
    preprocessed output. This is a deliberate design choice - all models
    receive the same preprocessor output for fair comparison.
    """
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=[GENDER_CATEGORIES, SMOKING_CATEGORIES],
            handle_unknown='use_encoded_value',
            unknown_value=-1,   # gracefully handles any unseen category
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical',    numerical_pipeline,   NUMERICAL_FEATURES),
            ('binary',       'passthrough',         BINARY_FEATURES),
            ('categorical',  categorical_pipeline,  CATEGORICAL_FEATURES),
        ],
        remainder='drop',   # drop any unexpected extra columns
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Feature / target separation
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y) from the dataset DataFrame.

    Returns (X, y) where X contains only the columns in FEATURE_ORDER.
    """
    X = df[FEATURE_ORDER].copy()
    y = df[TARGET].copy()
    return X, y


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_preprocessing(preprocessor: ColumnTransformer, path: str | Path) -> None:
    """Persist the fitted preprocessing pipeline to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessing pipeline saved - {path}")


def load_preprocessing(path: str | Path) -> ColumnTransformer:
    """Load a fitted preprocessing pipeline from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessing file not found: '{path}'.\n"
            "Run  python train_model.py  to generate it."
        )
    return joblib.load(path)
