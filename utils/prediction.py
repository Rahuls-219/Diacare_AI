"""
prediction.py - Inference utilities for DiaCare AI.

Loads the saved model and preprocessing pipeline, validates input,
and returns the risk probability with a categorised assessment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import FEATURE_ORDER

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

MODEL_PATH         = Path(__file__).parent.parent / "model" / "diabetes_model.pkl"
PREPROCESSING_PATH = Path(__file__).parent.parent / "model" / "preprocessing.pkl"

# ---------------------------------------------------------------------------
# Risk thresholds (documented, consistent throughout the entire application)
# ---------------------------------------------------------------------------

LOW_RISK_THRESHOLD      = 0.30   # probability < 0.30  - LOW RISK
MODERATE_RISK_THRESHOLD = 0.60   # 0.30 - prob < 0.60  - MODERATE RISK
                                  # prob - 0.60          - HIGHER RISK

RISK_COLORS = {
    "LOW RISK":      "#10B981",   # green
    "MODERATE RISK": "#F59E0B",   # amber
    "HIGHER RISK":   "#EF4444",   # red
}

RISK_ICONS = {
    "LOW RISK":      "-",
    "MODERATE RISK": "--",
    "HIGHER RISK":   "-",
}


# ---------------------------------------------------------------------------
# Model / preprocessor loading
# ---------------------------------------------------------------------------

def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load and return the trained sklearn model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'.\n"
            "Run  python train_model.py  to train and save the model."
        )
    return joblib.load(model_path)


def load_preprocessing_pipeline(preprocessing_path: Path = PREPROCESSING_PATH) -> Any:
    """Load and return the fitted preprocessing pipeline."""
    preprocessing_path = Path(preprocessing_path)
    if not preprocessing_path.exists():
        raise FileNotFoundError(
            f"Preprocessing file not found: '{preprocessing_path}'.\n"
            "Run  python train_model.py  to generate it."
        )
    return joblib.load(preprocessing_path)


# ---------------------------------------------------------------------------
# Risk categorisation
# ---------------------------------------------------------------------------

def get_risk_category(probability: float) -> tuple[str, str]:
    """
    Map a raw probability to a (risk_label, description) pair.

    Thresholds:
        < 0.30  - LOW RISK
        0.30-0.60 - MODERATE RISK
        - 0.60  - HIGHER RISK
    """
    if probability < LOW_RISK_THRESHOLD:
        label = "LOW RISK"
        desc = (
            "The model's estimated probability is relatively low based on the "
            "provided information. This does not rule out diabetes."
        )
    elif probability < MODERATE_RISK_THRESHOLD:
        label = "MODERATE RISK"
        desc = (
            "The model's estimated probability is in the moderate range. "
            "Consider discussing these results with a qualified healthcare professional."
        )
    else:
        label = "HIGHER RISK"
        desc = (
            "The model's estimated probability is relatively high based on the "
            "provided health information. Consultation with a healthcare professional is advisable."
        )
    return label, desc


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_risk(input_data: dict[str, Any], model: Any, preprocessor: Any) -> dict:
    """
    Predict diabetes risk for a single patient.

    Parameters
    ----------
    input_data   : dict with keys matching FEATURE_ORDER
    model        : fitted sklearn estimator (loaded via load_model)
    preprocessor : fitted ColumnTransformer (loaded via load_preprocessing_pipeline)

    Returns
    -------
    dict with:
        probability     : float  (0.0 - 1.0)
        probability_pct : float  (0.0 - 100.0, rounded to 1 dp)
        risk_category   : str    ("LOW RISK" / "MODERATE RISK" / "HIGHER RISK")
        description     : str
        risk_color      : str    (hex color for UI)
        risk_icon       : str    (emoji)
    """
    # Build DataFrame in the exact column order used during training
    df_input = pd.DataFrame([input_data])[FEATURE_ORDER]

    # Apply fitted preprocessing (same transformations as training)
    X_processed = preprocessor.transform(df_input)

    # Predict probability of positive class (diabetes = 1)
    probability = float(model.predict_proba(X_processed)[0][1])

    risk_category, description = get_risk_category(probability)

    return {
        'probability':     probability,
        'probability_pct': round(probability * 100, 1),
        'risk_category':   risk_category,
        'description':     description,
        'risk_color':      RISK_COLORS[risk_category],
        'risk_icon':       RISK_ICONS[risk_category],
    }


def get_feature_influence(model: Any, feature_names: list[str]) -> dict[str, float] | None:
    """
    Extract feature influence scores from the model where available.

    Returns a dict {feature_name: score} or None if not supported.
    """
    model_type = type(model).__name__

    if hasattr(model, 'feature_importances_'):
        # Random Forest, Gradient Boosting etc.
        return dict(zip(feature_names, model.feature_importances_))

    elif hasattr(model, 'coef_'):
        # Logistic Regression
        coefs = np.abs(model.coef_[0])
        return dict(zip(feature_names, coefs))

    return None
