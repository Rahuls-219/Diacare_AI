"""
data_loader.py - Dataset loading and exploratory data analysis utilities for DiaCare AI.

Dataset: Diabetes Prediction Dataset
Source:  Kaggle - https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
License: CC0 Public Domain
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "data" / "diabetes.csv"

# Publicly known statistics of the CC0 Kaggle Diabetes Prediction Dataset
# used to generate a representative synthetic dataset when the real CSV is
# not available (e.g., during offline demo).  Values sourced from the
# Kaggle dataset card and public EDA notebooks.
_SYNTHETIC_N = 96000          # approximate post-dedup size of the real dataset
_DIABETIC_FRAC = 0.085        # ~8.5 % positive class
_GENDER_PROBS = [0.585, 0.413, 0.002]   # Male / Female / Other
_SMOKING_CATS = ['never', 'No Info', 'current', 'former', 'ever', 'not current']
_SMOKING_PROBS = [0.351, 0.358, 0.093, 0.094, 0.040, 0.064]

DOWNLOAD_URLS = [
    # Public GitHub mirrors of the CC0 dataset (tried in order)
    "https://raw.githubusercontent.com/anushkumar27/diabetes-prediction/main/diabetes_prediction_dataset.csv",
    "https://raw.githubusercontent.com/dsrscientist/DSData/master/diabetes_prediction_dataset.csv",
]


# ---------------------------------------------------------------------------
# Dataset acquisition
# ---------------------------------------------------------------------------

def _try_download(dest: Path) -> bool:
    """Attempt to download the CSV from known public mirrors. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in DOWNLOAD_URLS:
        try:
            print(f"  Trying: {url}")
            urllib.request.urlretrieve(url, dest)
            # Quick sanity check - file must have expected columns
            df = pd.read_csv(dest, nrows=5)
            if "diabetes" in df.columns and "gender" in df.columns:
                print(f"  Download successful from: {url}")
                return True
            else:
                dest.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  Failed ({exc})")
    return False


def _generate_synthetic_dataset(n: int = _SYNTHETIC_N, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a representative synthetic dataset that mirrors the statistical
    properties of the real CC0 Kaggle Diabetes Prediction Dataset.

    This function is a FALLBACK used only when the real CSV cannot be obtained.
    The real dataset should be downloaded from:
        https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

    Statistical targets sourced from the Kaggle dataset card and public EDA reports.
    """
    rng = np.random.default_rng(random_state)

    n_diabetic = int(n * _DIABETIC_FRAC)
    n_healthy = n - n_diabetic

    def _gender_array(size: int) -> np.ndarray:
        return rng.choice(['Male', 'Female', 'Other'], size=size, p=_GENDER_PROBS)

    def _smoking_array(size: int) -> np.ndarray:
        return rng.choice(_SMOKING_CATS, size=size, p=_SMOKING_PROBS)

    # ---- DIABETIC PATIENTS ----
    d_age  = rng.normal(52, 15, n_diabetic).clip(1, 80)
    d_bmi  = rng.normal(32, 8, n_diabetic).clip(10, 95)
    d_hba1c   = rng.normal(7.5, 0.85, n_diabetic).clip(3.5, 9.0)
    d_glucose = rng.normal(185, 42, n_diabetic).clip(80, 300).astype(int)
    d_hyper   = rng.binomial(1, 0.26, n_diabetic)
    d_heart   = rng.binomial(1, 0.10, n_diabetic)

    # ---- HEALTHY PATIENTS ----
    h_age  = rng.normal(38, 22, n_healthy).clip(0.08, 80)
    h_bmi  = rng.normal(26, 5.5, n_healthy).clip(10, 60)
    h_hba1c   = rng.normal(5.2, 0.55, n_healthy).clip(3.5, 6.9)
    h_glucose = rng.normal(118, 28, n_healthy).clip(80, 200).astype(int)
    h_hyper   = rng.binomial(1, 0.064, n_healthy)
    h_heart   = rng.binomial(1, 0.033, n_healthy)

    diabetic_df = pd.DataFrame({
        'gender':              _gender_array(n_diabetic),
        'age':                 np.round(d_age, 1),
        'hypertension':        d_hyper,
        'heart_disease':       d_heart,
        'smoking_history':     _smoking_array(n_diabetic),
        'bmi':                 np.round(d_bmi, 2),
        'HbA1c_level':         np.round(d_hba1c, 1),
        'blood_glucose_level': d_glucose,
        'diabetes':            1,
    })

    healthy_df = pd.DataFrame({
        'gender':              _gender_array(n_healthy),
        'age':                 np.round(h_age, 1),
        'hypertension':        h_hyper,
        'heart_disease':       h_heart,
        'smoking_history':     _smoking_array(n_healthy),
        'bmi':                 np.round(h_bmi, 2),
        'HbA1c_level':         np.round(h_hba1c, 1),
        'blood_glucose_level': h_glucose,
        'diabetes':            0,
    })

    df = pd.concat([diabetic_df, healthy_df], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def ensure_dataset(data_path: Path = DATA_PATH) -> Path:
    """
    Ensure the dataset CSV exists at *data_path*.

    Priority:
      1. File already exists - use it directly.
      2. Try downloading from public mirrors.
      3. Generate a statistically representative synthetic dataset as fallback.

    Returns the path to the dataset CSV.
    """
    data_path = Path(data_path)

    if data_path.exists():
        print(f"Dataset found at: {data_path}")
        return data_path

    print("Dataset not found locally. Attempting download ...")
    if _try_download(data_path):
        return data_path

    print(
        "\nAll download attempts failed.\n"
        "Generating a representative synthetic dataset as fallback.\n"
        "NOTE: For the real dataset, download from:\n"
        "  https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset\n"
        "and place it at:  data/diabetes.csv\n"
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df = _generate_synthetic_dataset()
    df.to_csv(data_path, index=False)
    print(f"Synthetic dataset saved to: {data_path}  ({len(df):,} rows)")
    return data_path


# ---------------------------------------------------------------------------
# Loading & EDA
# ---------------------------------------------------------------------------

def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset CSV and return a DataFrame."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            "Run  python train_model.py  to download or generate it."
        )
    return pd.read_csv(data_path)


def clean_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove duplicate rows and reset index."""
    original_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = original_len - len(df)
    if verbose and removed > 0:
        print(f"  Removed {removed:,} duplicate rows.  Remaining: {len(df):,}")
    return df


def print_eda_report(df: pd.DataFrame) -> None:
    """Print a structured EDA report to stdout."""
    sep = "=" * 60

    print(f"\n{sep}")
    print("  DATASET EDA REPORT - DiaCare AI")
    print(sep)

    print(f"\nShape:    {df.shape[0]:,} rows - {df.shape[1]} columns")
    print(f"Columns:  {list(df.columns)}")

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "  No missing values.")

    print(f"\n--- Duplicate Rows ---  {df.duplicated().sum():,}")

    print("\n--- Class Distribution (diabetes) ---")
    vc = df['diabetes'].value_counts()
    for k, v in vc.items():
        print(f"  {k}: {v:,}  ({v / len(df) * 100:.1f}%)")

    if 'gender' in df.columns:
        print("\n--- Gender Distribution ---")
        gc = df['gender'].value_counts()
        for k, v in gc.items():
            print(f"  {k}: {v:,}  ({v / len(df) * 100:.1f}%)")

    print("\n--- Descriptive Statistics (numerical) ---")
    print(df.describe().round(2).to_string())
    print(f"\n{sep}\n")


def get_eda_stats(df: pd.DataFrame) -> dict:
    """Return a dict of key EDA statistics for display in the app."""
    stats: dict = {
        'total_records':   len(df),
        'num_features':    len(df.columns) - 1,   # exclude target
        'missing_values':  int(df.isnull().sum().sum()),
        'duplicate_records': int(df.duplicated().sum()),
        'positive_cases':  int((df['diabetes'] == 1).sum()),
        'negative_cases':  int((df['diabetes'] == 0).sum()),
        'male_count':   0,
        'female_count': 0,
        'other_count':  0,
    }
    if 'gender' in df.columns:
        gc = df['gender'].value_counts()
        stats['male_count']   = int(gc.get('Male', 0))
        stats['female_count'] = int(gc.get('Female', 0))
        stats['other_count']  = int(gc.get('Other', 0))
    return stats
