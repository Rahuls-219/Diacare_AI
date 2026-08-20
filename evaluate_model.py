"""
evaluate_model.py - Standalone evaluation script for DiaCare AI.

Usage:
    python evaluate_model.py

Reloads the saved model and preprocessing pipeline, re-evaluates on a
fresh test split, and prints a detailed performance report.  This is
separate from train_model.py to demonstrate the ability to evaluate an
already-trained model without retraining.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import clean_dataset, load_dataset
from utils.preprocessing import FEATURE_ORDER, prepare_features

warnings.filterwarnings('ignore')

DATA_PATH          = Path('data') / 'diabetes.csv'
MODEL_PATH         = Path('model') / 'diabetes_model.pkl'
PREPROCESSING_PATH = Path('model') / 'preprocessing.pkl'
RESULTS_PATH       = Path('model') / 'evaluation_results.json'
RANDOM_STATE       = 42
TEST_SIZE          = 0.20


def main() -> None:
    print('\n' + '=' * 60)
    print('  DiaCare AI - Model Evaluation Script')
    print('=' * 60)

    # -- Check required files --------------------------------------------------
    for fpath in [DATA_PATH, MODEL_PATH, PREPROCESSING_PATH]:
        if not fpath.exists():
            print(f'\n  ERROR: Required file not found: {fpath}')
            print('  Run  python train_model.py  first.\n')
            sys.exit(1)

    # -- Load artifacts --------------------------------------------------------
    print('\n[1/4] Loading saved model and preprocessing pipeline ...')
    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSING_PATH)
    print(f'  Model type: {type(model).__name__}')

    # -- Prepare test set (same split parameters as training) ------------------
    print('\n[2/4] Preparing test data ...')
    df = pd.read_csv(DATA_PATH)
    df = clean_dataset(df, verbose=False)
    X, y = prepare_features(df)

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_test_proc = preprocessor.transform(X_test)
    print(f'  Test set size : {len(X_test):,} rows')

    # -- Compute metrics -------------------------------------------------------
    print('\n[3/4] Computing evaluation metrics ...')
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    auc     = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # -- Print report ----------------------------------------------------------
    print('\n' + '-' * 60)
    print(f'  {"METRIC":<20}  {"VALUE":>10}')
    print('-' * 60)
    for label, val in [
        ('Accuracy',  acc),
        ('Precision', prec),
        ('Recall',    rec),
        ('F1-Score',  f1),
        ('ROC-AUC',   auc),
    ]:
        print(f'  {label:<20}  {val:>10.4f}')
    print('-' * 60)

    print('\n  Classification Report:')
    print(classification_report(
        y_test, y_pred,
        target_names=['No Diabetes (0)', 'Diabetes (1)']
    ))

    print('  Confusion Matrix:')
    print(f'  TN={cm[0,0]:,}  FP={cm[0,1]:,}')
    print(f'  FN={cm[1,0]:,}  TP={cm[1,1]:,}')

    # -- Verify metrics match saved results ------------------------------------
    print('\n[4/4] Cross-checking against saved results ...')
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            saved = json.load(f)
        best = saved.get('best_model', 'unknown')
        saved_auc = saved.get('models', {}).get(best, {}).get('roc_auc', None)
        if saved_auc is not None:
            diff = abs(auc - saved_auc)
            status = '- OK' if diff < 0.001 else '- MISMATCH'
            print(f'  Saved ROC-AUC : {saved_auc:.4f}')
            print(f'  Re-evaluated  : {auc:.4f}')
            print(f'  Difference    : {diff:.6f}  {status}')
    else:
        print('  evaluation_results.json not found - skipping cross-check.')

    print('\n' + '=' * 60)
    print('  Evaluation complete.')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
