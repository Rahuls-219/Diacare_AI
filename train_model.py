"""
train_model.py - Model training script for DiaCare AI.

Usage:
    python train_model.py

This script:
  1. Ensures the dataset exists (downloads or generates it if needed).
  2. Performs exploratory data analysis and cleaning.
  3. Builds and fits the preprocessing pipeline.
  4. Trains and cross-evaluates three ML models:
       - Logistic Regression
       - Random Forest Classifier
       - Support Vector Machine (SVC)
  5. Selects the best model by ROC-AUC (preferred over accuracy for
     imbalanced data - ~8.5 % positive class).
  6. Saves the best model and preprocessing pipeline to model/.
  7. Saves evaluation results to model/evaluation_results.json.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# -- Local imports ------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import clean_dataset, ensure_dataset, print_eda_report
from utils.preprocessing import (
    FEATURE_ORDER,
    build_preprocessing_pipeline,
    prepare_features,
    save_preprocessing,
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH          = Path('data') / 'diabetes.csv'
MODEL_DIR          = Path('model')
MODEL_PATH         = MODEL_DIR / 'diabetes_model.pkl'
PREPROCESSING_PATH = MODEL_DIR / 'preprocessing.pkl'
RESULTS_PATH       = MODEL_DIR / 'evaluation_results.json'

RANDOM_STATE = 42
TEST_SIZE    = 0.20    # 80 / 20 train-test split

# ---------------------------------------------------------------------------
# Model candidates
# ---------------------------------------------------------------------------

MODEL_CANDIDATES = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        C=1.0,
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    'Support Vector Machine': CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        cv=3,
    ),
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                    model_name: str) -> dict:
    """Compute all evaluation metrics for one model on the test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, zero_division=0)
    rec       = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)
    cm        = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    print(f"\n  {'-' * 40}")
    print(f"  {model_name}")
    print(f"  {'-' * 40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])}")

    return {
        'accuracy':          round(float(acc),     4),
        'precision':         round(float(prec),    4),
        'recall':            round(float(rec),     4),
        'f1':                round(float(f1),      4),
        'roc_auc':           round(float(roc_auc), 4),
        'confusion_matrix':  cm,
        'fpr':               fpr.tolist(),
        'tpr':               tpr.tolist(),
    }


# ---------------------------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------------------------

def main() -> None:
    print('\n' + '=' * 60)
    print('  DiaCare AI - Model Training Script')
    print('=' * 60)

    # -- Step 1: Ensure dataset exists ----------------------------------------
    print('\n[1/7] Ensuring dataset ...')
    ensure_dataset(DATA_PATH)

    # -- Step 2: Load and inspect ---------------------------------------------
    print('\n[2/7] Loading and inspecting dataset ...')
    df = pd.read_csv(DATA_PATH)
    print_eda_report(df)

    # -- Step 3: Clean --------------------------------------------------------
    print('[3/7] Cleaning dataset ...')
    df = clean_dataset(df, verbose=True)
    print(f'  Clean dataset shape: {df.shape}')

    # Verify both genders present
    if 'gender' in df.columns:
        gc = df['gender'].value_counts()
        print(f'  Gender counts: {gc.to_dict()}')
        assert gc.get('Male', 0) > 0,   'ERROR: No male records found!'
        assert gc.get('Female', 0) > 0, 'ERROR: No female records found!'
        print('  [OK] Both male and female patients confirmed in dataset.')

    # -- Step 4: Prepare features ---------------------------------------------
    print('\n[4/7] Preparing features ...')
    X, y = prepare_features(df)
    print(f'  Feature shape : {X.shape}')
    print(f'  Target shape  : {y.shape}')
    print(f'  Class balance : {y.value_counts().to_dict()}')

    # Train / Test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f'  Train : {X_train.shape[0]:,} rows')
    print(f'  Test  : {X_test.shape[0]:,} rows')

    # -- Step 5: Fit preprocessing on TRAIN only (no leakage) -----------------
    print('\n[5/7] Fitting preprocessing pipeline (on training data only) ...')
    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc  = preprocessor.transform(X_test)          # only transform, never fit
    print(f'  Preprocessed train shape: {X_train_proc.shape}')

    # -- Step 6: Train and evaluate all models ---------------------------------
    print('\n[6/7] Training and evaluating models ...')
    all_results: dict[str, dict] = {}
    trained_models: dict[str, object] = {}

    for name, model in MODEL_CANDIDATES.items():
        print(f'\n  Training: {name} ...')
        t0 = time.time()
        model.fit(X_train_proc, y_train)
        elapsed = time.time() - t0
        print(f'  Training time: {elapsed:.1f}s')

        metrics = _evaluate_model(model, X_test_proc, y_test, name)
        all_results[name] = metrics
        trained_models[name] = model

    # -- Step 7: Select best model by ROC-AUC ---------------------------------
    print('\n[7/7] Selecting best model ...')
    print('\n  MODEL COMPARISON (sorted by ROC-AUC)')
    print(f'  {"Model":<30} {"Acc":>6} {"Prec":>6} {"Rec":>6} {"F1":>6} {"AUC":>6}')
    print(f'  {"-"*30} {"-"*6} {"-"*6} {"-"*6} {"-"*6} {"-"*6}')

    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1]['roc_auc'],
        reverse=True,
    )
    for name, res in sorted_models:
        flag = '  <- BEST' if name == sorted_models[0][0] else ''
        print(
            f'  {name:<30} '
            f'{res["accuracy"]:>6.4f} '
            f'{res["precision"]:>6.4f} '
            f'{res["recall"]:>6.4f} '
            f'{res["f1"]:>6.4f} '
            f'{res["roc_auc"]:>6.4f}'
            f'{flag}'
        )

    best_name  = sorted_models[0][0]
    best_model = trained_models[best_name]
    best_res   = all_results[best_name]

    print(f'\n  Selected model: {best_name}')
    print(f'  ROC-AUC: {best_res["roc_auc"]:.4f}')
    print(
        f'\n  Reason: ROC-AUC was chosen as the primary selection metric because '
        f'the dataset is class-imbalanced (~8.5% positive). ROC-AUC better reflects '
        f'discrimination ability than accuracy in imbalanced settings.'
    )

    # Feature names after preprocessing
    feature_names_out = (
        ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']   # numerical (scaled)
        + ['hypertension', 'heart_disease']                       # binary (passthrough)
        + ['gender_enc', 'smoking_enc']                           # categorical (encoded)
    )

    # -- Save artifacts --------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f'\n  Model saved        -> {MODEL_PATH}')
    save_preprocessing(preprocessor, PREPROCESSING_PATH)

    # Build evaluation results JSON
    results_json = {
        'best_model':        best_name,
        'selection_metric':  'ROC-AUC',
        'selection_reason':  (
            f'{best_name} achieved the highest ROC-AUC ({best_res["roc_auc"]:.4f}) '
            f'on the held-out test set. ROC-AUC is preferred over accuracy for this '
            f'dataset due to class imbalance (~8.5 % positive class).'
        ),
        'dataset_info': {
            'name':     'Diabetes Prediction Dataset',
            'source':   'Kaggle - iammustafatz',
            'url':      'https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset',
            'license':  'CC0: Public Domain',
            'total_records': len(df),
            'num_features':  len(FEATURE_ORDER),
            'features':      FEATURE_ORDER,
            'target':        'diabetes',
        },
        'models':              all_results,
        'feature_names_out':   feature_names_out,
        'test_size':           TEST_SIZE,
        'random_state':        RANDOM_STATE,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f'  Results saved      -> {RESULTS_PATH}')

    print('\n' + '=' * 60)
    print('  Training complete! Run:  streamlit run app.py')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
