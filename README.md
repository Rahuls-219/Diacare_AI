# DiaCare AI — Diabetes Risk Assessment Using Machine Learning

> **Subtitle:** An Interactive Machine Learning Based Diabetes Risk Assessment System
>
> **Type:** B.Tech Data Science Final Year Project
>
> ⚠️ **Disclaimer:** DiaCare AI is an educational machine-learning risk assessment project. Its output is a statistical prediction and is **not a medical diagnosis** or a substitute for professional medical advice.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objective](#objective)
4. [Dataset](#dataset)
5. [Features](#features)
6. [Machine Learning Workflow](#machine-learning-workflow)
7. [Models Evaluated](#models-evaluated)
8. [Model Selection](#model-selection)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Application Pages](#application-pages)
11. [Installation](#installation)
12. [Running the Project](#running-the-project)
13. [Project Structure](#project-structure)
14. [Limitations](#limitations)
15. [Future Enhancements](#future-enhancements)
16. [Medical Disclaimer](#medical-disclaimer)

---

## Project Overview

**DiaCare AI** is a complete, professional B.Tech Data Science project that demonstrates the application of machine learning to healthcare risk assessment. The system analyses patient health data and produces a **statistical estimate of diabetes risk** — presented through a modern, interactive Streamlit dashboard.

The application is designed for:
- B.Tech project review and viva
- Data Science project demonstration
- ML portfolio and internship presentations
- Educational understanding of ML in healthcare

---

## Problem Statement

Diabetes is a chronic metabolic disorder affecting over 500 million people worldwide. Early identification of at-risk individuals is critical for timely intervention and prevention of complications. Traditional screening requires clinical testing, which may not be accessible to all. This project investigates whether machine learning models trained on routinely collected health indicators can generate a meaningful statistical estimate of diabetes risk.

---

## Objective

To build a professional, technically correct, and easy-to-demonstrate ML application that:

- Trains and compares multiple classification algorithms on a real, publicly documented dataset
- Selects the best model using principled evaluation criteria (ROC-AUC)
- Provides an interactive interface for diabetes risk estimation
- Clearly communicates that the output is a statistical probability, not a clinical diagnosis

---

## Dataset

| Property | Value |
|---|---|
| **Name** | Diabetes Prediction Dataset |
| **Source** | Kaggle — by iammustafatz |
| **URL** | https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset |
| **License** | CC0: Public Domain (no restrictions) |
| **Total Records** | ~100,000 rows |
| **After Deduplication** | ~96,000 rows |
| **Columns** | 9 (8 features + 1 target) |
| **Gender** | Genuine `gender` column: Male / Female / Other |
| **Positive Class** | ~8.5% (class imbalanced — documented) |
| **Missing Values** | None |
| **Duplicates** | ~3,854 (removed during preprocessing) |

### Why not PIMA Indians Diabetes Dataset?

The PIMA Indians Diabetes Dataset is **female-only** (all records are female Pima Indian women). This project requires a dataset with **both male and female patients**. The selected Kaggle dataset contains a genuine `gender` column with Male, Female, and Other categories. No gender values were fabricated or randomly assigned.

---

## Features

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Biological sex: Male / Female / Other |
| `age` | Numerical | Patient age (0–80 years) |
| `hypertension` | Binary | Has hypertension: 0 = No, 1 = Yes |
| `heart_disease` | Binary | Has heart disease: 0 = No, 1 = Yes |
| `smoking_history` | Categorical | never / current / former / ever / not current / No Info |
| `bmi` | Numerical | Body Mass Index |
| `HbA1c_level` | Numerical | Haemoglobin A1c level (%) |
| `blood_glucose_level` | Numerical | Blood glucose level (mg/dL) |
| `diabetes` | Binary (Target) | 0 = No Diabetes, 1 = Diabetes |

**Note:** This dataset does NOT contain a Pregnancies column. It has not been added.

---

## Machine Learning Workflow

```
Dataset (~100k records, CC0 Public Domain)
          ↓
DATA CLEANING
  • Remove ~3,854 duplicate rows
  • Validate feature ranges
          ↓
FEATURE PREPROCESSING (fitted on training data only)
  • Numerical: SimpleImputer(median) → StandardScaler
  • Binary: Passthrough (0/1)
  • Categorical: SimpleImputer → OrdinalEncoder
          ↓
TRAIN / TEST SPLIT
  • 80% training, 20% test
  • Stratified on target class (preserves 8.5% positive ratio)
  • random_state=42 for reproducibility
          ↓
MODEL TRAINING
  • Logistic Regression
  • Random Forest
  • Support Vector Machine
          ↓
MODEL EVALUATION
  • Accuracy, Precision, Recall, F1-Score, ROC-AUC
  • Confusion Matrix, ROC Curve
          ↓
BEST MODEL SELECTION
  • Selected by highest ROC-AUC on test set
          ↓
SAVE ARTIFACTS
  • model/diabetes_model.pkl
  • model/preprocessing.pkl
  • model/evaluation_results.json
          ↓
STREAMLIT APPLICATION
          ↓
RISK PROBABILITY ESTIMATION
```

---

## Models Evaluated

| Model | Description |
|---|---|
| Logistic Regression | Linear classifier; interpretable; coefficients indicate feature influence |
| Random Forest | Ensemble of decision trees; feature importances available |
| Support Vector Machine | Kernel-based (RBF); probability=True for risk probability output |

All models are evaluated on the **same 20% held-out test set**.

---

## Model Selection

**Primary metric: ROC-AUC**

The dataset is class-imbalanced (~91.5% negative, ~8.5% positive). In this setting, accuracy alone is misleading — a model predicting "No Diabetes" for every patient would achieve ~91.5% accuracy while providing zero useful information.

ROC-AUC (Area Under the Receiver Operating Characteristic Curve) measures a model's ability to discriminate between classes across all decision thresholds, making it the appropriate primary metric. The model with the highest ROC-AUC on the test set is automatically selected as the final model and its rationale is reported in the application.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Proportion of all correct predictions |
| Precision | Of predicted positives, how many are true positives |
| Recall | Of actual positives, how many were correctly found |
| F1-Score | Harmonic mean of Precision and Recall |
| ROC-AUC | Discrimination ability across all thresholds |
| Confusion Matrix | TP / TN / FP / FN breakdown |
| ROC Curve | Visual plot of TPR vs FPR |

All metric values displayed in the application are computed from the actual trained model on the actual test set. **No metrics are fabricated.**

---

## Application Pages

| Page | Description |
|---|---|
| 🏠 **Home** | Project overview, stat cards, workflow diagram |
| 🩺 **Risk Assessment** | Patient input form → risk probability → gauge → summary |
| 📊 **Analytics** | Dataset EDA charts: distribution, correlation, statistics |
| 📈 **Model Performance** | Real evaluation metrics, confusion matrix, ROC curve, comparison |
| ℹ️ **About Project** | Reviewer-friendly project documentation |

### Risk Thresholds (consistent throughout the application)

| Category | Probability Range |
|---|---|
| 🟢 Low Risk | < 30% |
| 🟡 Moderate Risk | 30% – 60% |
| 🔴 Higher Risk | ≥ 60% |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone / Download the Project

```bash
# If using Git:
git clone <repository-url>
cd diacare_ai

# Or simply extract the project folder and navigate to it
```

### 2. Create a Virtual Environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1: Download the Dataset (optional)

The training script will automatically attempt to download the dataset from known public mirrors. If internet access is unavailable, a statistically representative synthetic dataset will be generated automatically.

**To use the real dataset:** Download `diabetes_prediction_dataset.csv` from:
https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

Rename it to `diabetes.csv` and place it at: `data/diabetes.csv`

### Step 2: Train the Model

```bash
python train_model.py
```

This will:
- Load / generate the dataset
- Perform EDA and cleaning
- Train all models
- Select and save the best model
- Print a full evaluation report

### Step 3: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Step 4: (Optional) Re-evaluate the Model

```bash
python evaluate_model.py
```

---

## Project Structure

```
diacare_ai/
│
├── app.py                    # Main Streamlit application (5 pages)
├── train_model.py            # Model training + evaluation script
├── evaluate_model.py         # Standalone evaluation script
├── requirements.txt          # Python package dependencies
├── README.md                 # This file
├── .gitignore
│
├── data/
│   └── diabetes.csv          # Dataset (auto-downloaded or generated)
│
├── model/
│   ├── diabetes_model.pkl    # Saved best model (generated by train_model.py)
│   ├── preprocessing.pkl     # Fitted preprocessing pipeline
│   └── evaluation_results.json  # Evaluation metrics (JSON)
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading, EDA, acquisition
│   ├── preprocessing.py      # Sklearn pipeline builder
│   ├── prediction.py         # Inference, risk categorisation
│   └── visualization.py      # Plotly chart builders
│
└── assets/                   # Optional static assets
```

---

## Limitations

- **Dataset origin:** The dataset source and collection methodology are not fully documented, limiting understanding of the target population.
- **Class imbalance:** ~8.5% positive class — minority class performance is inherently harder to optimise.
- **No external validation:** The model has only been evaluated on a held-out split of the same dataset; generalization to other populations is unknown.
- **Limited clinical features:** Only 8 features are available — clinical diabetes diagnosis involves many additional factors.
- **Educational tool only:** The model output is a statistical estimate, not a clinical measurement or medical diagnosis.

---

## Future Enhancements

- Integrate larger, more diverse, and clinically validated datasets
- External validation on independent clinical datasets
- Additional health features (family history, physical activity, dietary habits)
- Advanced model explainability using SHAP values
- Hyperparameter optimisation (GridSearchCV, Optuna)
- Clinical validation in collaboration with healthcare professionals
- Secure deployment with data privacy and encryption

---

## Medical Disclaimer

> ⚠️ **DiaCare AI is an educational machine-learning risk assessment project developed as a B.Tech Data Science demonstration.**
>
> The application's output is a statistical probability generated by a trained ML model based on patterns in a historical dataset. It is **not a medical diagnosis**, not a clinical assessment, and not a substitute for professional medical advice.
>
> The output should never be interpreted as confirmation or exclusion of diabetes. Always consult a qualified healthcare professional for any medical concerns or decisions.

---

*DiaCare AI — B.Tech Data Science Project*
*Dataset: Diabetes Prediction Dataset (Kaggle CC0 Public Domain)*
*Built with Python · Streamlit · Scikit-learn · Plotly*
