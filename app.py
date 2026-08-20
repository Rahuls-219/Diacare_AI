"""
app.py — DiaCare AI Streamlit Application.

DiaCare AI: Diabetes Risk Assessment Using Machine Learning
An Interactive Machine Learning Based Diabetes Risk Assessment System

IMPORTANT: This is an educational ML risk assessment system.
           It is NOT a medical diagnosis tool.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import get_eda_stats, load_dataset
from utils.prediction import (
    LOW_RISK_THRESHOLD,
    MODERATE_RISK_THRESHOLD,
    get_feature_influence,
    load_model,
    load_preprocessing_pipeline,
    predict_risk,
)
from utils.visualization import (
    plot_confusion_matrix,
    plot_feature_distribution,
    plot_feature_importance,
    plot_gender_distribution,
    plot_model_comparison,
    plot_outcome_distribution,
    plot_risk_gauge,
    plot_roc_curve,
    plot_correlation_heatmap,
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH          = Path('data') / 'diabetes.csv'
MODEL_PATH         = Path('model') / 'diabetes_model.pkl'
PREPROCESSING_PATH = Path('model') / 'preprocessing.pkl'
RESULTS_PATH       = Path('model') / 'evaluation_results.json'

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title='DiaCare AI — Diabetes Risk Assessment',
    page_icon='🩺',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Typography ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}

/* ── Background ─────────────────────────────────────────────────────────── */
.stApp {
    background-color: #0F172A;
}

/* Main content text */
.stApp, .stApp p, .stApp label, .stApp span {
    color: #E2E8F0;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #F8FAFC !important;
}

/* Streamlit markdown */
[data-testid="stMarkdownContainer"] {
    color: #E2E8F0;
}

/* Input boxes */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border-color: #475569 !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E7490 0%, #0891B2 100%);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #FFFFFF !important;
    font-size: 14px;
}

/* ── Cards ───────────────────────────────────────────────────────────────── */
.diai-card {
    background: #1E293B;
    border-radius: 12px;
    padding: 24px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 8px rgba(0,0,0,0.04);
    border: 1px solid #E2E8F0;
    margin-bottom: 16px;
}

.diai-metric-card {
    background: #1E293B;
    border-radius: 10px;
    padding: 18px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #E2E8F0;
    text-align: center;
}

.diai-metric-card .metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #0891B2;
}

.diai-metric-card .metric-label {
    font-size: 12px;
    color: #64748B;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* ── Section headers ─────────────────────────────────────────────────────── */
.diai-section-header {
    font-size: 18px;
    font-weight: 600;
    color: #F8FAFC;
    padding-bottom: 8px;
    border-bottom: 2px solid #0891B2;
    margin-bottom: 20px;
}

/* ── Risk badge ──────────────────────────────────────────────────────────── */
.risk-badge-low {
    background: #D1FAE5; color: #065F46;
    padding: 6px 16px; border-radius: 20px;
    font-weight: 600; font-size: 14px;
    display: inline-block; margin: 4px 0;
}
.risk-badge-moderate {
    background: #FEF3C7; color: #92400E;
    padding: 6px 16px; border-radius: 20px;
    font-weight: 600; font-size: 14px;
    display: inline-block; margin: 4px 0;
}
.risk-badge-higher {
    background: #FEE2E2; color: #991B1B;
    padding: 6px 16px; border-radius: 20px;
    font-weight: 600; font-size: 14px;
    display: inline-block; margin: 4px 0;
}

/* ── Disclaimer ──────────────────────────────────────────────────────────── */
.disclaimer-box {
    background: #FFF7ED;
    border: 1px solid #FDBA74;
    border-left: 4px solid #F97316;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    color: #7C2D12;
    margin: 20px 0;
}

/* ── Workflow ─────────────────────────────────────────────────────────────── */
.workflow-step {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 10px 16px;
    text-align: center;
    font-size: 13px;
    font-weight: 600;
    color: #1E40AF;
}

/* ── Home stat cards ─────────────────────────────────────────────────────── */
.home-stat-card {
    background: linear-gradient(135deg, #0891B2 0%, #0E7490 100%);
    border-radius: 12px;
    padding: 22px 18px;
    text-align: center;
    color: white;
}
.home-stat-card .stat-value {
    font-size: 26px;
    font-weight: 700;
    color: white;
}
.home-stat-card .stat-label {
    font-size: 12px;
    opacity: 0.85;
    margin-top: 4px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.home-stat-card .stat-sub {
    font-size: 11px;
    opacity: 0.7;
    margin-top: 2px;
}

/* ── About sections ──────────────────────────────────────────────────────── */
.about-section {
   background: #1E293B;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 14px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.about-section h4 {
    color: #0891B2;
    font-weight: 600;
    margin-bottom: 8px;
    font-size: 15px;
}

/* ── Hide Streamlit branding ─────────────────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner='Loading model ...')
def _load_model_cached():
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner='Loading preprocessing pipeline ...')
def _load_preprocessor_cached():
    return load_preprocessing_pipeline(PREPROCESSING_PATH)


@st.cache_data(show_spinner='Loading dataset ...')
def _load_data_cached() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_data(show_spinner=False)
def _load_results_cached() -> dict:
    if not RESULTS_PATH.exists():
        return {}
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _check_artifacts() -> bool:
    """Return True if all required files exist; show error otherwise."""
    missing = []
    for p in [MODEL_PATH, PREPROCESSING_PATH, RESULTS_PATH]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        st.error(
            '**Model not trained yet.** Missing files:\n\n'
            + '\n'.join(f'• `{m}`' for m in missing)
            + '\n\n**Run the training script first:**\n```\npython train_model.py\n```'
        )
        return False
    return True


# ---------------------------------------------------------------------------
# ── PAGE: HOME ────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def page_home():
    results = _load_results_cached()
    dataset_info = results.get('dataset_info', {})

    # ── Hero header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 32px 0 20px 0;">
        <div style="font-size:48px; margin-bottom:4px;">🩺</div>
        <h1 style="font-size:38px; font-weight:700; color:#F8FAFC; margin:0;">
            DiaCare AI
        </h1>
        <p style="font-size:20px; color:#0891B2; font-weight:600; margin:4px 0;">
            Diabetes Risk Assessment Using Machine Learning
        </p>
        <p style="font-size:15px; color:#94A3B8; max-width:620px; margin:12px auto 0 auto; line-height:1.6;">
            An interactive machine-learning application that estimates diabetes risk 
            from patient health information using trained classification models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('---')

    # ── Stat cards ────────────────────────────────────────────────────────────
    total_records = dataset_info.get('total_records', '—')
    num_features  = dataset_info.get('num_features', '—')
    best_model    = results.get('best_model', '—')
    dataset_name  = 'Kaggle CC0'

    if isinstance(total_records, int):
        total_records = f'{total_records:,}'

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, sub in [
        (c1, dataset_name,   'Dataset',       'Diabetes Prediction'),
        (c2, total_records,  'Patients',       'Training Records'),
        (c3, str(num_features), 'Features',   'Health Indicators'),
        (c4, best_model,     'Final Model',    'Best by ROC-AUC'),
    ]:
        col.markdown(f"""
        <div class="home-stat-card">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Two-column layout: about + workflow ───────────────────────────────────
    col_left, col_right = st.columns([1.1, 0.9], gap='large')

    with col_left:
        st.markdown('<div class="diai-section-header">About DiaCare AI</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="diai-card">
        <p style="color:#E2E8F0; line-height:1.75; font-size:14px;">
        <strong>DiaCare AI</strong> is a B.Tech Data Science project that demonstrates
        how machine learning can be applied to healthcare risk assessment.
        The system analyses patient health data and outputs a statistical probability
        of diabetes risk — <em>not a medical diagnosis</em>.
        </p>

        <p style="color:#E2E8F0; line-height:1.75; font-size:14px; margin-top:12px;">
        The application uses the <strong>Diabetes Prediction Dataset</strong>
        (Kaggle, CC0 Public Domain) which contains records for both <strong>Male</strong>
        and <strong>Female</strong> patients across eight clinically meaningful health
        indicators including HbA1c level, blood glucose, BMI, age, hypertension,
        heart disease, and smoking history.
        </p>

        <p style="color:#E2E8F0; line-height:1.75; font-size:14px; margin-top:12px;">
        Three ML classification algorithms are trained and compared:
        Logistic Regression, Random Forest, and SVM.
        The best model is selected by <strong>ROC-AUC</strong> — a metric that 
        accounts for class imbalance.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="diai-section-header">Application Workflow</div>', unsafe_allow_html=True)
        steps = [
            ('📋', 'Patient Data', 'Enter health information'),
            ('⚙️', 'Data Preprocessing', 'Scale & encode features'),
            ('🧠', 'ML Model', 'Trained classifier'),
            ('📊', 'Risk Probability', 'Computed by model'),
            ('🩺', 'Risk Assessment', 'Low / Moderate / Higher'),
        ]
        for i, (icon, title, desc) in enumerate(steps):
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:8px;">
                <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px;
                            padding:10px 14px; flex:1; display:flex; align-items:center; gap:10px;">
                    <span style="font-size:20px;">{icon}</span>
                    <div>
                        <div style="font-weight:600; font-size:13px; color:#1E40AF;">{title}</div>
                        <div style="font-size:11px;color:#94A3B8;">{desc}</div>
                    </div>
                </div>
            </div>
            {'<div style="text-align:center; color:#CBD5E1; font-size:12px;">↓</div>' if i < len(steps)-1 else ''}
            """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── CTA button ────────────────────────────────────────────────────────────
    col_cta = st.columns([1, 2, 1])[1]
    with col_cta:
        if st.button('🩺  Start Risk Assessment', use_container_width=True, type='primary'):
            st.session_state['page'] = 'Risk Assessment'
            st.rerun()

    # ── Medical disclaimer ────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ <strong>Disclaimer:</strong> DiaCare AI is an educational machine-learning risk assessment 
    project. Its output is a statistical prediction and is <strong>not a medical diagnosis</strong> 
    or a substitute for professional medical advice. Always consult a qualified healthcare 
    professional for medical decisions.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ── PAGE: RISK ASSESSMENT ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def page_risk_assessment():
    if not _check_artifacts():
        return

    model        = _load_model_cached()
    preprocessor = _load_preprocessor_cached()

    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="font-size:28px; font-weight:700;color:#F8FAFC; margin:0;">
            🩺 Diabetes Risk Assessment
        </h2>
        <p style="color:#94A3B8; font-size:14px; margin-top:4px;">
            Enter patient health information below and click <strong>Assess Diabetes Risk</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ <strong>Disclaimer:</strong> DiaCare AI is an educational machine-learning risk assessment 
    project. Its output is a <strong>statistical prediction, not a medical diagnosis</strong> and 
    is not a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form('risk_form'):

        # ── PATIENT INFORMATION ───────────────────────────────────────────────
        st.markdown('<div class="diai-section-header">👤 Patient Information</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                'Gender',
                options=['Male', 'Female', 'Other'],
                help='Biological sex of the patient as recorded in the clinical dataset.',
            )
            age = st.slider(
                'Age (years)',
                min_value=1, max_value=80, value=35, step=1,
                help='Patient age in years. Range: 1–80.',
            )

        with col2:
            hypertension = st.radio(
                'Hypertension',
                options=['No', 'Yes'],
                horizontal=True,
                help='Whether the patient has been diagnosed with hypertension (high blood pressure).',
            )
            heart_disease = st.radio(
                'Heart Disease',
                options=['No', 'Yes'],
                horizontal=True,
                help='Whether the patient has a diagnosed heart condition.',
            )

        # ── HEALTH PARAMETERS ─────────────────────────────────────────────────
        st.markdown('<div class="diai-section-header">🏥 Health Parameters</div>', unsafe_allow_html=True)

        col3, col4, col5 = st.columns(3)

        with col3:
            smoking_history = st.selectbox(
                'Smoking History',
                options=['never', 'No Info', 'current', 'former', 'ever', 'not current'],
                help='Patient\'s reported smoking status.',
            )
            bmi = st.number_input(
                'BMI (Body Mass Index)',
                min_value=10.0, max_value=70.0, value=25.0, step=0.1, format='%.1f',
                help='Body Mass Index = weight(kg) / height(m)². Normal: 18.5–24.9.',
            )

        with col4:
            hba1c = st.number_input(
                'HbA1c Level (%)',
                min_value=3.5, max_value=15.0, value=5.5, step=0.1, format='%.1f',
                help=(
                    'Haemoglobin A1c — average blood sugar over ~3 months.\n'
                    'Normal: < 5.7%  |  Pre-diabetic: 5.7–6.4%  |  Diabetic: ≥ 6.5%'
                ),
            )

        with col5:
            blood_glucose = st.number_input(
                'Blood Glucose Level (mg/dL)',
                min_value=70, max_value=400, value=120, step=1,
                help='Fasting blood glucose. Normal: 70–99 mg/dL  |  Pre-diabetic: 100–125.',
            )

        submit = st.form_submit_button(
            '🔍  Assess Diabetes Risk',
            use_container_width=True,
            type='primary',
        )

    # ── Validation + Prediction ───────────────────────────────────────────────
    if submit:
        errors = []
        if bmi <= 0:
            errors.append('BMI must be greater than 0.')
        if hba1c < 3.5:
            errors.append('HbA1c Level must be at least 3.5%.')
        if blood_glucose < 70:
            errors.append('Blood Glucose Level must be at least 70 mg/dL.')
        if age < 1:
            errors.append('Age must be at least 1 year.')

        if errors:
            for e in errors:
                st.error(f'⚠️ {e}')
            return

        # Build input dict matching FEATURE_ORDER exactly
        input_data = {
            'age':                 float(age),
            'bmi':                 float(bmi),
            'HbA1c_level':         float(hba1c),
            'blood_glucose_level': int(blood_glucose),
            'hypertension':        1 if hypertension == 'Yes' else 0,
            'heart_disease':       1 if heart_disease == 'Yes' else 0,
            'gender':              gender,
            'smoking_history':     smoking_history,
        }

        try:
            result = predict_risk(input_data, model, preprocessor)
        except Exception as exc:
            st.error(f'Prediction error: {exc}')
            return

        prob   = result['probability_pct']
        cat    = result['risk_category']
        desc   = result['description']
        color  = result['risk_color']

        # ── Gauge + risk badge ───────────────────────────────────────────────
        st.markdown('---')
        st.markdown('<div class="diai-section-header">📊 Estimated Diabetes Risk</div>',
                    unsafe_allow_html=True)

        col_g, col_r = st.columns([1, 1.1])

        with col_g:
            gauge_fig = plot_risk_gauge(result['probability'], cat)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_r:
            st.markdown('<br>', unsafe_allow_html=True)

            badge_cls = {
                'LOW RISK':      'low',
                'MODERATE RISK': 'moderate',
                'HIGHER RISK':   'higher',
            }.get(cat, 'low')

            st.markdown(f"""
            <div class="diai-card">
                <p style="font-size:13px; color:#94A3B8; margin:0 0 8px 0; font-weight:500;">
                    RISK PROBABILITY
                </p>
                <p style="font-size:40px; font-weight:700; color:{color}; margin:0;">
                    {prob}%
                </p>
                <div class="risk-badge-{badge_cls}" style="margin-top:10px;">
                    {result['risk_icon']} {cat}
                </div>
                <p style="font-size:13px; color:#475569; margin-top:14px; line-height:1.65;">
                    {desc}
                </p>

            </div>
            """, unsafe_allow_html=True)

        # ── Patient summary table ────────────────────────────────────────────
        st.markdown('<div class="diai-section-header">📋 Patient Assessment Summary</div>',
                    unsafe_allow_html=True)

        display_data = {
            'Feature':       ['Gender', 'Age', 'Hypertension', 'Heart Disease',
                              'Smoking History', 'BMI', 'HbA1c Level', 'Blood Glucose Level'],
            'Entered Value': [
                gender,
                f'{age} years',
                hypertension,
                heart_disease,
                smoking_history,
                f'{bmi:.1f}',
                f'{hba1c:.1f}%',
                f'{blood_glucose} mg/dL',
            ],
        }
        summary_df = pd.DataFrame(display_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # ── Understanding the result ─────────────────────────────────────────
        st.markdown('<div class="diai-section-header">💡 Understanding Your Result</div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div class="diai-card">
        <p style="color:#E2E8F0; font-size:14px; line-height:1.7;">
        The model estimated a diabetes risk probability of <strong>{prob}%</strong> 
        based on the health information provided. This is a <em>statistical estimate</em> 
        generated by a machine-learning model trained on a diabetes dataset — 
        it is not a clinical measurement or a medical diagnosis.
        </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature influence (if available)
        results_data = _load_results_cached()
        feature_names_out = results_data.get('feature_names_out', [])
        influence = get_feature_influence(model, feature_names_out)

        if influence:
            st.markdown("""
            <p style="color:#E2E8F0; font-size:14px; line-height:1.7; margin-top:8px;">
            <strong>Features that had greater influence on the model's assessment:</strong>
            </p>
            <p style="font-size:12px; color:#94A3B8; margin-top:4px; font-style:italic;">
            Note: Feature influence indicates which inputs the model weighted more heavily 
            during prediction. It does <strong>not</strong> imply medical causation.
            </p>
            """, unsafe_allow_html=True)

            # Sort and show top features
            top_features = sorted(influence.items(), key=lambda x: x[1], reverse=True)[:5]
            for feat, score in top_features:
                bar_pct = int((score / max(influence.values())) * 100)
                st.markdown(f"""
                <div style="margin:6px 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                        <span style="font-size:13px; color:#E2E8F0;">{feat}</span>
                        <span style="font-size:12px;color:#94A3B8;">{score:.4f}</span>
                    </div>
                    <div style="background:#E2E8F0; border-radius:4px; height:6px;">
                        <div style="background:#0891B2; width:{bar_pct}%; height:6px; border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Final disclaimer ─────────────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer-box" style="margin-top:24px;">
        ⚠️ <strong>Important:</strong> DiaCare AI is an educational machine-learning risk 
        assessment project developed as a B.Tech Data Science demonstration. 
        The estimated risk probability is based on statistical patterns in a training dataset 
        and <strong>cannot diagnose diabetes</strong>. 
        Please consult a qualified healthcare professional for any medical concerns.
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ── PAGE: ANALYTICS ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def page_analytics():
    if not DATA_PATH.exists():
        st.warning(
            'Dataset not found. Run `python train_model.py` first to generate the dataset.'
        )
        return

    df = _load_data_cached()
    stats = get_eda_stats(df)

    st.markdown("""
    <div style="margin-bottom:20px;">
        <h2 style="font-size:28px; font-weight:700; color:#F8FAFC; margin:0;">
            📊 Dataset Analytics
        </h2>
        <p style="color:#94A3B8; font-size:14px; margin-top:4px;">
            Exploratory analysis of the Diabetes Prediction Dataset (CC0 Public Domain).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Overview stat cards ───────────────────────────────────────────────────
    st.markdown('<div class="diai-section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    overview_items = [
        ('Total Records',   f"{stats['total_records']:,}",      ''),
        ('Features',        str(stats['num_features']),          'Predictors'),
        ('Missing Values',  str(stats['missing_values']),        'In dataset'),
        ('Duplicates',      str(stats['duplicate_records']),     'Before cleaning'),
    ]
    for col, (label, value, sub) in zip(cols, overview_items):
        col.markdown(f"""
        <div class="diai-metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {f'<div style="font-size:11px;color:#94A3B8;">{sub}</div>' if sub else ''}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    cols2 = st.columns(4)
    class_items = [
        ('Positive Cases',  f"{stats['positive_cases']:,}",   'Diabetes = 1'),
        ('Negative Cases',  f"{stats['negative_cases']:,}",   'Diabetes = 0'),
        ('Male Patients',   f"{stats['male_count']:,}",        'Gender = Male'),
        ('Female Patients', f"{stats['female_count']:,}",      'Gender = Female'),
    ]
    for col, (label, value, sub) in zip(cols2, class_items):
        col.markdown(f"""
        <div class="diai-metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            <div style="font-size:11px;color:#94A3B8;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('---')

    # ── Charts row 1: Outcome + Gender ────────────────────────────────────────
    col_a, col_b = st.columns(2, gap='medium')

    with col_a:
        st.markdown('<div class="diai-section-header">Outcome Distribution</div>',
                    unsafe_allow_html=True)
        fig_out = plot_outcome_distribution(df)
        st.plotly_chart(fig_out, use_container_width=True)

    with col_b:
        st.markdown('<div class="diai-section-header">Gender Distribution</div>',
                    unsafe_allow_html=True)
        fig_gen = plot_gender_distribution(df)
        st.plotly_chart(fig_gen, use_container_width=True)

    st.markdown('---')

    # ── Feature distribution ──────────────────────────────────────────────────
    st.markdown('<div class="diai-section-header">Feature Distribution</div>',
                unsafe_allow_html=True)

    num_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    feature_labels = {
        'age':                 'Age (years)',
        'bmi':                 'BMI',
        'HbA1c_level':         'HbA1c Level (%)',
        'blood_glucose_level': 'Blood Glucose Level (mg/dL)',
    }
    selected_feature = st.selectbox(
        'Select a feature to explore:',
        options=num_features,
        format_func=lambda x: feature_labels.get(x, x),
    )
    fig_dist = plot_feature_distribution(df, selected_feature)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown('---')

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown('<div class="diai-section-header">Correlation Heatmap</div>',
                unsafe_allow_html=True)
    st.caption('Pearson correlation coefficients among all numerical features. Values range from −1 to +1.')
    fig_corr = plot_correlation_heatmap(df)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('---')

    # ── Descriptive statistics ────────────────────────────────────────────────
    st.markdown('<div class="diai-section-header">Descriptive Statistics</div>',
                unsafe_allow_html=True)
    st.dataframe(df.describe().round(3), use_container_width=True)


# ---------------------------------------------------------------------------
# ── PAGE: MODEL PERFORMANCE ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def page_model_performance():
    if not _check_artifacts():
        return

    results = _load_results_cached()
    best_model_name = results.get('best_model', '')
    models_data     = results.get('models', {})

    if not models_data:
        st.warning('No evaluation results found. Run `python train_model.py` first.')
        return

    best_res = models_data.get(best_model_name, {})

    st.markdown("""
    <div style="margin-bottom:20px;">
        <h2 style="font-size:28px; font-weight:700; color:#F8FAFC; margin:0;">
            📈 Model Performance
        </h2>
        <p style="color:#94A3B8; font-size:14px; margin-top:4px;">
            Evaluation metrics computed on the held-out 20% test set. All values are real — not fabricated.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Best model selection banner ───────────────────────────────────────────
    reason = results.get('selection_reason', '')
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #0891B2, #0E7490);
                border-radius:12px; padding:18px 24px; color:white; margin-bottom:24px;">
        <p style="margin:0; font-size:12px; opacity:0.8; font-weight:500; text-transform:uppercase; letter-spacing:0.06em;">
            SELECTED FINAL MODEL
        </p>
        <p style="margin:4px 0 6px 0; font-size:24px; font-weight:700;">{best_model_name}</p>
        <p style="margin:0; font-size:13px; opacity:0.88; line-height:1.6;">{reason}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown('<div class="diai-section-header">Key Metrics — Best Model</div>',
                unsafe_allow_html=True)

    metric_items = [
        ('Accuracy',  'accuracy'),
        ('Precision', 'precision'),
        ('Recall',    'recall'),
        ('F1-Score',  'f1'),
        ('ROC-AUC',   'roc_auc'),
    ]
    cols = st.columns(5)
    for col, (label, key) in zip(cols, metric_items):
        val = best_res.get(key, 0)
        col.markdown(f"""
        <div class="diai-metric-card">
            <div class="metric-value">{val:.4f}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('---')

    # ── Confusion matrix + ROC curve ─────────────────────────────────────────
    col_cm, col_roc = st.columns(2, gap='medium')

    with col_cm:
        st.markdown('<div class="diai-section-header">Confusion Matrix</div>',
                    unsafe_allow_html=True)
        cm = best_res.get('confusion_matrix', [[0, 0], [0, 0]])
        fig_cm = plot_confusion_matrix(cm)
        st.plotly_chart(fig_cm, use_container_width=True)

        # CM explanation
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        st.markdown(f"""
        <div style="font-size:12px;color:#94A3B8; line-height:1.8;">
            <strong>TN</strong> (True Negative) = {tn:,} &nbsp;|&nbsp;
            <strong>FP</strong> (False Positive) = {fp:,}<br>
            <strong>FN</strong> (False Negative) = {fn:,} &nbsp;|&nbsp;
            <strong>TP</strong> (True Positive) = {tp:,}
        </div>
        """, unsafe_allow_html=True)

    with col_roc:
        st.markdown('<div class="diai-section-header">ROC Curve</div>',
                    unsafe_allow_html=True)
        fpr  = best_res.get('fpr', [0, 1])
        tpr  = best_res.get('tpr', [0, 1])
        auc  = best_res.get('roc_auc', 0.5)
        fig_roc = plot_roc_curve(fpr, tpr, auc)
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown('---')

    # ── Feature importance ────────────────────────────────────────────────────
    model = _load_model_cached()
    feature_names_out = results.get('feature_names_out', [])
    influence = get_feature_influence(model, feature_names_out)

    if influence:
        st.markdown('<div class="diai-section-header">Feature Influence</div>',
                    unsafe_allow_html=True)
        st.caption(
            'Features with higher scores had greater influence on the model\'s predictions. '
            'This does not imply medical causation.'
        )
        names = list(influence.keys())
        scores = list(influence.values())
        fig_imp = plot_feature_importance(names, scores, best_model_name)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown('---')

    # ── Model comparison table + chart ───────────────────────────────────────
    st.markdown('<div class="diai-section-header">Model Comparison</div>',
                unsafe_allow_html=True)

    rows = []
    for name, res in models_data.items():
        rows.append({
            'Model':     name,
            'Accuracy':  res.get('accuracy', 0),
            'Precision': res.get('precision', 0),
            'Recall':    res.get('recall', 0),
            'F1-Score':  res.get('f1', 0),
            'ROC-AUC':   res.get('roc_auc', 0),
            'Selected':  '✅ Best' if name == best_model_name else '',
        })

    comp_df = pd.DataFrame(rows).set_index('Model')
    st.dataframe(
        comp_df.style.format({
            'Accuracy': '{:.4f}', 'Precision': '{:.4f}',
            'Recall': '{:.4f}', 'F1-Score': '{:.4f}', 'ROC-AUC': '{:.4f}',
        }).highlight_max(
            subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            color='#CFFAFE',
        ),
        use_container_width=True,
    )

    st.markdown('<br>', unsafe_allow_html=True)
    fig_comp = plot_model_comparison({
        name: res for name, res in models_data.items()
    })
    st.plotly_chart(fig_comp, use_container_width=True)


# ---------------------------------------------------------------------------
# ── PAGE: ABOUT PROJECT ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def page_about():
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h2 style="font-size:28px; font-weight:700; color:#F8FAFC; margin:0;">
            ℹ️ About DiaCare AI
        </h2>
        <p style="color:#94A3B8; font-size:14px; margin-top:4px;">
            A reviewer-friendly overview of the project — suitable for B.Tech viva, 
            project demonstration, and portfolio presentation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sections = [
        (
            '1. Problem Statement',
            """
            Diabetes is a chronic metabolic disorder affecting millions globally. 
            Early identification of at-risk individuals can enable timely intervention. 
            This project explores whether machine learning classification models can 
            estimate diabetes risk from routinely collected health indicators, providing 
            an accessible decision-support tool for educational demonstration.
            """
        ),
        (
            '2. Objective',
            """
            To build a professional, technically correct, and easy-to-demonstrate 
            machine-learning application that:
            <ul style="margin-top:8px; line-height:1.9; color:#E2E8F0;">
                <li>Trains and compares multiple classification algorithms</li>
                <li>Selects the best model using principled evaluation criteria</li>
                <li>Provides an interactive Streamlit interface for risk estimation</li>
                <li>Clearly communicates that output is a statistical prediction, not a diagnosis</li>
            </ul>
            """
        ),
        (
            '3. Dataset',
            """
            <strong>Name:</strong> Diabetes Prediction Dataset<br>
            <strong>Source:</strong> Kaggle — iammustafatz<br>
            <strong>URL:</strong> https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset<br>
            <strong>License:</strong> CC0 Public Domain (free to use, no restrictions)<br>
            <strong>Records:</strong> ~100,000 rows (after deduplication: ~96,000)<br>
            <strong>Gender:</strong> Male, Female, Other — genuine feature, not fabricated<br>
            <strong>Why not PIMA?</strong> The PIMA Indians Diabetes Dataset is <em>female-only</em>. 
            This project uses a dataset with <em>both male and female patients</em>.
            """
        ),
        (
            '4. Features',
            """
            <table style="width:100%; font-size:13px; border-collapse:collapse;">
            <tr style="background:#F1F5F9;">
                <th style="padding:8px; text-align:left; border:1px solid #E2E8F0;">Feature</th>
                <th style="padding:8px; text-align:left; border:1px solid #E2E8F0;">Type</th>
                <th style="padding:8px; text-align:left; border:1px solid #E2E8F0;">Description</th>
            </tr>
            <tr><td style="padding:8px; border:1px solid #E2E8F0;">gender</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Categorical</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Male / Female / Other</td></tr>
            <tr style="background:#F8FAFC;"><td style="padding:8px; border:1px solid #E2E8F0;">age</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Numerical</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Patient age in years (0–80)</td></tr>
            <tr><td style="padding:8px; border:1px solid #E2E8F0;">hypertension</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Binary (0/1)</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Has hypertension</td></tr>
            <tr style="background:#F8FAFC;"><td style="padding:8px; border:1px solid #E2E8F0;">heart_disease</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Binary (0/1)</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Has heart disease</td></tr>
            <tr><td style="padding:8px; border:1px solid #E2E8F0;">smoking_history</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Categorical</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">never / current / former / ever / not current / No Info</td></tr>
            <tr style="background:#F8FAFC;"><td style="padding:8px; border:1px solid #E2E8F0;">bmi</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Numerical</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Body Mass Index</td></tr>
            <tr><td style="padding:8px; border:1px solid #E2E8F0;">HbA1c_level</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Numerical</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Haemoglobin A1c level (%)</td></tr>
            <tr style="background:#F8FAFC;"><td style="padding:8px; border:1px solid #E2E8F0;">blood_glucose_level</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Numerical</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Blood glucose (mg/dL)</td></tr>
            <tr><td style="padding:8px; border:1px solid #E2E8F0;"><strong>diabetes</strong></td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Binary Target</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">0 = No Diabetes, 1 = Diabetes</td></tr>
            </table>
            """
        ),
        (
            '5. Data Preprocessing',
            """
            <ul style="line-height:1.9; color:#E2E8F0; margin-top:4px;">
                <li><strong>Duplicate removal:</strong> ~3,854 duplicate rows removed before splitting</li>
                <li><strong>Train/Test split:</strong> 80/20, stratified on the target class to preserve class balance</li>
                <li><strong>No data leakage:</strong> The preprocessing pipeline is fitted ONLY on training data, then applied to test data</li>
                <li><strong>Numerical features:</strong> SimpleImputer (median) → StandardScaler</li>
                <li><strong>Binary features:</strong> Passed through as-is (0/1 integers)</li>
                <li><strong>Categorical features:</strong> SimpleImputer (most frequent) → OrdinalEncoder with explicit categories</li>
                <li><strong>Pipeline:</strong> sklearn ColumnTransformer ensures identical transforms during prediction</li>
            </ul>
            """
        ),
        (
            '6. Machine Learning Algorithms',
            """
            Three algorithms were trained and evaluated:
            <ol style="line-height:1.9; color:#E2E8F0; margin-top:6px;">
                <li><strong>Logistic Regression</strong> — Linear model, interpretable coefficients, fast training</li>
                <li><strong>Random Forest</strong> — Ensemble of decision trees, feature importance available</li>
                <li><strong>Support Vector Machine (SVM)</strong> — Kernel-based classifier with RBF kernel</li>
            </ol>
            All models were trained on the same preprocessed training data and evaluated on the same held-out test set.
            """
        ),
        (
            '7. Model Selection',
            """
            <strong>Selection Metric: ROC-AUC</strong><br><br>
            The dataset is <em>class-imbalanced</em> (~91.5% negative, ~8.5% positive). 
            In such settings, accuracy can be misleading — a model predicting "No Diabetes" for everyone 
            would achieve ~91.5% accuracy while being useless. 
            <br><br>
            <strong>ROC-AUC</strong> measures the model's ability to distinguish between classes regardless 
            of the decision threshold, making it the appropriate primary metric here. 
            The model with the highest ROC-AUC on the test set was selected as the final model.
            """
        ),
        (
            '8. Evaluation Metrics',
            """
            <ul style="line-height:1.9; color:#E2E8F0; margin-top:4px;">
                <li><strong>Accuracy:</strong> Overall proportion of correct predictions</li>
                <li><strong>Precision:</strong> Of all predicted positives, how many are actually positive</li>
                <li><strong>Recall:</strong> Of all actual positives, how many were correctly identified</li>
                <li><strong>F1-Score:</strong> Harmonic mean of Precision and Recall</li>
                <li><strong>ROC-AUC:</strong> Area under the Receiver Operating Characteristic curve (discrimination ability)</li>
                <li><strong>Confusion Matrix:</strong> Table of TP, TN, FP, FN counts</li>
            </ul>
            All metrics are computed on the 20% held-out test set. No metrics are fabricated.
            """
        ),
        (
            '9. Application Workflow',
            """
            <ol style="line-height:1.9; color:#E2E8F0; margin-top:4px;">
                <li>Patient enters health information in the Risk Assessment form</li>
                <li>Input is validated for range and completeness</li>
                <li>Input is converted into the same feature structure used during training</li>
                <li>The saved preprocessing pipeline is applied (same transforms as training)</li>
                <li>The saved model generates a probability of diabetes (class 1)</li>
                <li>Probability is categorised: Low (&lt;30%) / Moderate (30–60%) / Higher (≥60%)</li>
                <li>Result is displayed with gauge, summary table, and feature influence</li>
            </ol>
            The model is <strong>never retrained</strong> during application use.
            """
        ),
        (
            '10. Limitations',
            """
            <ul style="line-height:1.9; color:#E2E8F0; margin-top:4px;">
                <li><strong>Dataset origin:</strong> The dataset source and collection methodology are not 
                fully documented, which limits understanding of the target population</li>
                <li><strong>Class imbalance:</strong> ~8.5% positive class — performance on the minority class 
                (diabetes positive) is inherently harder to optimise</li>
                <li><strong>No external validation:</strong> The model has only been evaluated on the same 
                dataset it was trained on; generalization to other populations is unknown</li>
                <li><strong>Limited clinical features:</strong> Only 8 features are available — clinical 
                diagnosis involves many more factors</li>
                <li><strong>Not a diagnostic tool:</strong> Model output is a statistical estimate, 
                not a clinical measurement or medical diagnosis</li>
            </ul>
            """
        ),
        (
            '11. Future Enhancements',
            """
            <ul style="line-height:1.9; color:#E2E8F0; margin-top:4px;">
                <li>Use larger and more diverse datasets with documented collection protocols</li>
                <li>External validation on independent clinical datasets</li>
                <li>Additional health features (e.g., family history, physical activity, diet)</li>
                <li>Advanced model explainability using SHAP values</li>
                <li>Hyperparameter optimisation using GridSearchCV or Optuna</li>
                <li>Clinical validation in collaboration with healthcare professionals</li>
                <li>Secure cloud deployment with data privacy measures</li>
            </ul>
            """
        ),
        (
            '12. Technology Stack',
            """
            <table style="width:100%; font-size:13px; border-collapse:collapse;">
            <tr style="background:#F1F5F9;">
                <th style="padding:8px; text-align:left; border:1px solid #E2E8F0;">Library</th>
                <th style="padding:8px; text-align:left; border:1px solid #E2E8F0;">Purpose</th>
            </tr>
            <tr><td style="padding:8px; border:1px solid #E2E8F0;">Python 3.9+</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Core programming language</td></tr>
            <tr style="background:#F8FAFC;"><td style="padding:8px; border:1px solid #E2E8F0;">Streamlit</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Interactive web application</td></tr>
            <tr><td style="padding:8px; border:1px solid #E2E8F0;">Pandas</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Data loading and manipulation</td></tr>
            <tr style="background:#F8FAFC;"><td style="padding:8px; border:1px solid #E2E8F0;">NumPy</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Numerical computation</td></tr>
            <tr><td style="padding:8px; border:1px solid #E2E8F0;">Scikit-learn</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">ML models, preprocessing, evaluation</td></tr>
            <tr style="background:#F8FAFC;"><td style="padding:8px; border:1px solid #E2E8F0;">Joblib</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Model serialisation</td></tr>
            <tr><td style="padding:8px; border:1px solid #E2E8F0;">Plotly</td>
                <td style="padding:8px; border:1px solid #E2E8F0;">Interactive charts and visualisations</td></tr>
            </table>
            """
        ),
    ]

    for title, content in sections:
        with st.expander(title, expanded=False):
            st.markdown(f'<div style="color:#E2E8F0; font-size:14px; line-height:1.75;">{content}</div>',
                        unsafe_allow_html=True)

    # ── Project credits ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer-box" style="margin-top:24px;">
    ⚠️ <strong>Medical Disclaimer:</strong> DiaCare AI is an educational machine-learning risk 
    assessment project developed as a B.Tech Data Science demonstration. 
    The application's output is a statistical prediction generated by a trained ML model 
    and is <strong>not a medical diagnosis</strong>, clinical assessment, or substitute for 
    professional medical advice. Always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:20px; color:#94A3B8; font-size:12px;">
        DiaCare AI — B.Tech Data Science Project<br>
        Dataset: Diabetes Prediction Dataset (Kaggle CC0 Public Domain)<br>
        Built with Python · Streamlit · Scikit-learn · Plotly
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def sidebar_navigation() -> str:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:16px 0 20px 0;">
            <div style="font-size:32px;">🩺</div>
            <div style="font-size:18px; font-weight:700; color:white; margin-top:6px;">DiaCare AI</div>
            <div style="font-size:11px; color:rgba(255,255,255,0.7); margin-top:2px;">
                Diabetes Risk Assessment
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('---')

        pages = [
            '🏠 Home',
            '🩺 Risk Assessment',
            '📊 Analytics',
            '📈 Model Performance',
            'ℹ️ About Project',
        ]

        # Initialise session state
        if 'page' not in st.session_state:
            st.session_state['page'] = '🏠 Home'

        # If page was set programmatically (e.g., from Home button)
        current_idx = 0
        for i, p in enumerate(pages):
            if st.session_state['page'] in p or p in st.session_state['page']:
                current_idx = i
                break

        selected = st.radio(
            'Navigation',
            pages,
            index=current_idx,
            label_visibility='collapsed',
        )
        st.session_state['page'] = selected

        st.markdown('---')

        # Model status indicator
        model_ready = MODEL_PATH.exists() and PREPROCESSING_PATH.exists()
        if model_ready:
            st.markdown("""
            <div style="background:rgba(16,185,129,0.2); border:1px solid rgba(16,185,129,0.4);
                        border-radius:8px; padding:10px 12px; font-size:12px;">
                <span style="color:#6EE7B7;">✓ Model Ready</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(239,68,68,0.2); border:1px solid rgba(239,68,68,0.4);
                        border-radius:8px; padding:10px 12px; font-size:12px;">
                <span style="color:#FCA5A5;">⚠ Run train_model.py</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:10px; color:rgba(255,255,255,0.5); text-align:center; line-height:1.6;">
            B.Tech Data Science Project<br>
            Educational Use Only<br>
            Not a Medical Device
        </div>
        """, unsafe_allow_html=True)

    return selected


# ---------------------------------------------------------------------------
# ── MAIN ──────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    page = sidebar_navigation()

    if 'Home' in page:
        page_home()
    elif 'Risk Assessment' in page:
        page_risk_assessment()
    elif 'Analytics' in page:
        page_analytics()
    elif 'Model Performance' in page:
        page_model_performance()
    elif 'About' in page:
        page_about()


if __name__ == '__main__':
    main()
