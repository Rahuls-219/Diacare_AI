"""
visualization.py - Plotly chart builders for DiaCare AI.

All functions return a plotly.graph_objects.Figure ready to pass to
st.plotly_chart(fig, use_container_width=True).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

PRIMARY     = '#0891B2'    # cyan-600
SECONDARY   = '#0E7490'    # cyan-700
ACCENT      = '#06B6D4'    # cyan-500
LOW_COLOR   = '#10B981'    # green
MOD_COLOR   = '#F59E0B'    # amber
HIGH_COLOR  = '#EF4444'    # red
GRID_COLOR  = '#E2E8F0'
FONT_FAMILY = 'Inter, system-ui, sans-serif'

_BASE_LAYOUT = dict(
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(family=FONT_FAMILY, size=12, color='#334155'),
    margin=dict(l=20, r=20, t=55, b=20),
)


def _apply_base(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, **_BASE_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Risk gauge
# ---------------------------------------------------------------------------

def plot_risk_gauge(probability: float, risk_category: str) -> go.Figure:
    """Circular gauge chart showing estimated risk probability."""
    pct = probability * 100

    if probability < 0.30:
        bar_color = LOW_COLOR
    elif probability < 0.60:
        bar_color = MOD_COLOR
    else:
        bar_color = HIGH_COLOR

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=pct,
        number={
            'suffix': '%',
            'font': {'size': 42, 'color': '#1E293B', 'family': FONT_FAMILY},
            'valueformat': '.1f',
        },
        title={
            'text': (
                f"Estimated Risk<br>"
                f"<span style='font-size:15px;color:#64748B'>{risk_category}</span>"
            ),
            'font': {'size': 16, 'family': FONT_FAMILY},
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': '#94A3B8',
                'tickvals': [0, 30, 60, 100],
                'ticktext': ['0%', '30%', '60%', '100%'],
                'tickfont': {'size': 11},
            },
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30],  'color': '#D1FAE5'},
                {'range': [30, 60], 'color': '#FEF3C7'},
                {'range': [60, 100],'color': '#FEE2E2'},
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 3},
                'thickness': 0.78,
                'value': pct,
            },
        },
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor='white',
        font={'family': FONT_FAMILY},
    )
    return fig


# ---------------------------------------------------------------------------
# Analytics charts
# ---------------------------------------------------------------------------

def plot_outcome_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart of the diabetes outcome class distribution."""
    counts = df['diabetes'].value_counts()
    fig = go.Figure(go.Pie(
        labels=['No Diabetes (0)', 'Diabetes (1)'],
        values=[counts.get(0, 0), counts.get(1, 0)],
        hole=0.52,
        marker=dict(colors=[LOW_COLOR, HIGH_COLOR], line=dict(color='white', width=2)),
        textinfo='label+percent',
        textfont=dict(size=13),
        hovertemplate='%{label}<br>Count: %{value:,}<br>Share: %{percent}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Outcome Distribution', font=dict(size=15)),
        legend=dict(orientation='h', yanchor='bottom', y=-0.12),
    )
    return _apply_base(fig, 370)


def plot_gender_distribution(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of gender - diabetes outcome."""
    if 'gender' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text='Gender column not available', showarrow=False)
        return _apply_base(fig)

    gender_order = ['Male', 'Female', 'Other']
    data = []
    for outcome, label, color in [(0, 'No Diabetes', LOW_COLOR), (1, 'Diabetes', HIGH_COLOR)]:
        sub = df[df['diabetes'] == outcome]['gender'].value_counts()
        data.append(go.Bar(
            name=label,
            x=gender_order,
            y=[sub.get(g, 0) for g in gender_order],
            marker_color=color,
            text=[f'{sub.get(g, 0):,}' for g in gender_order],
            textposition='outside',
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='group',
        title=dict(text='Gender Distribution by Outcome', font=dict(size=15)),
        xaxis_title='Gender',
        yaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return _apply_base(fig, 400)


def plot_feature_distribution(df: pd.DataFrame, feature: str) -> go.Figure:
    """Overlaid histogram of a numerical feature split by outcome."""
    fig = go.Figure()
    for outcome, label, color, opacity in [
        (0, 'No Diabetes', LOW_COLOR, 0.65),
        (1, 'Diabetes',    HIGH_COLOR, 0.75),
    ]:
        subset = df[df['diabetes'] == outcome][feature].dropna()
        fig.add_trace(go.Histogram(
            x=subset,
            name=label,
            marker_color=color,
            opacity=opacity,
            nbinsx=35,
            hovertemplate=f'{label}<br>{feature}: %{{x}}<br>Count: %{{y}}<extra></extra>',
        ))

    fig.update_layout(
        barmode='overlay',
        title=dict(text=f'Distribution of {feature}', font=dict(size=15)),
        xaxis_title=feature,
        yaxis=dict(title='Count', gridcolor=GRID_COLOR),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return _apply_base(fig, 420)


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of all numerical features."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect='auto',
    )
    fig.update_layout(
        title=dict(text='Feature Correlation Heatmap', font=dict(size=15)),
        coloraxis_colorbar=dict(title='Corr.'),
    )
    return _apply_base(fig, 450)


# ---------------------------------------------------------------------------
# Model performance charts
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: list | np.ndarray) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    cm_arr = np.array(cm)
    labels = ['No Diabetes (0)', 'Diabetes (1)']
    annotations = [[str(cm_arr[i][j]) for j in range(2)] for i in range(2)]

    fig = px.imshow(
        cm_arr,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x='Predicted Label', y='Actual Label', color='Count'),
    )
    fig.update_traces(textfont=dict(size=16))
    fig.update_layout(
        title=dict(text='Confusion Matrix', font=dict(size=15)),
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
    )
    return _apply_base(fig, 380)


def plot_roc_curve(fpr: list | np.ndarray, tpr: list | np.ndarray, auc_score: float) -> go.Figure:
    """ROC curve with AUC annotation and random baseline."""
    fig = go.Figure()

    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='#94A3B8', dash='dash', width=1.5),
        name='Random Baseline (AUC = 0.50)',
    ))

    # Model ROC
    fig.add_trace(go.Scatter(
        x=list(fpr), y=list(tpr),
        mode='lines',
        line=dict(color=PRIMARY, width=2.5),
        fill='tozeroy',
        fillcolor='rgba(8,145,178,0.12)',
        name=f'Model (AUC = {auc_score:.4f})',
    ))

    fig.update_layout(
        title=dict(text='ROC Curve', font=dict(size=15)),
        xaxis=dict(title='False Positive Rate', gridcolor=GRID_COLOR, range=[0, 1]),
        yaxis=dict(title='True Positive Rate', gridcolor=GRID_COLOR, range=[0, 1.02]),
        legend=dict(x=0.55, y=0.06),
    )
    return _apply_base(fig, 420)


def plot_feature_importance(feature_names: list[str], importances: list[float],
                            model_type: str = 'Model') -> go.Figure:
    """Horizontal bar chart of feature influence scores."""
    # Sort ascending so highest bar is at top
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])
    names, values = zip(*pairs)

    # Colour scale: most important in primary colour
    max_v = max(values)
    colours = [PRIMARY if v == max_v else ACCENT if v > max_v * 0.6 else '#BAE6FD'
               for v in values]

    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(names),
        orientation='h',
        marker=dict(color=colours),
        text=[f'{v:.4f}' for v in values],
        textposition='outside',
        hovertemplate='%{y}: %{x:.4f}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text=f'Feature Influence - {model_type}',
            font=dict(size=15),
        ),
        xaxis=dict(title='Influence Score', gridcolor=GRID_COLOR),
        yaxis=dict(title=''),
    )
    return _apply_base(fig, 430)


def plot_model_comparison(results: dict) -> go.Figure:
    """Grouped bar chart comparing all trained models across five metrics."""
    metrics       = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    model_names   = list(results.keys())
    palette       = [PRIMARY, '#0E7490', '#0284C7', '#7C3AED', '#059669']

    fig = go.Figure()
    for i, model_name in enumerate(model_names):
        vals = [results[model_name].get(m, 0) for m in metrics]
        fig.add_trace(go.Bar(
            name=model_name,
            x=metric_labels,
            y=vals,
            marker_color=palette[i % len(palette)],
            text=[f'{v:.3f}' for v in vals],
            textposition='outside',
        ))

    fig.update_layout(
        barmode='group',
        title=dict(text='Model Performance Comparison', font=dict(size=15)),
        yaxis=dict(title='Score', range=[0, 1.12], gridcolor=GRID_COLOR),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return _apply_base(fig, 430)
