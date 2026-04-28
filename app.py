import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
)
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from gemini_explainer import explain_bias, generate_fairness_report, suggest_mitigation

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FairCheck AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# HELPER FUNCTIONS (Must be defined before use)
# ============================================================

def risk_label(value: float) -> str:
    v = abs(value)
    if v < 0.10:
        return "LOW"
    elif v < 0.20:
        return "MEDIUM"
    return "HIGH"

def risk_color(label: str) -> str:
    return {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}[label]

def is_binary_classification(series: pd.Series) -> bool:
    """Check if a column is binary classification"""
    unique_values = series.dropna().unique()
    return len(unique_values) == 2

def is_regression(series: pd.Series) -> bool:
    """Check if a column is regression (numeric with many unique values)"""
    if pd.api.types.is_numeric_dtype(series):
        unique_values = series.dropna().unique()
        return len(unique_values) > 10  # More than 10 unique values suggests regression
    return False

def binary_encode(series: pd.Series):
    series = series.astype(str).str.strip()
    values = sorted(series.unique())
    if len(values) != 2:
        raise ValueError(f"Column must have exactly 2 unique values. Found: {values[:10]}")
    mapping = {values[0]: 0, values[1]: 1}
    return series.map(mapping), mapping, values[1]

def analyze_regression_bias(df, target_col, sensitive_col):
    """Bias analysis for regression tasks"""
    
    temp = pd.DataFrame({"group": df[sensitive_col], "target": df[target_col]})
    group_stats = (
        temp.groupby("group")
        .agg(
            Total=("target", "count"),
            Mean_Value=("target", "mean"),
            Median_Value=("target", "median"),
            Std_Value=("target", "std"),
            Min_Value=("target", "min"),
            Max_Value=("target", "max"),
        )
        .reset_index()
    )
    
    # Calculate group disparities
    highest = group_stats.loc[group_stats["Mean_Value"].idxmax()]
    lowest = group_stats.loc[group_stats["Mean_Value"].idxmin()]
    gap = abs(highest["Mean_Value"] - lowest["Mean_Value"])
    relative_gap = gap / (group_stats["Mean_Value"].mean() + 1e-6)  # Relative difference
    risk = risk_label(relative_gap)
    
    return group_stats, highest, lowest, gap, relative_gap, risk

def analyze_model_regression_bias(y_true, y_pred, sensitive):
    """Analyze bias for regression model predictions"""
    
    # Create dataframe for analysis
    df_analysis = pd.DataFrame({
        'true': y_true,
        'pred': y_pred,
        'sensitive': sensitive
    })
    
    # Calculate metrics per group
    group_metrics = df_analysis.groupby('sensitive').agg({
        'true': ['mean', 'std', 'count'],
        'pred': ['mean', 'std']
    }).round(4)
    
    # Calculate error metrics per group
    df_analysis['error'] = abs(df_analysis['true'] - df_analysis['pred'])
    df_analysis['squared_error'] = (df_analysis['true'] - df_analysis['pred']) ** 2
    
    group_errors = df_analysis.groupby('sensitive').agg({
        'error': 'mean',
        'squared_error': 'mean'
    }).round(4)
    
    group_metrics.columns = ['True_Mean', 'True_Std', 'Count', 'Pred_Mean', 'Pred_Std']
    group_metrics['MAE'] = group_errors['error']
    group_metrics['RMSE'] = np.sqrt(group_errors['squared_error'])
    
    # Calculate overall metrics
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    overall_r2 = r2_score(y_true, y_pred)
    
    # Calculate disparity (max difference in MAE between groups)
    mae_by_group = group_metrics['MAE'].values
    disparity = mae_by_group.max() - mae_by_group.min()
    relative_disparity = disparity / (overall_mae + 1e-6)
    risk = risk_label(relative_disparity)
    
    return group_metrics, overall_mae, overall_rmse, overall_r2, disparity, relative_disparity, risk

def create_regression_visualizations(df, target_col, sensitive_col, group_stats):
    """Create visualizations for regression bias"""
    
    plot_df = pd.DataFrame({
        'sensitive_group': df[sensitive_col],
        'target': df[target_col]
    })
    
    # 1. Box plots for each group
    fig1 = go.Figure()
    groups_unique = df[sensitive_col].unique()
    
    for group in groups_unique:
        group_data = df[df[sensitive_col] == group][target_col].dropna()
        fig1.add_trace(go.Box(
            y=group_data,
            name=str(group),
            boxmean='sd',
            marker_color='#f5c842'
        ))
    
    fig1.update_layout(
        title=f"Target Variable Distribution Across Groups",
        xaxis_title="Sensitive Group",
        yaxis_title=target_col,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,19,28,0.5)',
        font=dict(color='#e8e6e0'),
        showlegend=False
    )
    
    # 2. Mean values bar chart
    fig2 = go.Figure()
    colors = ['#4caf50' if abs(rate - group_stats['Mean_Value'].mean()) < group_stats['Mean_Value'].std()/2 
              else '#ff9800' if abs(rate - group_stats['Mean_Value'].mean()) < group_stats['Mean_Value'].std() 
              else '#f44336' 
              for rate in group_stats['Mean_Value']]
    
    fig2.add_trace(go.Bar(
        x=group_stats['group'],
        y=group_stats['Mean_Value'],
        marker_color=colors,
        error_y=dict(type='data', array=group_stats['Std_Value'], visible=True),
        text=[f"{v:.2f}" for v in group_stats['Mean_Value']],
        textposition='outside'
    ))
    
    fig2.add_hline(y=group_stats['Mean_Value'].mean(), line_dash="dash", 
                   line_color="#f5c842", 
                   annotation_text=f"Overall Mean: {group_stats['Mean_Value'].mean():.2f}")
    
    fig2.update_layout(
        title="Mean Values by Group (with Standard Deviation)",
        xaxis_title="Sensitive Group",
        yaxis_title=f"Mean {target_col}",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,19,28,0.5)',
        font=dict(color='#e8e6e0')
    )
    
    # 3. Violin plot for distribution comparison
    fig3 = go.Figure()
    for group in groups_unique:
        group_data = df[df[sensitive_col] == group][target_col].dropna()
        fig3.add_trace(go.Violin(
            y=group_data,
            name=str(group),
            box_visible=True,
            meanline_visible=True,
            fillcolor='rgba(245, 200, 66, 0.4)',
            line_color='#f5c842'
        ))
    
    fig3.update_layout(
        title=f"Distribution Density (Violin Plot) by Group",
        xaxis_title="Sensitive Group",
        yaxis_title=target_col,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,19,28,0.5)',
        font=dict(color='#e8e6e0'),
        showlegend=False
    )
    
    return fig1, fig2, fig3

def create_model_regression_visualizations(y_true, y_pred, sensitive):
    """Create visualizations for regression model bias"""
    
    df_analysis = pd.DataFrame({
        'true': y_true,
        'pred': y_pred,
        'sensitive': sensitive,
        'error': abs(y_true - y_pred),
        'residual': y_true - y_pred
    })
    
    groups = df_analysis['sensitive'].unique()
    
    # 1. Actual vs Predicted scatter plot
    fig1 = go.Figure()
    for group in groups:
        group_data = df_analysis[df_analysis['sensitive'] == group]
        fig1.add_trace(go.Scatter(
            x=group_data['true'],
            y=group_data['pred'],
            mode='markers',
            name=str(group),
            marker=dict(size=8, opacity=0.6),
            text=[f"Group: {group}<br>True: {t:.2f}<br>Pred: {p:.2f}" for t, p in zip(group_data['true'], group_data['pred'])],
            hoverinfo='text'
        ))
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig1.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='#f5c842', width=2)
    ))
    
    fig1.update_layout(
        title="Actual vs Predicted Values by Group",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,19,28,0.5)',
        font=dict(color='#e8e6e0')
    )
    
    # 2. Error distribution by group (box plot)
    fig2 = go.Figure()
    for group in groups:
        group_errors = df_analysis[df_analysis['sensitive'] == group]['error']
        fig2.add_trace(go.Box(
            y=group_errors,
            name=str(group),
            boxmean='sd',
            marker_color='#f5c842'
        ))
    
    fig2.update_layout(
        title="Prediction Error Distribution by Group",
        xaxis_title="Sensitive Group",
        yaxis_title="Absolute Error",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,19,28,0.5)',
        font=dict(color='#e8e6e0')
    )
    
    # 3. Residual plot
    fig3 = go.Figure()
    for group in groups:
        group_data = df_analysis[df_analysis['sensitive'] == group]
        fig3.add_trace(go.Scatter(
            x=group_data['pred'],
            y=group_data['residual'],
            mode='markers',
            name=str(group),
            marker=dict(size=8, opacity=0.6),
            text=[f"Group: {group}<br>Pred: {p:.2f}<br>Residual: {r:.2f}" for p, r in zip(group_data['pred'], group_data['residual'])],
            hoverinfo='text'
        ))
    
    fig3.add_hline(y=0, line_dash="dash", line_color="#f5c842")
    fig3.update_layout(
        title="Residual Plot by Group",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals (Actual - Predicted)",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,19,28,0.5)',
        font=dict(color='#e8e6e0')
    )
    
    # 4. MAE comparison bar chart
    mae_by_group = df_analysis.groupby('sensitive')['error'].mean().reset_index()
    fig4 = go.Figure()
    colors = ['#4caf50' if mae < mae_by_group['error'].mean() 
              else '#ff9800' if mae < mae_by_group['error'].mean() * 1.5 
              else '#f44336' 
              for mae in mae_by_group['error']]
    
    fig4.add_trace(go.Bar(
        x=mae_by_group['sensitive'],
        y=mae_by_group['error'],
        marker_color=colors,
        text=[f"{v:.3f}" for v in mae_by_group['error']],
        textposition='outside'
    ))
    
    fig4.add_hline(y=mae_by_group['error'].mean(), line_dash="dash", 
                   line_color="#f5c842", 
                   annotation_text=f"Overall MAE: {mae_by_group['error'].mean():.3f}")
    
    fig4.update_layout(
        title="Mean Absolute Error (MAE) by Group",
        xaxis_title="Sensitive Group",
        yaxis_title="MAE",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,19,28,0.5)',
        font=dict(color='#e8e6e0')
    )
    
    return fig1, fig2, fig3, fig4

# ============================================================
# CUSTOM CSS — Dark judicial aesthetic
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e6e0;
}

.main { background-color: #0a0a0f; }

h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem !important;
    font-weight: 900 !important;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #f5c842 0%, #e8a020 50%, #c97b00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0 !important;
}

h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #f5c842;
    border-bottom: 1px solid #2a2a35;
    padding-bottom: 0.4rem;
}

.stMetric {
    background: #13131c;
    border: 1px solid #2a2a35;
    border-left: 3px solid #f5c842;
    border-radius: 6px;
    padding: 1rem 1.2rem;
}

.stMetric label { color: #888 !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }
.stMetric [data-testid="metric-container"] > div:nth-child(2) { color: #f5c842 !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; }

.stDataFrame { border: 1px solid #2a2a35; border-radius: 6px; }

.stButton > button {
    background: linear-gradient(135deg, #f5c842, #c97b00);
    color: #0a0a0f;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1.5rem;
    letter-spacing: 0.05em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.stSelectbox > div > div {
    background: #13131c;
    border: 1px solid #2a2a35;
    color: #e8e6e0;
}

.stFileUploader {
    background: #13131c;
    border: 1px dashed #2a2a35;
    border-radius: 8px;
}

.risk-low    { color: #4caf50; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.risk-medium { color: #ff9800; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.risk-high   { color: #f44336; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }

.hero-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    color: #555;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0;
}

.info-card {
    background: #13131c;
    border: 1px solid #2a2a35;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
}

.step-badge {
    display: inline-block;
    background: #f5c842;
    color: #0a0a0f;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 2px;
    margin-right: 8px;
    letter-spacing: 0.1em;
}

div[data-testid="stExpander"] {
    background: #13131c;
    border: 1px solid #2a2a35;
    border-radius: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<h1>⚖️ FairCheck AI</h1>
<p class="hero-subtitle">Bias Detection & Fairness Evaluation Toolkit · Powered by Gemini AI</p>
""",
    unsafe_allow_html=True,
)

# API key check
api_ok = bool(os.getenv("GEMINI_API_KEY"))
if api_ok:
    st.success("✅ Gemini API connected", icon="🤖")
else:
    st.error("❌ GEMINI_API_KEY not found. Add it to your .env file.")

st.divider()

# ============================================================
# SIDEBAR — ABOUT
# ============================================================

with st.sidebar:
    st.markdown("### ⚖️ FairCheck AI")
    st.markdown(
        """
        **Hack2Skill Hackathon Project**  
        
        Audit your ML models and datasets for bias across sensitive attributes like gender, race, or age.
        
        **Supported Tasks:**
        - ✅ Binary Classification
        - ✅ Regression (Continuous Values)
        
        ---
        **Pipeline:**
        1. Upload Dataset
        2. Dataset Bias Analysis
        3. AI Explanation (Gemini)
        4. Upload Model Predictions
        5. Model Fairness Metrics
        6. What-If Simulator
        
        ---
        **Fairness Thresholds:**
        - 🟢 LOW: < 10% disparity
        - 🟡 MEDIUM: 10–20% disparity
        - 🔴 HIGH: > 20% disparity
        """
    )

# ============================================================
# STEP 1 — UPLOAD DATASET
# ============================================================

st.markdown('<span class="step-badge">STEP 01</span> **Upload Dataset**', unsafe_allow_html=True)

dataset_file = st.file_uploader(
    "Upload your dataset CSV", type=["csv"], key="dataset_file"
)

if dataset_file is None:
    st.markdown(
        '<div class="info-card">⬆️ Upload a CSV to begin. The file should contain a <b>target column</b> (outcome) and at least one <b>sensitive attribute</b> (e.g. gender, race, age group).</div>',
        unsafe_allow_html=True,
    )
    st.stop()

df = pd.read_csv(dataset_file)

st.markdown(f"**{len(df):,} rows · {len(df.columns):,} columns**")
st.dataframe(df.head(8), use_container_width=True)

st.divider()

# ============================================================
# STEP 2 — COLUMN SELECTION
# ============================================================

st.markdown('<span class="step-badge">STEP 02</span> **Select Columns**', unsafe_allow_html=True)

cols = df.columns.tolist()
c1, c2 = st.columns(2)
with c1:
    target_col = st.selectbox("🎯 Target Column (outcome)", cols)
with c2:
    sensitive_col = st.selectbox(
        "🔒 Sensitive Attribute Column",
        [c for c in cols if c != target_col],
    )

# Detect task type
unique_target_values = df[target_col].nunique()
if is_binary_classification(df[target_col]):
    task_type = "Binary Classification"
    st.info(f"📊 Detected Task: **{task_type}** (Target has {unique_target_values} unique values)")
elif is_regression(df[target_col]):
    task_type = "Regression"
    st.info(f"📈 Detected Task: **{task_type}** (Target has {unique_target_values} unique numeric values)")
else:
    task_type = "Multi-class Classification"
    st.warning(f"⚠️ Detected Task: **Multi-class Classification** (Target has {unique_target_values} unique values). FairCheck AI works best with binary classification or regression tasks.")

# Quick preview
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Target distribution:**")
    if unique_target_values <= 10:
        st.dataframe(df[target_col].value_counts().head(10).rename("count").reset_index(), use_container_width=True)
    else:
        st.markdown(f"*Target has {unique_target_values} unique values (regression task)*")
        st.dataframe(df[target_col].describe(), use_container_width=True)
        
with c2:
    st.markdown(f"**Sensitive attribute distribution:**")
    st.dataframe(df[sensitive_col].value_counts().rename("count").reset_index(), use_container_width=True)

st.divider()

# ============================================================
# STEP 3 — DATASET BIAS ANALYSIS
# ============================================================

st.markdown('<span class="step-badge">STEP 03</span> **Dataset Bias Analysis**', unsafe_allow_html=True)

if st.button("🔍 Analyse Dataset Bias", use_container_width=True):

    with st.spinner("Computing group statistics..."):
        
        if task_type == "Binary Classification" and is_binary_classification(df[target_col]):
            # Classification analysis
            y_true, mapping, positive_label = binary_encode(df[target_col])
            temp = pd.DataFrame({"group": df[sensitive_col], "target": y_true})
            group_stats = (
                temp.groupby("group")
                .agg(
                    Total=("target", "count"),
                    Positives=("target", "sum"),
                    Positive_Rate=("target", "mean"),
                )
                .reset_index()
            )
            group_stats["Positive_Rate_%"] = (group_stats["Positive_Rate"] * 100).round(2)

            highest = group_stats.loc[group_stats["Positive_Rate"].idxmax()]
            lowest = group_stats.loc[group_stats["Positive_Rate"].idxmin()]
            gap = highest["Positive_Rate"] - lowest["Positive_Rate"]
            relative_gap = gap
            risk = risk_label(gap)
            css_class = risk_color(risk)

            st.subheader("Group Statistics (Classification)")
            st.dataframe(group_stats.drop(columns="Positive_Rate"), use_container_width=True)
            
            # Create classification visualizations
            from classification_viz import create_classification_visualizations
            fig1, fig2, fig3, fig4, fig5 = create_classification_visualizations(df, target_col, sensitive_col, group_stats)
            
            # Display classification visualizations
            st.subheader("📊 Bias Visualization Dashboard")
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                st.plotly_chart(fig1, use_container_width=True, key="viz1")
                st.plotly_chart(fig3, use_container_width=True, key="viz3")
            with viz_col2:
                st.plotly_chart(fig2, use_container_width=True, key="viz2")
                st.plotly_chart(fig4, use_container_width=True, key="viz4")
            if fig5:
                st.plotly_chart(fig5, use_container_width=True, key="viz5")
            
        else:
            # Regression analysis
            group_stats, highest, lowest, gap, relative_gap, risk = analyze_regression_bias(df, target_col, sensitive_col)
            css_class = risk_color(risk)
            
            st.subheader("Group Statistics (Regression)")
            st.dataframe(group_stats.round(3), use_container_width=True)
            
            # Create regression visualizations
            fig1, fig2, fig3 = create_regression_visualizations(df, target_col, sensitive_col, group_stats)
            
            # Display regression visualizations
            st.subheader("📊 Bias Visualization Dashboard")
            st.plotly_chart(fig1, use_container_width=True, key="viz1")
            st.plotly_chart(fig2, use_container_width=True, key="viz2")
            st.plotly_chart(fig3, use_container_width=True, key="viz3")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Highest Rate/Mean Group", highest["group"])
        m2.metric("Lowest Rate/Mean Group", lowest["group"])
        m3.metric("Absolute Gap", f"{gap:.4f}")
        m4.metric("Bias Risk Level", risk)

        st.markdown(
            f'<p>Bias Risk: <span class="{css_class}">{risk}</span></p>',
            unsafe_allow_html=True,
        )
        
        # Feature distribution analysis
        st.subheader("📈 Feature Distribution Analysis")
        st.caption("How features are distributed across sensitive groups")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != target_col]
        
        if len(numeric_cols) > 0:
            selected_feature = st.selectbox("Select a numeric feature to analyze across groups", numeric_cols, key="feature_dist")
            
            fig6 = go.Figure()
            groups_unique = df[sensitive_col].unique()
            
            for group in groups_unique:
                group_data = df[df[sensitive_col] == group][selected_feature].dropna()
                fig6.add_trace(go.Box(
                    y=group_data,
                    name=str(group),
                    boxmean='sd',
                    marker_color='#f5c842'
                ))
            
            fig6.update_layout(
                title=f"Distribution of '{selected_feature}' Across {sensitive_col} Groups",
                xaxis_title="Sensitive Group",
                yaxis_title=selected_feature,
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(19,19,28,0.5)',
                font=dict(color='#e8e6e0'),
                showlegend=False
            )
            
            st.plotly_chart(fig6, use_container_width=True, key="viz6")
            
            # Statistical summary
            st.markdown("**Statistical Summary by Group:**")
            summary_stats = df.groupby(sensitive_col)[selected_feature].agg(['count', 'mean', 'std', 'min', 'max'])
            st.dataframe(summary_stats.round(3), use_container_width=True)
        
        # Bias risk gauge
        st.subheader("🎯 Bias Risk Summary")
        fig7 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=relative_gap * 100,
            title={'text': "Dataset Bias Score (%)", 'font': {'color': '#e8e6e0'}},
            delta={'reference': 10, 'increasing': {'color': "#f44336"}, 'decreasing': {'color': "#4caf50"}},
            gauge={
                'axis': {'range': [None, 30], 'tickwidth': 1, 'tickcolor': "#e8e6e0"},
                'bar': {'color': "#f5c842"},
                'bgcolor': "#13131c",
                'borderwidth': 2,
                'bordercolor': "#2a2a35",
                'steps': [
                    {'range': [0, 10], 'color': "#1a3a1a"},
                    {'range': [10, 20], 'color': "#4a3a1a"},
                    {'range': [20, 30], 'color': "#3a1a1a"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': relative_gap * 100
                }
            }
        ))
        fig7.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color': "#e8e6e0"})
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.plotly_chart(fig7, use_container_width=True, key="viz7")

    # Store for AI buttons
    st.session_state["dataset_metrics_text"] = f"""
Task type: {task_type}
Sensitive attribute: {sensitive_col}
Target column: {target_col}
Groups analysed: {", ".join(group_stats["group"].astype(str).tolist())}
Highest rate/mean group: {highest['group']}
Lowest rate/mean group: {lowest['group']}
Absolute gap: {gap:.4f}
Relative gap: {relative_gap*100:.2f}%
Risk level: {risk}
"""

# ---- AI Explanation ----
if "dataset_metrics_text" in st.session_state:
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("🤖 Explain Bias (Gemini)", use_container_width=True):
            with st.spinner("Gemini analysing bias..."):
                explanation = explain_bias(st.session_state["dataset_metrics_text"])
            st.session_state["dataset_explanation"] = explanation

    with c2:
        if st.button("📄 Generate Audit Report", use_container_width=True):
            with st.spinner("Gemini generating report..."):
                report = generate_fairness_report(st.session_state["dataset_metrics_text"])
            st.session_state["dataset_report"] = report

    with c3:
        if st.button("💡 Suggest Mitigations", use_container_width=True):
            with st.spinner("Gemini suggesting mitigations..."):
                mitigations = suggest_mitigation(st.session_state["dataset_metrics_text"])
            st.session_state["dataset_mitigations"] = mitigations

    if "dataset_explanation" in st.session_state:
        with st.expander("🤖 Gemini Bias Explanation", expanded=True):
            st.markdown(st.session_state["dataset_explanation"])

    if "dataset_mitigations" in st.session_state:
        with st.expander("💡 Mitigation Strategies", expanded=True):
            st.markdown(st.session_state["dataset_mitigations"])

    if "dataset_report" in st.session_state:
        with st.expander("📄 Full Audit Report", expanded=True):
            st.markdown(st.session_state["dataset_report"])
            st.download_button(
                "⬇️ Download Report (.txt)",
                data=st.session_state["dataset_report"],
                file_name="faircheck_dataset_audit.txt",
                mime="text/plain",
            )

st.divider()

# ============================================================
# STEP 4 — MODEL PREDICTION BIAS (Now with Regression Support!)
# ============================================================

st.markdown('<span class="step-badge">STEP 04</span> **Model Prediction Bias**', unsafe_allow_html=True)
st.caption(
    "Upload a CSV with columns: true labels, model predictions, and a sensitive attribute."
)

pred_file = st.file_uploader("Upload predictions CSV", type=["csv"], key="pred_file")

if pred_file is not None:

    pred_df = pd.read_csv(pred_file)
    st.markdown(f"**{len(pred_df):,} rows · {len(pred_df.columns):,} columns**")
    st.dataframe(pred_df.head(6), use_container_width=True)

    pred_cols = pred_df.columns.tolist()
    
    # Auto-detect if this is regression or classification based on the true labels
    true_col = st.selectbox("✅ True Label Column", pred_cols, key="sel_true_col")
    
    # Detect task type for model predictions
    if is_binary_classification(pred_df[true_col]):
        model_task_type = "Binary Classification"
    elif is_regression(pred_df[true_col]):
        model_task_type = "Regression"
    else:
        model_task_type = "Unknown"
    
    st.info(f"📊 Detected Model Task: **{model_task_type}**")
    
    pred_col = st.selectbox("🔮 Prediction Column", pred_cols, key="sel_pred_col")
    sensitive_pred = st.selectbox("🔒 Sensitive Column", pred_cols, key="sel_sensitive_pred")

    if st.button("🔍 Analyse Model Bias", use_container_width=True):

        with st.spinner("Computing fairness metrics..."):
            
            if model_task_type == "Binary Classification":
                # Classification analysis
                y_true, _, _ = binary_encode(pred_df[true_col])
                y_pred, _, positive_label = binary_encode(pred_df[pred_col])
                sensitive = pred_df[sensitive_pred]

                metric_frame = MetricFrame(
                    metrics={
                        "Selection Rate": selection_rate,
                        "Accuracy": accuracy_score,
                        "TPR": true_positive_rate,
                        "FPR": false_positive_rate,
                        "FNR": false_negative_rate,
                    },
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive,
                )

                dp_diff = demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive
                )
                eo_diff = equalized_odds_difference(
                    y_true, y_pred, sensitive_features=sensitive
                )
                accuracy = accuracy_score(y_true, y_pred)
                risk = risk_label(max(abs(dp_diff), abs(eo_diff)))
                css_class = risk_color(risk)

                st.subheader("Per-Group Fairness Metrics (Classification)")
                group_report = metric_frame.by_group.reset_index()
                st.dataframe(group_report, use_container_width=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
                m2.metric("Demographic Parity Diff", f"{dp_diff:.4f}")
                m3.metric("Equalized Odds Diff", f"{eo_diff:.4f}")
                m4.metric("Model Bias Risk", risk)

                st.markdown(
                    f'<p>Model Bias Risk: <span class="{css_class}">{risk}</span></p>',
                    unsafe_allow_html=True,
                )
                
                model_metrics_text = f"""
Task type: Classification
Sensitive attribute: {sensitive_pred}
Accuracy: {accuracy * 100:.2f}%
Demographic Parity Difference: {dp_diff:.4f}
Equalized Odds Difference: {eo_diff:.4f}
Bias risk level: {risk}
Per-group metrics:
{group_report.to_string(index=False)}
"""
                
            else:
                # Regression analysis
                y_true = pred_df[true_col].values
                y_pred = pred_df[pred_col].values
                sensitive = pred_df[sensitive_pred]
                
                group_metrics, overall_mae, overall_rmse, overall_r2, disparity, relative_disparity, risk = analyze_model_regression_bias(y_true, y_pred, sensitive)
                css_class = risk_color(risk)
                
                st.subheader("Per-Group Fairness Metrics (Regression)")
                st.dataframe(group_metrics.round(4), use_container_width=True)
                
                # Create visualizations
                fig1, fig2, fig3, fig4 = create_model_regression_visualizations(y_true, y_pred, sensitive)
                
                st.subheader("📊 Model Bias Visualization Dashboard")
                st.plotly_chart(fig1, use_container_width=True, key="model_viz1")
                st.plotly_chart(fig2, use_container_width=True, key="model_viz2")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig3, use_container_width=True, key="model_viz3")
                with col2:
                    st.plotly_chart(fig4, use_container_width=True, key="model_viz4")
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Overall MAE", f"{overall_mae:.4f}")
                m2.metric("Overall RMSE", f"{overall_rmse:.4f}")
                m3.metric("R² Score", f"{overall_r2:.4f}")
                m4.metric("MAE Disparity", f"{disparity:.4f}")
                m5.metric("Model Bias Risk", risk)

                st.markdown(
                    f'<p>Model Bias Risk: <span class="{css_class}">{risk}</span></p>',
                    unsafe_allow_html=True,
                )
                
                model_metrics_text = f"""
Task type: Regression
Sensitive attribute: {sensitive_pred}
Overall MAE: {overall_mae:.4f}
Overall RMSE: {overall_rmse:.4f}
R² Score: {overall_r2:.4f}
MAE Disparity: {disparity:.4f}
Relative Disparity: {relative_disparity*100:.2f}%
Bias risk level: {risk}
Per-group metrics:
{group_metrics.to_string()}
"""

        st.session_state["model_metrics_text"] = model_metrics_text
        st.session_state["pred_df"] = pred_df
        st.session_state["sensitive_pred"] = sensitive_pred
        st.session_state["pred_col"] = pred_col
        st.session_state["model_task_type"] = model_task_type

    # ---- AI Buttons for Model ----
    if "model_metrics_text" in st.session_state:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🤖 Explain Model Bias (Gemini)", use_container_width=True):
                with st.spinner("Gemini analysing model fairness..."):
                    explanation = explain_bias(st.session_state["model_metrics_text"])
                st.session_state["model_explanation"] = explanation

        with c2:
            if st.button("💡 Model Mitigation Strategies", use_container_width=True):
                with st.spinner("Gemini generating strategies..."):
                    mitigations = suggest_mitigation(
                        st.session_state["model_metrics_text"],
                        context=f"This is a {st.session_state.get('model_task_type', 'regression')} model.",
                    )
                st.session_state["model_mitigations"] = mitigations

        if "model_explanation" in st.session_state:
            with st.expander("🤖 Gemini Model Bias Explanation", expanded=True):
                st.markdown(st.session_state["model_explanation"])

        if "model_mitigations" in st.session_state:
            with st.expander("💡 Model Mitigation Strategies", expanded=True):
                st.markdown(st.session_state["model_mitigations"])

st.divider()

# ============================================================
# STEP 5 — WHAT-IF BIAS SIMULATOR
# ============================================================

st.markdown('<span class="step-badge">STEP 05</span> **What-If Bias Simulator**', unsafe_allow_html=True)
st.caption(
    "Select a record and change its sensitive attribute to see how predictions change across groups."
)

if "pred_df" not in st.session_state:
    st.info("Complete Step 4 (Model Prediction Bias) to unlock the What-If Simulator.")
else:
    sim_df = st.session_state["pred_df"]
    sens_col = st.session_state["sensitive_pred"]
    p_col = st.session_state["pred_col"]
    true_col = st.session_state.get("sel_true_col", None)

    unique_groups = sorted(sim_df[sens_col].astype(str).unique())

    c1, c2 = st.columns(2)
    with c1:
        record_index = st.number_input(
            "Record index", min_value=0, max_value=len(sim_df) - 1, value=0, step=1
        )
    with c2:
        new_group = st.selectbox("Simulate: change group to", unique_groups)

    if st.button("⚡ Run Simulation", use_container_width=True):
        original_group = str(sim_df.loc[record_index, sens_col])
        original_pred = sim_df.loc[record_index, p_col]
        original_true = sim_df.loc[record_index, true_col] if true_col else "N/A"

        st.markdown("#### Simulation Result")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original Group", original_group)
        col2.metric("Simulated Group", new_group)
        col3.metric("True Value", original_true if isinstance(original_true, (int, float)) else str(original_true))
        col4.metric("Model Prediction", original_pred if isinstance(original_pred, (int, float)) else str(original_pred))

        if original_group != new_group:
            # Calculate group average predictions to show bias pattern
            group_avg_pred = sim_df.groupby(sens_col)[p_col].mean()
            st.info(f"📊 **Group Context:** The average prediction for '{original_group}' is {group_avg_pred[original_group]:.3f}, while for '{new_group}' it's {group_avg_pred[new_group]:.3f} (difference: {abs(group_avg_pred[original_group] - group_avg_pred[new_group]):.3f})")
            
            st.warning(
                "⚠️ **Potential Bias Indicator**: Changing the sensitive attribute would move this record to a different group. "
                "In a fair model, the sensitive attribute shouldn't directly determine the prediction."
            )
            
            # Ask Gemini to comment
            sim_text = f"""
Record index: {record_index}
Original group ({sens_col}): {original_group}
Simulated group: {new_group}
True value: {original_true}
Model prediction: {original_pred}
Average prediction for {original_group}: {group_avg_pred[original_group]:.3f}
Average prediction for {new_group}: {group_avg_pred[new_group]:.3f}
This is a what-if scenario to test if the model is sensitive to the protected attribute.
"""
            with st.spinner("Gemini assessing the simulation..."):
                sim_explanation = explain_bias(sim_text)
            st.subheader("🤖 Gemini Assessment")
            st.markdown(sim_explanation)
        else:
            st.success(
                "✅ No group change detected (same group selected). Try a different group to test counterfactual fairness."
            )

st.divider()
st.markdown(
    '<p style="text-align:center; color:#333; font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem;">FairCheck AI · Hack2Skill Hackathon · Powered by Gemini AI</p>',
    unsafe_allow_html=True,
)