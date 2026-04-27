import streamlit as st
import pandas as pd
import numpy as np
import os
import io

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
)
from sklearn.metrics import accuracy_score
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
<p class="hero-subtitle">Bias Detection & Fairness Evaluation Toolkit · Powered by Claude AI</p>
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
        
        ---
        **Pipeline:**
        1. Upload Dataset
        2. Dataset Bias Analysis
        3. AI Explanation (Claude)
        4. Upload Model Predictions
        5. Model Fairness Metrics
        6. What-If Simulator
        
        ---
        **Fairness Thresholds:**
        - 🟢 LOW: < 10%
        - 🟡 MEDIUM: 10–20%
        - 🔴 HIGH: > 20%
        """
    )

# ============================================================
# HELPER FUNCTIONS
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


def binary_encode(series: pd.Series):
    series = series.astype(str).str.strip()
    values = sorted(series.unique())
    if len(values) != 2:
        st.error(
            f"Column must have exactly 2 unique values. Found: {values[:10]}"
        )
        st.stop()
    mapping = {values[0]: 0, values[1]: 1}
    return series.map(mapping), mapping, values[1]


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

st.markdown(f"**{len(df):,} rows · {len(df.columns)} columns**")
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

# Quick preview
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Target distribution:**")
    st.dataframe(df[target_col].value_counts().rename("count").reset_index(), use_container_width=True)
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
        risk = risk_label(gap)
        css_class = risk_color(risk)

    st.subheader("Group Statistics")
    st.dataframe(group_stats.drop(columns="Positive_Rate"), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Highest Rate Group", highest["group"])
    m2.metric("Lowest Rate Group", lowest["group"])
    m3.metric("Outcome Gap", f"{gap * 100:.2f}%")
    m4.metric("Bias Risk Level", risk)

    st.markdown(
        f'<p>Bias Risk: <span class="{css_class}">{risk}</span></p>',
        unsafe_allow_html=True,
    )

    # Store for AI buttons
    st.session_state["dataset_metrics_text"] = f"""
Sensitive attribute: {sensitive_col}
Target column: {target_col} (positive label = "{positive_label}")
Groups analysed: {", ".join(group_stats["group"].astype(str).tolist())}
Highest positive rate group: {highest['group']} ({highest['Positive_Rate']*100:.2f}%)
Lowest positive rate group: {lowest['group']} ({lowest['Positive_Rate']*100:.2f}%)
Outcome gap: {gap*100:.2f}%
Risk level: {risk}
"""

# ---- AI Explanation ----
if "dataset_metrics_text" in st.session_state:
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("🤖 Explain Bias (Claude)", use_container_width=True):
            with st.spinner("Claude analysing bias..."):
                explanation = explain_bias(st.session_state["dataset_metrics_text"])
            st.session_state["dataset_explanation"] = explanation

    with c2:
        if st.button("📄 Generate Audit Report", use_container_width=True):
            with st.spinner("Claude generating report..."):
                report = generate_fairness_report(st.session_state["dataset_metrics_text"])
            st.session_state["dataset_report"] = report

    with c3:
        if st.button("💡 Suggest Mitigations", use_container_width=True):
            with st.spinner("Claude suggesting mitigations..."):
                mitigations = suggest_mitigation(st.session_state["dataset_metrics_text"])
            st.session_state["dataset_mitigations"] = mitigations

    if "dataset_explanation" in st.session_state:
        with st.expander("🤖 Claude Bias Explanation", expanded=True):
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
# STEP 4 — MODEL PREDICTION BIAS
# ============================================================

st.markdown('<span class="step-badge">STEP 04</span> **Model Prediction Bias**', unsafe_allow_html=True)
st.caption(
    "Upload a CSV with columns: true labels, model predictions, and a sensitive attribute."
)

pred_file = st.file_uploader("Upload predictions CSV", type=["csv"], key="pred_file")

if pred_file is not None:

    pred_df = pd.read_csv(pred_file)
    st.markdown(f"**{len(pred_df):,} rows · {len(pred_df.columns)} columns**")
    st.dataframe(pred_df.head(6), use_container_width=True)

    pred_cols = pred_df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        true_col = st.selectbox("✅ True Label Column", pred_cols, key="sel_true_col")
    with c2:
        pred_col = st.selectbox("🔮 Prediction Column", pred_cols, key="sel_pred_col")
    with c3:
        sensitive_pred = st.selectbox(
            "🔒 Sensitive Column", pred_cols, key="sel_sensitive_pred"
        )

    if st.button("🔍 Analyse Model Bias", use_container_width=True):

        with st.spinner("Computing fairness metrics..."):
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

        st.subheader("Per-Group Fairness Metrics")
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
Sensitive attribute: {sensitive_pred}
Accuracy: {accuracy * 100:.2f}%
Demographic Parity Difference: {dp_diff:.4f}
Equalized Odds Difference: {eo_diff:.4f}
Bias risk level: {risk}
Per-group metrics:
{group_report.to_string(index=False)}
"""
        st.session_state["model_metrics_text"] = model_metrics_text
        st.session_state["pred_df"] = pred_df
        st.session_state["sensitive_pred"] = sensitive_pred
        st.session_state["pred_col"] = pred_col

    # ---- AI Buttons for Model ----
    if "model_metrics_text" in st.session_state:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🤖 Explain Model Bias (Claude)", use_container_width=True):
                with st.spinner("Claude analysing model fairness..."):
                    explanation = explain_bias(st.session_state["model_metrics_text"])
                st.session_state["model_explanation"] = explanation

        with c2:
            if st.button("💡 Model Mitigation Strategies", use_container_width=True):
                with st.spinner("Claude generating strategies..."):
                    mitigations = suggest_mitigation(
                        st.session_state["model_metrics_text"],
                        context="This is a classification model, focus on post-processing and in-processing fairness techniques.",
                    )
                st.session_state["model_mitigations"] = mitigations

        if "model_explanation" in st.session_state:
            with st.expander("🤖 Claude Model Bias Explanation", expanded=True):
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
    "Select a record and change its sensitive attribute to see if the model prediction changes — a sign of reliance on sensitive features."
)

if "pred_df" not in st.session_state:
    st.info("Complete Step 4 (Model Prediction Bias) to unlock the What-If Simulator.")
else:
    sim_df = st.session_state["pred_df"]
    sens_col = st.session_state["sensitive_pred"]
    p_col = st.session_state["pred_col"]

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
        original_pred = str(sim_df.loc[record_index, p_col])

        st.markdown("#### Simulation Result")
        col1, col2, col3 = st.columns(3)
        col1.metric("Original Group", original_group)
        col2.metric("Simulated Group", new_group)
        col3.metric("Model Prediction", original_pred)

        if original_group != new_group:
            st.warning(
                "⚠️ **Potential Bias Detected**: Changing the sensitive attribute altered the group membership. "
                "If a real model re-evaluated this record with the new group and changed its prediction, "
                "it would indicate direct reliance on the sensitive attribute."
            )
            # Ask Claude to comment
            sim_text = f"""
Record index: {record_index}
Original group ({sens_col}): {original_group}
Simulated group: {new_group}
Current prediction: {original_pred}
This is a what-if scenario to test if the model is sensitive to the protected attribute.
"""
            with st.spinner("Claude assessing the simulation..."):
                sim_explanation = explain_bias(sim_text)
            st.subheader("🤖 Claude Assessment")
            st.markdown(sim_explanation)
        else:
            st.success(
                "✅ No group change detected (same group selected). Try a different group to test counterfactual fairness."
            )

st.divider()
st.markdown(
    '<p style="text-align:center; color:#333; font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem;">FairCheck AI · Hack2Skill Hackathon · Powered by Gemini (Google)</p>',
    unsafe_allow_html=True,
)