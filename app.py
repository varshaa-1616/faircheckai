# ============================================================
# FairCheck AI - Dataset Bias + Model Prediction Bias Checker
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

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


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="FairCheck AI",
    page_icon="⚖️",
    layout="wide"
)

# ============================================================
# CSS Fix for Dropdown + UI
# ============================================================

st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div {
        max-height: 300px !important;
        overflow-y: auto !important;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #4A90E2;
    }

    .subtitle {
        font-size: 18px;
        color: #555;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        background-color: #f7f9fc;
        border: 1px solid #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Header
# ============================================================

st.markdown("<div class='main-title'>⚖️ FairCheck AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Dataset Bias Detection + Model Prediction Fairness Checker</div>",
    unsafe_allow_html=True
)

st.write("")
st.write(
    "FairCheck AI helps users inspect datasets and model predictions for hidden unfairness. "
    "First upload a dataset to check dataset-level bias. Then optionally upload model predictions "
    "to check whether the model makes biased decisions."
)


# ============================================================
# Helper Functions
# ============================================================

def risk_label(value):
    value = abs(value)

    if value < 0.10:
        return "LOW"
    elif value < 0.20:
        return "MEDIUM"
    else:
        return "HIGH"


def show_risk(level, message):
    if level == "LOW":
        st.success(message)
    elif level == "MEDIUM":
        st.warning(message)
    else:
        st.error(message)


def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def binary_encode(series, column_name="column"):
    series = series.astype(str).str.strip()

    if series.nunique() != 2:
        st.error(
            f"Column **{column_name}** must be binary for this MVP. "
            f"It currently has {series.nunique()} unique values."
        )
        st.write("Unique values found:")
        st.write(series.unique())
        st.stop()

    values = sorted(series.unique())
    mapping = {values[0]: 0, values[1]: 1}
    encoded = series.map(mapping).values
    positive_label = values[1]

    return encoded, mapping, positive_label


def get_sensitive_candidates(df, target_col=None):
    candidates = []

    possible_keywords = [
        "gender", "sex", "race", "age", "caste", "religion",
        "region", "state", "country", "nationality", "disability",
        "marital", "relationship", "workclass", "education"
    ]

    for col in df.columns:
        if col == target_col:
            continue

        unique_count = df[col].nunique()

        if unique_count <= 20:
            candidates.append(col)
        else:
            for key in possible_keywords:
                if key in col.lower():
                    candidates.append(col)
                    break

    # remove duplicates
    candidates = list(dict.fromkeys(candidates))

    if len(candidates) == 0:
        candidates = [col for col in df.columns if col != target_col]

    return candidates


def dataset_bias_analysis(df, target_col, sensitive_col):
    y_true, target_mapping, positive_label = binary_encode(df[target_col], target_col)
    sensitive = df[sensitive_col].astype(str).str.strip()

    temp = pd.DataFrame({
        "sensitive_group": sensitive,
        "target_encoded": y_true
    })

    group_stats = (
        temp
        .groupby("sensitive_group")
        .agg(
            total_records=("target_encoded", "count"),
            positive_outcomes=("target_encoded", "sum"),
            positive_rate=("target_encoded", "mean")
        )
        .reset_index()
        .rename(columns={"sensitive_group": sensitive_col})
    )

    highest_group = group_stats.loc[group_stats["positive_rate"].idxmax()]
    lowest_group = group_stats.loc[group_stats["positive_rate"].idxmin()]

    positive_gap = highest_group["positive_rate"] - lowest_group["positive_rate"]
    dataset_risk = risk_label(positive_gap)

    return {
        "group_stats": group_stats,
        "highest_group": highest_group,
        "lowest_group": lowest_group,
        "positive_gap": positive_gap,
        "dataset_risk": dataset_risk,
        "target_mapping": target_mapping,
        "positive_label": positive_label
    }


def model_bias_analysis(pred_df, true_col, pred_col, sensitive_col):
    y_true, true_mapping, true_positive_label = binary_encode(pred_df[true_col], true_col)
    y_pred, pred_mapping, pred_positive_label = binary_encode(pred_df[pred_col], pred_col)
    sensitive = pred_df[sensitive_col].astype(str).str.strip()

    metric_frame = MetricFrame(
        metrics={
            "Selection Rate": selection_rate,
            "Accuracy": accuracy_score,
            "True Positive Rate": true_positive_rate,
            "False Positive Rate": false_positive_rate,
            "False Negative Rate": false_negative_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )

    group_report = metric_frame.by_group.reset_index()

    if "sensitive_feature_0" in group_report.columns:
        group_report = group_report.rename(columns={"sensitive_feature_0": sensitive_col})
    elif group_report.columns[0] != sensitive_col:
        group_report = group_report.rename(columns={group_report.columns[0]: sensitive_col})

    dp_diff = demographic_parity_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive
    )

    eo_diff = equalized_odds_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive
    )

    overall_accuracy = accuracy_score(y_true, y_pred)

    highest_selection = group_report.loc[group_report["Selection Rate"].idxmax()]
    lowest_selection = group_report.loc[group_report["Selection Rate"].idxmin()]

    highest_accuracy = group_report.loc[group_report["Accuracy"].idxmax()]
    lowest_accuracy = group_report.loc[group_report["Accuracy"].idxmin()]

    highest_fnr = group_report.loc[group_report["False Negative Rate"].idxmax()]
    lowest_fnr = group_report.loc[group_report["False Negative Rate"].idxmin()]

    selection_gap = highest_selection["Selection Rate"] - lowest_selection["Selection Rate"]
    accuracy_gap = highest_accuracy["Accuracy"] - lowest_accuracy["Accuracy"]
    fnr_gap = highest_fnr["False Negative Rate"] - lowest_fnr["False Negative Rate"]

    final_risk = risk_label(max(abs(dp_diff), abs(eo_diff), abs(selection_gap), abs(accuracy_gap)))

    return {
        "group_report": group_report,
        "dp_diff": dp_diff,
        "eo_diff": eo_diff,
        "overall_accuracy": overall_accuracy,
        "highest_selection": highest_selection,
        "lowest_selection": lowest_selection,
        "highest_accuracy": highest_accuracy,
        "lowest_accuracy": lowest_accuracy,
        "highest_fnr": highest_fnr,
        "lowest_fnr": lowest_fnr,
        "selection_gap": selection_gap,
        "accuracy_gap": accuracy_gap,
        "fnr_gap": fnr_gap,
        "final_risk": final_risk,
        "true_mapping": true_mapping,
        "pred_mapping": pred_mapping,
        "true_positive_label": true_positive_label,
        "pred_positive_label": pred_positive_label
    }


# ============================================================
# Step 1: Upload Dataset
# ============================================================

st.header("Step 1: Upload Dataset")

dataset_file = st.file_uploader(
    "Upload original dataset CSV",
    type=["csv"],
    key="dataset_upload"
)

if dataset_file is None:
    st.info("Upload a dataset CSV to begin.")
    st.stop()

df = pd.read_csv(dataset_file)

st.subheader("Raw Dataset Preview")
st.dataframe(df.head())

original_rows = df.shape[0]
df = df.dropna()
clean_rows = df.shape[0]

st.subheader("Dataset Information")

c1, c2, c3 = st.columns(3)
c1.metric("Original Rows", original_rows)
c2.metric("Rows After Removing Missing Values", clean_rows)
c3.metric("Columns", df.shape[1])

columns = df.columns.tolist()

st.write("---")


# # ============================================================
# # Step 2: Select Dataset Bias Columns
# # ============================================================

# st.header("Step 2: Select Columns for Dataset Bias Check")

# target_col = st.selectbox(
#     "Select target / outcome column",
#     columns,
#     help="Example: income, approved, hired, selected, admitted, loan_status"
# )

# sensitive_candidates = get_sensitive_candidates(df, target_col)

# st.write("Suggested sensitive columns:")
# st.write(", ".join(sensitive_candidates[:10]))

# sensitive_col = st.selectbox(
#     "Select sensitive column",
#     sensitive_candidates,
#     help="Example: gender, sex, age, race, region, workclass"
# )

# if target_col == sensitive_col:
#     st.warning("Target column and sensitive column must be different.")
#     st.stop()

# st.info(
#     f"You selected **{target_col}** as the decision/outcome column and "
#     f"**{sensitive_col}** as the sensitive attribute."
# )

# ============================================================
# Step 2: Select Dataset Bias Columns
# ============================================================

st.header("Step 2: Select Columns for Dataset Bias Check")

columns = df.columns.tolist()

st.subheader("Select target / outcome column")

target_col = st.radio(
    "Choose the decision/outcome column",
    columns,
    index=columns.index("income") if "income" in columns else 0,
    horizontal=False,
    key="target_radio"
)

sensitive_candidates = get_sensitive_candidates(df, target_col)

st.subheader("Select sensitive column")

sensitive_col = st.radio(
    "Choose the sensitive/group column",
    sensitive_candidates,
    index=sensitive_candidates.index("sex") if "sex" in sensitive_candidates else 0,
    horizontal=False,
    key="sensitive_radio"
)

if target_col == sensitive_col:
    st.warning("Target column and sensitive column must be different.")
    st.stop()

st.info(
    f"You selected **{target_col}** as the target/outcome column and "
    f"**{sensitive_col}** as the sensitive attribute."
)
st.write("---")


# ============================================================
# Step 3: Dataset Bias Analysis
# ============================================================

st.header("Step 3: Dataset Bias Analysis")

if st.button("Check Dataset Bias"):

    result = dataset_bias_analysis(df, target_col, sensitive_col)

    group_stats = result["group_stats"]
    highest_group = result["highest_group"]
    lowest_group = result["lowest_group"]
    positive_gap = result["positive_gap"]
    dataset_risk = result["dataset_risk"]
    target_mapping = result["target_mapping"]
    positive_label = result["positive_label"]

    st.subheader("Target Meaning")

    st.write(f"The selected target column is **{target_col}**.")
    st.write(f"The app treats **{positive_label}** as the positive / favorable outcome.")
    st.write("Encoding used:")
    st.write(target_mapping)

    st.subheader("Group-wise Dataset Bias Report")
    st.dataframe(group_stats)

    st.subheader("Dataset Bias Risk")

    show_risk(
        dataset_risk,
        f"{dataset_risk} DATASET BIAS RISK: Positive outcome gap is {positive_gap * 100:.2f}%."
    )

    st.subheader("Explanation in Simple Words")

    st.write(
        f"In this dataset, group **{highest_group[sensitive_col]}** receives the favorable outcome "
        f"**{positive_label}** most often."
    )

    st.write(
        f"Positive outcome rate for **{highest_group[sensitive_col]}**: "
        f"**{pct(highest_group['positive_rate'])}**"
    )

    st.write(
        f"Group **{lowest_group[sensitive_col]}** receives the favorable outcome least often."
    )

    st.write(
        f"Positive outcome rate for **{lowest_group[sensitive_col]}**: "
        f"**{pct(lowest_group['positive_rate'])}**"
    )

    st.write(
        f"The gap between these two groups is **{positive_gap * 100:.2f}%**."
    )

    st.subheader("What This Means")

    if dataset_risk == "HIGH":
        st.error(
            f"The dataset may contain strong historical unfairness. "
            f"If a model is trained directly on this data, it may learn to favor "
            f"**{highest_group[sensitive_col]}** over **{lowest_group[sensitive_col]}**."
        )
    elif dataset_risk == "MEDIUM":
        st.warning(
            f"The dataset shows moderate difference across **{sensitive_col}** groups. "
            f"This should be reviewed before training a model."
        )
    else:
        st.success(
            f"The dataset does not show a large favorable-outcome gap for **{sensitive_col}**. "
            f"However, other sensitive columns should also be tested."
        )

    st.subheader("Personalized Dataset Bias Suggestions")

    if positive_gap >= 0.20:
        st.write(
            f"**1.** Review why **{lowest_group[sensitive_col]}** has a much lower "
            f"positive outcome rate than **{highest_group[sensitive_col]}**."
        )
        st.write(
            f"**2.** Collect more representative data for **{lowest_group[sensitive_col]}**."
        )
        st.write(
            f"**3.** Use rebalancing or sample reweighting before training a model."
        )
        st.write(
            f"**4.** Check whether columns related to **{sensitive_col}** are acting as proxy variables."
        )

    elif positive_gap >= 0.10:
        st.write(
            f"**1.** The dataset shows moderate imbalance. Review the records for "
            f"**{lowest_group[sensitive_col]}**."
        )
        st.write(
            f"**2.** Try comparing results after balancing the dataset."
        )

    else:
        st.write(
            f"**1.** Dataset bias appears low for **{sensitive_col}**."
        )
        st.write(
            f"**2.** Still test other sensitive columns before deciding the dataset is fair."
        )

st.write("---")


# ============================================================
# Step 4: Optional Predictions Upload
# ============================================================

st.header("Step 4: Optional Model Prediction Fairness Check")

st.write(
    "Now upload a CSV file containing model predictions. "
    "This file should contain:"
)

st.code(
    """
Required columns:
1. True label column
2. Prediction column
3. Sensitive column

Example:
income, predicted_income, gender
<=50K, <=50K, Female
>50K, <=50K, Male
""",
    language="text"
)

pred_file = st.file_uploader(
    "Upload model predictions CSV",
    type=["csv"],
    key="prediction_upload"
)

if pred_file is not None:

    pred_df = pd.read_csv(pred_file)
    pred_df = pred_df.dropna()

    st.subheader("Predictions File Preview")
    st.dataframe(pred_df.head())

    pred_columns = pred_df.columns.tolist()

    st.subheader("Select Prediction File Columns")

    true_col = st.selectbox(
        "Select true label column",
        pred_columns,
        help="This is the actual correct label."
    )

    pred_col = st.selectbox(
        "Select prediction column",
        pred_columns,
        help="This is the model predicted label."
    )

    prediction_sensitive_candidates = get_sensitive_candidates(pred_df, true_col)

    sensitive_pred_col = st.selectbox(
        "Select sensitive column in predictions file",
        prediction_sensitive_candidates,
        help="This should be the group column, such as gender, race, age_group, region."
    )

    if true_col == pred_col:
        st.warning("True label column and prediction column should be different.")
        st.stop()

    if true_col == sensitive_pred_col or pred_col == sensitive_pred_col:
        st.warning("Sensitive column should be different from true label and prediction column.")
        st.stop()

    if st.button("Check Model Prediction Bias"):

        model_result = model_bias_analysis(
            pred_df=pred_df,
            true_col=true_col,
            pred_col=pred_col,
            sensitive_col=sensitive_pred_col
        )

        group_report = model_result["group_report"]
        dp_diff = model_result["dp_diff"]
        eo_diff = model_result["eo_diff"]
        overall_accuracy = model_result["overall_accuracy"]

        highest_selection = model_result["highest_selection"]
        lowest_selection = model_result["lowest_selection"]

        highest_accuracy = model_result["highest_accuracy"]
        lowest_accuracy = model_result["lowest_accuracy"]

        highest_fnr = model_result["highest_fnr"]
        lowest_fnr = model_result["lowest_fnr"]

        selection_gap = model_result["selection_gap"]
        accuracy_gap = model_result["accuracy_gap"]
        fnr_gap = model_result["fnr_gap"]
        final_risk = model_result["final_risk"]

        pred_mapping = model_result["pred_mapping"]
        pred_positive_label = model_result["pred_positive_label"]

        st.header("Model Prediction Fairness Results")

        st.subheader("Prediction Meaning")
        st.write(f"The selected prediction column is **{pred_col}**.")
        st.write(f"The app treats **{pred_positive_label}** as the positive model prediction.")
        st.write("Prediction encoding used:")
        st.write(pred_mapping)

        st.subheader("Overall Model Accuracy")
        st.metric("Accuracy", f"{overall_accuracy * 100:.2f}%")

        st.write(
            f"The uploaded model predictions matched the true labels in "
            f"**{overall_accuracy * 100:.2f}%** of records."
        )

        st.subheader("Group-wise Model Fairness Report")
        st.dataframe(group_report)

        st.subheader("Overall Model Bias Risk")

        show_risk(
            final_risk,
            f"{final_risk} MODEL BIAS RISK detected across **{sensitive_pred_col}** groups."
        )

        st.header("Metric Explanations Based on Your Predictions")

        st.subheader("1. Selection Rate")

        st.write(
            f"Selection Rate means how often the model predicts the favorable outcome "
            f"**{pred_positive_label}** for each **{sensitive_pred_col}** group."
        )

        st.write(
            f"The model gives favorable predictions most often to "
            f"**{highest_selection[sensitive_pred_col]}** at "
            f"**{pct(highest_selection['Selection Rate'])}**."
        )

        st.write(
            f"The model gives favorable predictions least often to "
            f"**{lowest_selection[sensitive_pred_col]}** at "
            f"**{pct(lowest_selection['Selection Rate'])}**."
        )

        st.write(
            f"The selection-rate gap is **{selection_gap * 100:.2f}%**."
        )

        st.subheader("2. Demographic Parity Difference")

        st.metric("Demographic Parity Difference", f"{dp_diff:.3f}")

        st.write(
            f"This metric checks whether different **{sensitive_pred_col}** groups receive "
            f"favorable predictions at similar rates."
        )

        st.write(
            f"In your predictions, the demographic parity gap is "
            f"**{abs(dp_diff) * 100:.2f}%**."
        )

        st.subheader("3. Equalized Odds Difference")

        st.metric("Equalized Odds Difference", f"{eo_diff:.3f}")

        st.write(
            f"This metric checks whether the model makes similar errors across "
            f"different **{sensitive_pred_col}** groups."
        )

        st.write(
            f"In your predictions, the equalized odds gap is "
            f"**{abs(eo_diff) * 100:.2f}%**."
        )

        st.subheader("4. Accuracy Gap")

        st.write(
            f"The model is most accurate for **{highest_accuracy[sensitive_pred_col]}** "
            f"with accuracy **{pct(highest_accuracy['Accuracy'])}**."
        )

        st.write(
            f"The model is least accurate for **{lowest_accuracy[sensitive_pred_col]}** "
            f"with accuracy **{pct(lowest_accuracy['Accuracy'])}**."
        )

        st.write(
            f"The accuracy gap is **{accuracy_gap * 100:.2f}%**."
        )

        st.subheader("5. False Negative Rate")

        st.write(
            f"False Negative Rate means the model wrongly denies the favorable outcome "
            f"**{pred_positive_label}** even when the true label was favorable."
        )

        st.write(
            f"The highest false negative rate is for **{highest_fnr[sensitive_pred_col]}** "
            f"at **{pct(highest_fnr['False Negative Rate'])}**."
        )

        st.header("Personalized Model Bias Suggestions")

        suggestions = []

        if selection_gap >= 0.20:
            suggestions.append(
                f"The model strongly favors **{highest_selection[sensitive_pred_col]}** over "
                f"**{lowest_selection[sensitive_pred_col]}** in favorable predictions."
            )
            suggestions.append(
                f"Use reweighting, resampling, or fairness-aware training to reduce this gap."
            )

        elif selection_gap >= 0.10:
            suggestions.append(
                f"The model shows moderate difference in favorable predictions between "
                f"**{highest_selection[sensitive_pred_col]}** and "
                f"**{lowest_selection[sensitive_pred_col]}**."
            )

        else:
            suggestions.append(
                f"The model's favorable prediction rates are relatively balanced across "
                f"**{sensitive_pred_col}** groups."
            )

        if accuracy_gap >= 0.15:
            suggestions.append(
                f"The model is much less accurate for **{lowest_accuracy[sensitive_pred_col]}**. "
                f"Collect more data or improve feature quality for this group."
            )

        if fnr_gap >= 0.15:
            suggestions.append(
                f"False negatives are much higher for **{highest_fnr[sensitive_pred_col]}**. "
                f"This group may be wrongly denied favorable outcomes more often."
            )

        if abs(dp_diff) >= 0.20:
            suggestions.append(
                f"Demographic parity difference is high. Reduce the favorable prediction gap "
                f"before deployment."
            )

        if abs(eo_diff) >= 0.20:
            suggestions.append(
                f"Equalized odds difference is high. Check false positives and false negatives "
                f"for each group separately."
            )

        for i, suggestion in enumerate(suggestions, start=1):
            st.write(f"**{i}.** {suggestion}")

        st.header("Deployment Recommendation")

        if final_risk == "LOW":
            st.success("Model can be considered for limited testing with continuous monitoring.")
        elif final_risk == "MEDIUM":
            st.warning("Model needs fairness review before deployment.")
        else:
            st.error("Model is not deployment-ready due to significant fairness risk.")