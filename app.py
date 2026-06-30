"""Streamlit app for customer churn prediction."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Required so joblib can resolve the custom transformer class.
from features import FeatureEngineer  # noqa: F401

MODEL_PATH = Path("model/churn_model.joblib")

st.set_page_config(
    page_title="Sales Marketing Churn Predictor",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def label(column: str) -> str:
    return column.replace("_", " ").title()


def number_input(column: str, stats: dict, key: str):
    minimum = stats.get("min")
    maximum = stats.get("max")
    median = stats.get("median")
    integer = stats.get("is_integer", False)

    if median is None or not np.isfinite(median):
        median = 0
    if integer:
        value = int(round(median))
        kwargs = {"value": value, "step": 1}
        if minimum is not None and np.isfinite(minimum):
            kwargs["min_value"] = int(np.floor(minimum))
        if maximum is not None and np.isfinite(maximum):
            kwargs["max_value"] = int(np.ceil(maximum))
    else:
        value = float(median)
        spread = 1.0
        if minimum is not None and maximum is not None:
            spread = max(float(maximum) - float(minimum), 1.0)
        kwargs = {"value": value, "step": max(spread / 100, 0.01), "format": "%.4f"}
        if minimum is not None and np.isfinite(minimum):
            kwargs["min_value"] = float(minimum)
        if maximum is not None and np.isfinite(maximum):
            kwargs["max_value"] = float(maximum)

    return st.number_input(label(column), key=key, **kwargs)


def render_field(column: str, metadata: dict, key_prefix: str):
    key = f"{key_prefix}_{column}"
    if column in metadata["binary_columns"]:
        choice = st.selectbox(label(column), [0, 1], format_func=lambda x: "Yes" if x else "No", key=key)
        return int(choice)
    if column in metadata["date_columns"]:
        default_value = date.fromisoformat(metadata["date_defaults"][column])
        return st.date_input(label(column), value=default_value, key=key).isoformat()
    if column in metadata["categorical_columns"]:
        options = metadata["category_options"].get(column, [])
        if not options:
            return st.text_input(label(column), key=key)
        return st.selectbox(label(column), options, key=key)
    if column in metadata["text_columns"]:
        return st.text_input(label(column), placeholder="Optional", key=key)
    if column in metadata["numeric_columns"]:
        return number_input(column, metadata["numeric_stats"][column], key)
    return st.text_input(label(column), key=key)


def predict(frame: pd.DataFrame, artifact: dict) -> tuple[np.ndarray, np.ndarray]:
    probabilities = artifact["pipeline"].predict_proba(frame)[:, 1]
    predictions = (probabilities >= artifact["threshold"]).astype(int)
    return probabilities, predictions


st.title("Customer Churn Prediction")
st.caption("Predict customer churn risk from sales and marketing behavior.")

if not MODEL_PATH.exists():
    st.error(
        "Model file is missing. Run `python train_model.py --data \"data/Sales Marketing.csv\"` first, "
        "then commit the generated `model/churn_model.joblib` file."
    )
    st.stop()

try:
    artifact = load_artifact()
except Exception as exc:
    st.exception(exc)
    st.stop()

metadata = artifact["metadata"]
metrics = artifact["test_metrics"]

metric_columns = st.columns(5)
metric_columns[0].metric("Selected model", artifact["model_name"])
metric_columns[1].metric("Test F1", f"{metrics['f1']:.3f}")
metric_columns[2].metric("Test recall", f"{metrics['recall']:.3f}")
metric_columns[3].metric("Test precision", f"{metrics['precision']:.3f}")
metric_columns[4].metric("Decision threshold", f"{artifact['threshold']:.3f}")

single_tab, batch_tab, model_tab = st.tabs(
    ["Single prediction", "Batch prediction", "Model details"]
)

with single_tab:
    st.subheader("Enter customer data")
    with st.form("single_prediction_form"):
        values = {}
        for group_name, columns in metadata["groups"].items():
            with st.expander(group_name, expanded=group_name == "Customer profile"):
                grid = st.columns(3)
                for index, column in enumerate(columns):
                    with grid[index % 3]:
                        values[column] = render_field(column, metadata, "single")

        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        for column in metadata["raw_feature_columns"]:
            values.setdefault(column, np.nan)
        input_frame = pd.DataFrame([values], columns=metadata["raw_feature_columns"])
        probabilities, predictions = predict(input_frame, artifact)
        probability = float(probabilities[0])
        prediction = int(predictions[0])

        st.subheader("Prediction result")
        left, right = st.columns([1, 2])
        left.metric("Churn probability", f"{probability:.1%}")
        if prediction == 1:
            right.error(
                "The customer is classified as likely to churn. Prioritize a retention action."
            )
        else:
            right.success(
                "The customer is classified as likely to stay. Continue normal engagement."
            )
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            f"Classification uses the validated threshold of {artifact['threshold']:.3f}, not a fixed 0.50 threshold."
        )

with batch_tab:
    st.subheader("Upload customer CSV")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = sorted(set(metadata["required_batch_columns"]) - set(batch.columns))
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            for column in metadata["raw_feature_columns"]:
                if column not in batch.columns:
                    batch[column] = np.nan
            model_input = batch[metadata["raw_feature_columns"]]
            probabilities, predictions = predict(model_input, artifact)
            result = batch.copy()
            result["churn_probability"] = probabilities
            result["churn_prediction"] = predictions

            st.dataframe(result.head(100), use_container_width=True)
            summary = pd.DataFrame(
                {
                    "prediction": ["Stay", "Churn"],
                    "customers": [int((predictions == 0).sum()), int((predictions == 1).sum())],
                }
            ).set_index("prediction")
            st.bar_chart(summary)
            st.download_button(
                "Download predictions",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

with model_tab:
    st.subheader("Validated model information")
    st.write(f"**Selection rule:** {artifact['selection_metric']}")
    st.write(f"**Best model:** {artifact['model_name']}")
    st.write(f"**Best parameters:** `{artifact['best_params']}`")
    st.write("**Test metrics with tuned threshold:**")
    metrics_table = pd.DataFrame(
        {
            "metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
            "value": [
                metrics["accuracy"], metrics["precision"], metrics["recall"],
                metrics["f1"], metrics["roc_auc"], metrics["average_precision"],
            ],
        }
    )
    st.dataframe(metrics_table, hide_index=True, use_container_width=True)
    st.write("**Confusion matrix:**")
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual stay", "Actual churn"],
        columns=["Predicted stay", "Predicted churn"],
    )
    st.dataframe(matrix, use_container_width=True)
    st.info(
        "This app estimates risk. Use the probability together with business costs, retention capacity, and customer context."
    )
