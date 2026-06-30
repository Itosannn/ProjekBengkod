"""Train, compare, tune, evaluate, and save a customer churn model."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from features import FeatureEngineer

TARGET = "churn"
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
RESULTS_PATH = MODEL_DIR / "model_comparison.csv"
METRICS_PATH = MODEL_DIR / "test_metrics.json"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

EXPECTED_COLUMNS = {
    "gender",
    "age",
    "country",
    "city",
    "signup_date",
    "last_purchase_date",
    "acquisition_channel",
    "device_type",
    "subscription_type",
    "is_premium_user",
    "total_visits",
    "avg_session_time",
    "pages_per_session",
    "email_open_rate",
    "email_click_rate",
    "total_spent",
    "avg_order_value",
    "discount_used",
    "coupon_code",
    "support_tickets",
    "refund_requested",
    "delivery_delay_days",
    "payment_method",
    "satisfaction_score",
    "nps_score",
    "marketing_spend_per_user",
    "lifetime_value",
    "last_3_month_purchase_freq",
    TARGET,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/Sales Marketing.csv",
        help="Path to the CSV dataset.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--n-iter",
        type=int,
        default=12,
        help="Maximum randomized-search combinations per model.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lr", "rf", "et"],
        default=["lr", "rf", "et"],
        help="Models to compare: lr, rf, et.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs. Use 1 if your computer has limited resources.",
    )
    return parser.parse_args()


def validate_data(df: pd.DataFrame) -> None:
    missing = sorted(EXPECTED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    target_values = set(pd.Series(df[TARGET]).dropna().unique().tolist())
    if not target_values.issubset({0, 1, False, True}):
        raise ValueError("Column 'churn' must contain binary values 0 and 1.")


def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    preview_engineer = FeatureEngineer().fit(X_train)
    engineered = preview_engineer.transform(X_train)

    numeric_columns = engineered.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [
        column for column in engineered.columns if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=2,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_columns, categorical_columns


def candidate_models(random_state: int, n_jobs: int) -> dict[str, tuple[str, Any, dict[str, list[Any]]]]:
    return {
        "lr": (
            "Logistic Regression",
            LogisticRegression(max_iter=3000, random_state=random_state),
            {
                "model__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "model__solver": ["lbfgs"],
            },
        ),
        "rf": (
            "Random Forest",
            RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
            {
                "model__n_estimators": [200, 350, 500],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.5],
            },
        ),
        "et": (
            "Extra Trees",
            ExtraTreesClassifier(random_state=random_state, n_jobs=n_jobs),
            {
                "model__n_estimators": [200, 350, 500],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.5],
            },
        ),
    }


def build_pipeline(preprocessor: ColumnTransformer, model: Any, random_state: int):
    return ImbPipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", preprocessor),
            # Oversampling occurs inside every CV training fold, preventing leakage.
            ("sampler", RandomOverSampler(random_state=random_state)),
            ("model", model),
        ]
    )


def safe_metric(metric_fn, y_true, values) -> float:
    try:
        return float(metric_fn(y_true, values))
    except ValueError:
        return float("nan")


def classification_metrics(y_true, probabilities, threshold: float) -> dict[str, Any]:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": safe_metric(roc_auc_score, y_true, probabilities),
        "average_precision": safe_metric(
            average_precision_score, y_true, probabilities
        ),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def find_best_threshold(y_true, probabilities) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def python_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def build_metadata(X_train: pd.DataFrame) -> dict[str, Any]:
    date_columns = [
        column for column in ("signup_date", "last_purchase_date") if column in X_train
    ]
    text_columns = [column for column in ("coupon_code",) if column in X_train]
    ignored_columns = [column for column in ("customer_id",) if column in X_train]

    categorical_columns = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    categorical_columns = [
        column
        for column in categorical_columns
        if column not in date_columns + text_columns
    ]

    numeric_columns = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    binary_columns = []
    for column in numeric_columns:
        unique = set(pd.Series(X_train[column]).dropna().unique().tolist())
        if unique and unique.issubset({0, 1, False, True}):
            binary_columns.append(column)
    numeric_columns = [column for column in numeric_columns if column not in binary_columns]

    category_options = {}
    for column in categorical_columns:
        values = sorted(
            X_train[column].dropna().astype(str).str.strip().loc[lambda s: s.ne("")].unique().tolist()
        )
        category_options[column] = values

    numeric_stats = {}
    for column in numeric_columns:
        values = pd.to_numeric(X_train[column], errors="coerce")
        numeric_stats[column] = {
            "min": python_value(values.min()),
            "max": python_value(values.max()),
            "median": python_value(values.median()),
            "is_integer": bool(pd.api.types.is_integer_dtype(X_train[column].dtype)),
        }

    date_defaults = {}
    for column in date_columns:
        dates = pd.to_datetime(X_train[column], errors="coerce").dropna()
        default_date = dates.median() if not dates.empty else pd.Timestamp("2024-01-01")
        date_defaults[column] = pd.Timestamp(default_date).date().isoformat()

    groups = {
        "Customer profile": [
            "gender", "age", "country", "city", "signup_date",
            "acquisition_channel", "device_type", "subscription_type",
            "is_premium_user", "payment_method",
        ],
        "Engagement": [
            "last_purchase_date", "total_visits", "avg_session_time",
            "pages_per_session", "email_open_rate", "email_click_rate",
            "last_3_month_purchase_freq",
        ],
        "Transactions": [
            "total_spent", "avg_order_value", "discount_used", "coupon_code",
            "refund_requested", "delivery_delay_days",
            "marketing_spend_per_user", "lifetime_value",
        ],
        "Customer experience": [
            "support_tickets", "satisfaction_score", "nps_score",
        ],
    }

    all_columns = list(X_train.columns)
    groups = {
        group: [column for column in columns if column in all_columns]
        for group, columns in groups.items()
    }

    return {
        "raw_feature_columns": all_columns,
        "required_batch_columns": [
            column for column in all_columns if column not in ignored_columns + text_columns
        ],
        "ignored_columns": ignored_columns,
        "date_columns": date_columns,
        "text_columns": text_columns,
        "categorical_columns": categorical_columns,
        "binary_columns": binary_columns,
        "numeric_columns": numeric_columns,
        "category_options": category_options,
        "numeric_stats": numeric_stats,
        "date_defaults": date_defaults,
        "groups": groups,
    }


def save_feature_importance(best_estimator) -> None:
    model = best_estimator.named_steps["model"]
    preprocessor = best_estimator.named_steps["preprocess"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return

    result = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)
    result.to_csv(IMPORTANCE_PATH, index=False)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Put the CSV in data/ or pass --data."
        )

    df = pd.read_csv(data_path)
    validate_data(df)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X_train)
    cv = StratifiedKFold(
        n_splits=args.cv, shuffle=True, random_state=args.random_state
    )
    scoring = {
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    comparison_rows = []
    fitted_searches = {}
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Numeric features after engineering: {len(numeric_columns)}")
    print(f"Categorical features after engineering: {len(categorical_columns)}")

    candidates = candidate_models(args.random_state, args.n_jobs)
    for code in args.models:
        name, model, parameters = candidates[code]
        pipeline = build_pipeline(preprocessor, model, args.random_state)
        total_combinations = len(list(ParameterGrid(parameters)))
        n_iter = min(args.n_iter, total_combinations)
        print(f"\nSearching {name}: {n_iter} parameter combinations...")

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameters,
            n_iter=n_iter,
            scoring=scoring,
            refit="f1",
            cv=cv,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        fitted_searches[name] = search
        best_index = search.best_index_
        row = {
            "model": name,
            "cv_f1": float(search.cv_results_["mean_test_f1"][best_index]),
            "cv_recall": float(search.cv_results_["mean_test_recall"][best_index]),
            "cv_precision": float(search.cv_results_["mean_test_precision"][best_index]),
            "cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][best_index]),
            "cv_average_precision": float(
                search.cv_results_["mean_test_average_precision"][best_index]
            ),
            "best_params": json.dumps(search.best_params_, default=str),
        }
        comparison_rows.append(row)
        print(
            f"Best CV F1={row['cv_f1']:.4f}, recall={row['cv_recall']:.4f}, "
            f"PR-AUC={row['cv_average_precision']:.4f}"
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("cv_f1", ascending=False)
    best_name = comparison.iloc[0]["model"]
    best_search = fitted_searches[best_name]
    best_estimator = best_search.best_estimator_

    print(f"\nSelected model by mean CV F1: {best_name}")
    print(f"Best parameters: {best_search.best_params_}")

    oof_probabilities = cross_val_predict(
        best_estimator,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=args.n_jobs,
    )[:, 1]
    tuned_threshold = find_best_threshold(y_train, oof_probabilities)

    test_probabilities = best_estimator.predict_proba(X_test)[:, 1]
    default_metrics = classification_metrics(y_test, test_probabilities, 0.5)
    tuned_metrics = classification_metrics(y_test, test_probabilities, tuned_threshold)

    metadata = build_metadata(X_train)
    artifact = {
        "pipeline": best_estimator,
        "threshold": tuned_threshold,
        "model_name": best_name,
        "best_params": best_search.best_params_,
        "selection_metric": "mean cross-validation F1",
        "default_test_metrics": default_metrics,
        "test_metrics": tuned_metrics,
        "metadata": metadata,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    comparison.to_csv(RESULTS_PATH, index=False)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "model_name": best_name,
                "best_params": best_search.best_params_,
                "default_threshold_metrics": default_metrics,
                "tuned_threshold_metrics": tuned_metrics,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    save_feature_importance(best_estimator)

    print("\nFinal test metrics with tuned threshold")
    print(json.dumps(tuned_metrics, indent=2))
    print(f"\nSaved model artifact to: {MODEL_PATH}")
    print(f"Saved comparison table to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
