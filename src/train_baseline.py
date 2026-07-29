"""Train the two baseline models and log them to MLflow.

Baselines (per the approved brief):
  1. DummyClassifier (majority class) - the "do nothing" reference. Any real
     model must beat it, especially on at-risk recall (the dummy scores 0).
  2. Logistic Regression - the simplest real model.

Rules enforced here:
  - All learned preprocessing (imputer, encoder, scaler) lives inside a
    sklearn Pipeline and is fitted on the TRAINING split only.
  - Metrics are computed on the VALIDATION split (2014B). The test split
    (2014J) is not touched - it stays frozen for the final evaluation.
  - Every run is logged to MLflow (local ./mlruns store, git-ignored).

Usage:  python src/train_baseline.py
"""

from pathlib import Path

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from features import CATEGORICAL, NUMERIC, KEY

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RANDOM_STATE = 42

CAT_FEATURES = [c for c in CATEGORICAL if c not in KEY] + ["code_module"]


def load_xy(split: str):
    df = pd.read_csv(PROCESSED / f"features_{split}.csv")
    y = df[config.TARGET_COLUMN]
    X = df.drop(columns=[config.TARGET_COLUMN, "code_presentation", "id_student"])
    return X, y


def make_preprocessor() -> ColumnTransformer:
    """Fit-on-train-only preprocessing (docs/preprocessing_manifest.json)."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CAT_FEATURES),
    ])


def evaluate(name, model, X_val, y_val) -> dict:
    pred = model.predict(X_val)
    metrics = {
        "recall_at_risk": recall_score(y_val, pred, zero_division=0),
        "precision_at_risk": precision_score(y_val, pred, zero_division=0),
        "f1_at_risk": f1_score(y_val, pred, zero_division=0),
        "accuracy": accuracy_score(y_val, pred),
    }
    if hasattr(model, "predict_proba"):
        metrics["pr_auc"] = average_precision_score(
            y_val, model.predict_proba(X_val)[:, 1]
        )
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    metrics.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    print(f"\n{name} (validation 2014B):")
    for k, v in metrics.items():
        print(f"  {k:18s} {v:.3f}" if isinstance(v, float) else f"  {k:18s} {v}")
    return metrics


def main() -> None:
    X_train, y_train = load_xy("train")
    X_val, y_val = load_xy("validation")
    print(f"train: {X_train.shape}, validation: {X_val.shape}")

    # Local SQLite store (git-ignored). Inspect runs with:
    #   mlflow ui --backend-store-uri sqlite:///mlflow.db
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("edu02-early-warning")

    runs = {
        "baseline_dummy_majority": DummyClassifier(strategy="most_frequent"),
        "baseline_logistic_regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
    }

    results = {}
    for run_name, estimator in runs.items():
        model = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", estimator),
        ])
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": type(estimator).__name__,
                "split": "train=2013B+2013J, val=2014B",
                "early_window_fraction": config.EARLY_WINDOW_FRACTION,
                "n_features_raw": X_train.shape[1],
                "random_state": RANDOM_STATE,
            })
            model.fit(X_train, y_train)
            metrics = evaluate(run_name, model, X_val, y_val)
            mlflow.log_metrics(metrics)
            results[run_name] = metrics

    # Sanity check the whole pipeline: the real baseline must clearly beat
    # the dummy on recall, otherwise something upstream is broken.
    assert results["baseline_logistic_regression"]["recall_at_risk"] > 0.5, \
        "logistic regression recall suspiciously low - check the pipeline"
    print("\nAll runs logged. View with: "
          "mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
