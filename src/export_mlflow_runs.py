"""Export the MLflow run log to a committed CSV.

The MLflow store (mlflow.db) is a local artifact and stays out of Git, so
without this export a reviewer could not inspect the tracked experiments.
This script writes reports/mlflow_runs.csv - one row per run with its
parameters and metrics - so the experiment record travels with the
repository.

Usage:  python src/export_mlflow_runs.py
"""

from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "mlflow_runs.csv"
EXPERIMENT = "edu02-early-warning"

# The columns worth reviewing, in reading order.
COLUMNS = [
    "run_name", "start_time", "model", "hypothesis", "split",
    "recall_at_risk", "precision_at_risk", "f1_at_risk", "pr_auc", "accuracy",
    "best_f1_threshold", "recall_at_best_thr", "precision_at_best_thr",
    "f1_at_best_thr", "tn", "fp", "fn", "tp",
    "test_recall", "test_precision", "test_f1", "test_pr_auc",
    "test_recall_no_ccc", "test_f1_no_ccc",
    "random_state", "early_window_fraction", "threshold",
]


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise SystemExit(
            f"experiment '{EXPERIMENT}' not found - run the training scripts first"
        )

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=["attributes.start_time ASC"])
    if runs.empty:
        raise SystemExit("no runs logged yet")

    # flatten: mlflow prefixes columns with params./metrics./tags.
    flat = pd.DataFrame({"run_name": runs["tags.mlflow.runName"],
                         "start_time": runs["start_time"].dt.strftime("%Y-%m-%d %H:%M")})
    for column in runs.columns:
        if column.startswith(("params.", "metrics.")):
            flat[column.split(".", 1)[1]] = runs[column]

    ordered = [c for c in COLUMNS if c in flat.columns]
    extra = [c for c in flat.columns if c not in ordered]
    flat = flat[ordered + extra]

    # keep the file readable: round metrics, drop all-empty columns
    for column in flat.columns:
        if pd.api.types.is_float_dtype(flat[column]):
            flat[column] = flat[column].round(4)
    flat = flat.dropna(axis=1, how="all")

    OUT.parent.mkdir(exist_ok=True)
    flat.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(flat)} runs, "
          f"{len(flat.columns)} columns)")
    print(flat[["run_name"] + [c for c in ("recall_at_risk", "f1_at_risk",
                                           "pr_auc", "test_recall")
                               if c in flat.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
