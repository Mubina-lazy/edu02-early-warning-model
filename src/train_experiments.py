"""Model Gate experiments: compare approaches and pick the final model.

Each run tests one hypothesis (logged to MLflow, evaluated on validation):

  E1 logreg_balanced   - does class weighting raise the baseline's recall?
  E2 random_forest     - do non-linear feature interactions help?
  E3 random_forest_deep- does letting the forest grow deeper help or overfit?
  E4 xgboost           - does gradient boosting beat the forest?
  E5 xgboost_weighted  - boosting + class weighting for recall

For every run we also compute the best-F1 decision threshold on validation
(the default 0.5 is not sacred - the brief asks for actionable risk flags,
so the operating point matters as much as the model).

The test split (2014J) is NOT touched here. Final selection happens on
validation; the chosen pipeline is saved for the one-shot test evaluation
in src/evaluate_final.py.

Usage:  python src/train_experiments.py
"""

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from train_baseline import (
    ROOT, RANDOM_STATE, evaluate, load_xy, make_preprocessor,
)

MODELS_DIR = ROOT / "models"


def best_f1_threshold(model, X_val, y_val):
    """Threshold on the predicted probability that maximizes F1 on validation."""
    proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thresholds[i]), float(f1[i]), float(recall[i]), float(precision[i])


def main() -> None:
    X_train, y_train = load_xy("train")
    X_val, y_val = load_xy("validation")
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"train {X_train.shape}, validation {X_val.shape}, "
          f"neg/pos weight = {pos_weight:.2f}")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("edu02-early-warning")

    experiments = {
        "E1_logreg_balanced": (
            "does class weighting raise baseline recall?",
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE),
        ),
        "E2_random_forest": (
            "do non-linear interactions help?",
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   class_weight="balanced_subsample",
                                   random_state=RANDOM_STATE, n_jobs=-1),
        ),
        "E3_random_forest_deep": (
            "does an unconstrained forest overfit?",
            RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                   class_weight="balanced_subsample",
                                   random_state=RANDOM_STATE, n_jobs=-1),
        ),
        "E4_xgboost": (
            "does gradient boosting beat the forest?",
            XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                          subsample=0.9, colsample_bytree=0.9,
                          random_state=RANDOM_STATE, eval_metric="logloss"),
        ),
        "E5_xgboost_weighted": (
            "boosting + class weighting for recall",
            XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                          subsample=0.9, colsample_bytree=0.9,
                          scale_pos_weight=pos_weight,
                          random_state=RANDOM_STATE, eval_metric="logloss"),
        ),
    }

    results = {}
    fitted = {}
    for run_name, (hypothesis, estimator) in experiments.items():
        model = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", estimator),
        ])
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": type(estimator).__name__,
                "hypothesis": hypothesis,
                "split": "train=2013B+2013J, val=2014B",
                "random_state": RANDOM_STATE,
                **{f"est_{k}": v for k, v in estimator.get_params().items()
                   if k in ("class_weight", "n_estimators", "min_samples_leaf",
                            "max_depth", "learning_rate", "scale_pos_weight")},
            })
            model.fit(X_train, y_train)
            metrics = evaluate(run_name, model, X_val, y_val)
            thr, f1_t, rec_t, prec_t = best_f1_threshold(model, X_val, y_val)
            metrics.update({"best_f1_threshold": thr, "f1_at_best_thr": f1_t,
                            "recall_at_best_thr": rec_t,
                            "precision_at_best_thr": prec_t})
            print(f"  tuned threshold {thr:.3f} -> "
                  f"recall {rec_t:.3f}, precision {prec_t:.3f}, F1 {f1_t:.3f}")
            mlflow.log_metrics(metrics)
            results[run_name] = metrics
            fitted[run_name] = model

    # Model selection: primary = F1 at tuned threshold (recall-oriented brief,
    # but recall alone would just pick the lowest threshold); tie-break PR-AUC.
    winner = max(results, key=lambda r: (round(results[r]["f1_at_best_thr"], 3),
                                         round(results[r]["pr_auc"], 3)))
    print(f"\nSelected final model: {winner}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[winner], MODELS_DIR / "final_model.joblib")
    # Record the library versions that produced the artifact. Loading a
    # pickled sklearn pipeline with a different scikit-learn raises
    # InconsistentVersionWarning, so predict.py compares against these and
    # tells the user what to install instead of silently scoring.
    import sklearn
    import xgboost
    import pandas

    meta = {
        "run_name": winner,
        "threshold": results[winner]["best_f1_threshold"],
        "fitted_with": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pandas.__version__,
        },
        # numpy scalars (e.g. confusion-matrix counts) are not JSON
        # serializable - cast everything to plain Python floats
        "validation_metrics": {k: float(v) for k, v in results[winner].items()},
        "trained_on": "2013B+2013J",
        "threshold_tuned_on": "2014B (validation)",
    }
    (MODELS_DIR / "final_model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved models/final_model.joblib + final_model_meta.json")


if __name__ == "__main__":
    main()
