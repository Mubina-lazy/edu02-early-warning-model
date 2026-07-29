"""One-shot final evaluation of the selected model on the frozen test split.

This script is run ONCE, after model selection is finished on validation.
It reports, on test (2014J):

  - headline metrics at the tuned threshold (primary) and at 0.5 (reference)
  - the DQ-06 protocol: full test AND test excluding module CCC, plus a
    per-module table (CCC has no training data - a documented cold-start)
  - comparison against both baselines (refitted on train, same configs)
  - fairness slices: disability, imd_band, gender, age_band
  - error analysis: what the missed at-risk students (false negatives)
    look like compared to the caught ones

Outputs: reports/final_evaluation.md, reports/figures/final_confusion.png,
reports/figures/final_pr_curve.png; metrics also logged to MLflow.

Usage:  python src/evaluate_final.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline

import config
from train_baseline import ROOT, RANDOM_STATE, load_xy, make_preprocessor

FIG = ROOT / "reports" / "figures"
REPORT = ROOT / "reports" / "final_evaluation.md"


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "flag_rate": float(pred.mean()),
        "cm": confusion_matrix(y_true, pred),
    }


def fmt(m):
    return f"recall {m['recall']:.3f}, precision {m['precision']:.3f}, f1 {m['f1']:.3f}"


def main() -> None:
    model = joblib.load(ROOT / "models" / "final_model.joblib")
    meta = json.loads((ROOT / "models" / "final_model_meta.json").read_text())
    thr = meta["threshold"]

    X_test, y_test = load_xy("test")
    test_df = pd.read_csv(ROOT / "data/processed/features_test.csv")
    proba = model.predict_proba(X_test)[:, 1]

    lines = ["# Final Evaluation on the Test Split (2014J)", ""]
    lines += [f"Model: **{meta['run_name']}** (see `reports/experiments_results.md`); "
              f"operating threshold **{thr:.3f}** tuned on validation. "
              f"Test = 8,746 students of the 2014J cohort, never used before "
              f"this evaluation.", ""]

    # --- headline metrics ---------------------------------------------------
    main_m = metrics_at(y_test, proba, thr)
    ref_m = metrics_at(y_test, proba, 0.5)
    pr_auc = average_precision_score(y_test, proba)

    # baselines refitted on train with the same configs as train_baseline.py
    X_train, y_train = load_xy("train")
    base_metrics = {}
    for name, est in [
        ("Dummy (majority)", DummyClassifier(strategy="most_frequent")),
        ("Logistic Regression", LogisticRegression(max_iter=2000,
                                                   random_state=RANDOM_STATE)),
    ]:
        pipe = Pipeline([("preprocess", make_preprocessor()), ("model", est)])
        pipe.fit(X_train, y_train)
        p = (pipe.predict_proba(X_test)[:, 1] if hasattr(est, "predict_proba")
             else pipe.predict(X_test))
        base_metrics[name] = metrics_at(y_test, p, 0.5)

    lines += ["## Headline results", "",
              "| Model | Threshold | Recall (at-risk) | Precision | F1 | PR-AUC |",
              "|---|---|---|---|---|---|"]
    for name, m in base_metrics.items():
        lines += [f"| {name} | 0.5 | {m['recall']:.3f} | {m['precision']:.3f} "
                  f"| {m['f1']:.3f} | — |"]
    lines += [f"| **XGBoost (final)** | 0.5 | {ref_m['recall']:.3f} "
              f"| {ref_m['precision']:.3f} | {ref_m['f1']:.3f} | {pr_auc:.3f} |",
              f"| **XGBoost (final)** | **{thr:.3f}** | **{main_m['recall']:.3f}** "
              f"| **{main_m['precision']:.3f}** | **{main_m['f1']:.3f}** | {pr_auc:.3f} |",
              ""]

    tn, fp, fn, tp = main_m["cm"].ravel()
    lines += ["Confusion matrix at the operating threshold:", "",
              "|  | predicted not at risk | predicted at risk |",
              "|---|---|---|",
              f"| **actually not at risk** | {tn:,} (TN) | {fp:,} (FP) |",
              f"| **actually at risk** | {fn:,} (FN) | {tp:,} (TP) |", ""]

    # --- DQ-06 protocol: with/without CCC + per-module ----------------------
    no_ccc = test_df.code_module != "CCC"
    m_no_ccc = metrics_at(y_test[no_ccc], proba[no_ccc], thr)
    lines += ["## DQ-06 protocol: cold-start module CCC", "",
              f"- Full test set: {fmt(main_m)}",
              f"- Excluding CCC (no training data for it): {fmt(m_no_ccc)}", "",
              "| Module | n | at-risk rate | recall | precision | f1 | note |",
              "|---|---|---|---|---|---|---|"]
    for mod, g in test_df.groupby("code_module"):
        idx = g.index
        mm = metrics_at(y_test.loc[idx], proba[idx], thr)
        note = "**no training data (cold start)**" if mod == "CCC" else ""
        lines += [f"| {mod} | {len(g):,} | {y_test.loc[idx].mean():.2f} "
                  f"| {mm['recall']:.3f} | {mm['precision']:.3f} "
                  f"| {mm['f1']:.3f} | {note} |"]
    lines += [""]

    # --- fairness slices ----------------------------------------------------
    lines += ["## Fairness slices (at the operating threshold)", "",
              "Flag rate = share of the group the model marks at-risk. "
              "Recall parity matters most here: are we missing struggling "
              "students more often in some groups?", ""]
    for col in ["disability", "gender", "age_band", "imd_band"]:
        lines += [f"**{col}**", "",
                  "| group | n | at-risk rate | recall | precision | flag rate |",
                  "|---|---|---|---|---|---|"]
        for val, g in test_df.groupby(col, dropna=False):
            idx = g.index
            mm = metrics_at(y_test.loc[idx], proba[idx], thr)
            lines += [f"| {val} | {len(g):,} | {y_test.loc[idx].mean():.2f} "
                      f"| {mm['recall']:.3f} | {mm['precision']:.3f} "
                      f"| {mm['flag_rate']:.2f} |"]
        lines += [""]

    # --- error analysis: who do we miss? ------------------------------------
    pred = (proba >= thr).astype(int)
    at_risk = test_df[y_test == 1].copy()
    at_risk["caught"] = pred[y_test == 1] == 1
    fn_g, tp_g = at_risk[~at_risk.caught], at_risk[at_risk.caught]
    lines += ["## Error analysis: the students we miss (false negatives)", "",
              f"{len(fn_g):,} at-risk students were not flagged. Compared with "
              f"the {len(tp_g):,} we caught, they look far more 'healthy' at "
              f"the cutoff:", "",
              "| median at cutoff | missed (FN) | caught (TP) |",
              "|---|---|---|",
              f"| early clicks | {fn_g.early_total_clicks.median():.0f} "
              f"| {tp_g.early_total_clicks.median():.0f} |",
              f"| active days | {fn_g.early_active_days.median():.0f} "
              f"| {tp_g.early_active_days.median():.0f} |",
              f"| TMA submitted | {fn_g.early_tma_submitted_count.median():.0f} "
              f"| {tp_g.early_tma_submitted_count.median():.0f} |",
              f"| TMA mean score | {fn_g.early_tma_mean_score.median():.0f} "
              f"| {tp_g.early_tma_mean_score.median():.0f} |", "",
              "Interpretation: the missed students were still engaged and "
              "scoring reasonably during the first quarter of the course - "
              "their problems develop **later** than our prediction point. "
              "This is an honest limitation of a single early prediction "
              "point, not a pipeline bug; re-scoring students at later "
              "checkpoints would be the natural extension.", ""]

    # --- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow([[tn, fp], [fn, tp]], cmap="Blues")
    for (i, j), v in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [tn, fp, fn, tp]):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], ["pred: not at risk", "pred: at risk"])
    ax.set_yticks([0, 1], ["true: not at risk", "true: at risk"])
    ax.set_title(f"Test confusion matrix (thr={thr:.3f})")
    plt.tight_layout()
    plt.savefig(FIG / "final_confusion.png", dpi=120)

    prec, rec, _ = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(rec, prec, color="#4878a8")
    ax.scatter([main_m["recall"]], [main_m["precision"]], color="#a85448",
               zorder=5, label=f"operating point (thr={thr:.3f})")
    ax.axhline(y_test.mean(), ls="--", c="gray", lw=1,
               label=f"baseline precision = {y_test.mean():.2f}")
    ax.set_xlabel("recall (at-risk)")
    ax.set_ylabel("precision (at-risk)")
    ax.set_title(f"Test PR curve (PR-AUC = {pr_auc:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "final_pr_curve.png", dpi=120)

    lines += ["## Figures", "",
              "![confusion matrix](figures/final_confusion.png)",
              "![PR curve](figures/final_pr_curve.png)", ""]

    # --- conclusion ----------------------------------------------------------
    lines += ["## Conclusion", "",
              f"On a completely unseen future cohort the model catches "
              f"**{main_m['recall']:.0%}** of at-risk students "
              f"({tp:,} of {tp+fn:,}) while **{main_m['precision']:.0%}** of "
              f"its flags are correct, versus 0% recall for the majority "
              f"baseline and {base_metrics['Logistic Regression']['recall']:.0%} "
              f"for plain logistic regression. Performance holds without the "
              f"cold-start module CCC ({fmt(m_no_ccc)}). The model is a "
              f"decision-support ranking tool for advisors, not an automatic "
              f"decision system; limitations and fairness observations above "
              f"must accompany any use of the scores.", ""]

    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT.relative_to(ROOT)}")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("edu02-early-warning")
    with mlflow.start_run(run_name="FINAL_test_evaluation"):
        mlflow.log_params({"model": meta["run_name"], "threshold": thr,
                           "split": "test=2014J (one-shot)"})
        mlflow.log_metrics({
            "test_recall": main_m["recall"], "test_precision": main_m["precision"],
            "test_f1": main_m["f1"], "test_pr_auc": pr_auc,
            "test_recall_no_ccc": m_no_ccc["recall"],
            "test_f1_no_ccc": m_no_ccc["f1"],
        })
    print("logged FINAL_test_evaluation to MLflow")
    print(f"\nTEST RESULT: {fmt(main_m)}, pr_auc {pr_auc:.3f}")


if __name__ == "__main__":
    main()
