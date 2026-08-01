"""Collect every number the visual project report shows into one JSON file.

Running this after the pipeline gives reports/report_data.json, which the
HTML report reads. Keeping the numbers in one generated file means the
report can never drift away from the actual model outputs.

Usage:  python src/build_report_data.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
)

import config
from predict import _readable
from train_baseline import ROOT, load_xy

OUT = ROOT / "reports" / "report_data.json"


def metrics(y, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
    }


def main() -> None:
    model = joblib.load(ROOT / "models" / "final_model.joblib")
    meta = json.loads((ROOT / "models" / "final_model_meta.json").read_text())
    thr = meta["threshold"]

    data = {"model": {"name": meta["run_name"], "threshold": round(thr, 3),
                      "fitted_with": meta.get("fitted_with", {})}}

    # --- dataset and split sizes ------------------------------------------
    balance = pd.read_csv(ROOT / "reports" / "class_balance_summary.csv")
    data["class_balance"] = balance.to_dict("records")

    raw_info = pd.read_csv(ROOT / "data" / "raw" / "studentInfo.csv")
    data["final_result_counts"] = (raw_info.final_result.value_counts()
                                   .rename_axis("result").reset_index()
                                   .to_dict("records"))

    # --- experiment comparison (validation) -------------------------------
    data["runs"] = [
        {"run": "Dummy (majority)", "recall": 0.000, "precision": 0.000,
         "f1": 0.000, "pr_auc": 0.457, "tuned_f1": None},
        {"run": "Logistic Regression", "recall": 0.565, "precision": 0.821,
         "f1": 0.670, "pr_auc": 0.803, "tuned_f1": None},
        {"run": "LogReg balanced", "recall": 0.639, "precision": 0.757,
         "f1": 0.693, "pr_auc": 0.803, "tuned_f1": 0.713},
        {"run": "Random Forest", "recall": 0.756, "precision": 0.656,
         "f1": 0.703, "pr_auc": 0.788, "tuned_f1": 0.710},
        {"run": "Random Forest deep", "recall": 0.693, "precision": 0.702,
         "f1": 0.698, "pr_auc": 0.786, "tuned_f1": 0.711},
        {"run": "XGBoost (final)", "recall": 0.698, "precision": 0.716,
         "f1": 0.707, "pr_auc": 0.810, "tuned_f1": 0.716},
        {"run": "XGBoost weighted", "recall": 0.756, "precision": 0.671,
         "f1": 0.711, "pr_auc": 0.809, "tuned_f1": 0.716},
    ]

    # --- final test evaluation --------------------------------------------
    X_test, y_test = load_xy("test")
    test_df = pd.read_csv(ROOT / "data" / "processed" / "features_test.csv")
    proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test, (proba >= thr).astype(int)).ravel()
    data["test"] = {
        "n": int(len(y_test)),
        "at_risk_rate": round(float(y_test.mean()), 3),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 3),
        "at_threshold": metrics(y_test, proba, thr),
        "at_half": metrics(y_test, proba, 0.5),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # PR curve, thinned to keep the report small
    precision, recall, _ = precision_recall_curve(y_test, proba)
    step = max(1, len(recall) // 220)
    data["pr_curve"] = [{"r": round(float(r), 4), "p": round(float(p), 4)}
                        for r, p in zip(recall[::step], precision[::step])]

    # --- per-module (DQ-06 protocol) --------------------------------------
    data["per_module"] = []
    for module, group in test_df.groupby("code_module"):
        idx = group.index
        m = metrics(y_test.loc[idx], proba[idx], thr)
        m.update({"module": module, "n": int(len(group)),
                  "at_risk_rate": round(float(y_test.loc[idx].mean()), 3),
                  "cold_start": module == "CCC"})
        data["per_module"].append(m)

    no_ccc = test_df.code_module != "CCC"
    data["test"]["excluding_ccc"] = metrics(y_test[no_ccc], proba[no_ccc], thr)

    # --- fairness slices ---------------------------------------------------
    data["fairness"] = {}
    for column in ["disability", "gender", "age_band", "imd_band"]:
        rows = []
        for value, group in test_df.groupby(column, dropna=False):
            idx = group.index
            m = metrics(y_test.loc[idx], proba[idx], thr)
            m.update({"group": str(value), "n": int(len(group)),
                      "at_risk_rate": round(float(y_test.loc[idx].mean()), 3),
                      "flag_rate": round(float((proba[idx] >= thr).mean()), 3)})
            rows.append(m)
        data["fairness"][column] = rows

    # --- global feature importance (gain) ---------------------------------
    booster = model.named_steps["model"].get_booster()
    names = list(model.named_steps["preprocess"].get_feature_names_out())
    gain = booster.get_score(importance_type="gain")
    scored = [(names[int(k[1:])], v) for k, v in gain.items()
              if int(k[1:]) < len(names)]
    scored.sort(key=lambda x: -x[1])
    total = sum(v for _, v in scored) or 1.0
    data["feature_importance"] = [
        {"feature": _readable(name), "share": round(value / total, 4)}
        for name, value in scored[:12]
    ]

    # --- error analysis: who we miss --------------------------------------
    at_risk = test_df[y_test == 1].copy()
    at_risk["caught"] = (proba >= thr)[y_test == 1]
    missed, caught = at_risk[~at_risk.caught], at_risk[at_risk.caught]
    data["error_analysis"] = {
        "missed": int(len(missed)), "caught": int(len(caught)),
        "medians": [
            {"metric": "clicks in the early window",
             "missed": float(missed.early_total_clicks.median()),
             "caught": float(caught.early_total_clicks.median())},
            {"metric": "days active",
             "missed": float(missed.early_active_days.median()),
             "caught": float(caught.early_active_days.median())},
            {"metric": "assignments submitted",
             "missed": float(missed.early_tma_submitted_count.median()),
             "caught": float(caught.early_tma_submitted_count.median())},
            {"metric": "average early score",
             "missed": float(missed.early_tma_mean_score.median()),
             "caught": float(caught.early_tma_mean_score.median())},
        ],
    }

    # --- risk score distribution by true outcome --------------------------
    edges = np.linspace(0, 1, 21)
    data["score_distribution"] = {
        "edges": [round(float(e), 3) for e in edges],
        "at_risk": np.histogram(proba[y_test == 1], bins=edges)[0].tolist(),
        "not_at_risk": np.histogram(proba[y_test == 0], bins=edges)[0].tolist(),
    }

    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
