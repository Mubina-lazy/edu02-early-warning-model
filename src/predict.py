"""Reusable inference for the early-warning model.

Turns one student's early-course data (a plain dict) into a risk
assessment: probability, Low/Medium/High band, flag decision, and
simple human-readable signals for the advisor.

Used by demo.ipynb; can also be imported by an API later.

    from predict import load_model, predict_risk
    model, meta = load_model()
    result = predict_risk(student_dict, model, meta)
"""

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# field name -> (type, min, max, required). None max = unbounded.
INPUT_SCHEMA = {
    "code_module": (str, None, None, True),
    "gender": (str, None, None, True),
    "region": (str, None, None, True),
    "highest_education": (str, None, None, True),
    "imd_band": (str, None, None, True),
    "age_band": (str, None, None, True),
    "disability": (str, None, None, True),
    "early_total_clicks": (float, 0, None, True),
    "early_active_days": (float, 0, 70, True),
    "early_clicks_per_active_day": (float, 0, None, False),
    "days_since_last_activity": (float, 0, 100, False),  # None = never active
    "early_tma_due_count": (float, 0, 10, True),
    "early_tma_submitted_count": (float, 0, 10, True),
    "early_tma_any_submitted": (float, 0, 1, False),
    "early_tma_mean_score": (float, 0, 100, False),  # None = nothing submitted
    "date_registration": (float, -400, 100, False),
    "num_of_prev_attempts": (float, 0, 10, True),
    "studied_credits": (float, 0, 700, True),
}
VALID_MODULES = {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"}


def check_versions(meta: dict) -> list[str]:
    """Warn if the environment differs from the one that fitted the model.

    A pickled scikit-learn pipeline is only guaranteed to reproduce its
    predictions under the version it was fitted with, so we surface a clear
    message instead of scoring silently on a mismatched library.
    """
    import sklearn
    import xgboost

    installed = {"scikit-learn": sklearn.__version__,
                 "xgboost": xgboost.__version__}
    expected = meta.get("fitted_with", {})
    return [
        f"{pkg}: installed {installed[pkg]}, model fitted with {expected[pkg]}"
        for pkg in installed
        if pkg in expected and installed[pkg] != expected[pkg]
    ]


def load_model(strict: bool = False):
    """Load the final pipeline and its metadata.

    strict=True raises on a version mismatch; the default only warns, so a
    demo still runs in an environment with slightly different libraries.
    """
    model = joblib.load(ROOT / "models" / "final_model.joblib")
    meta = json.loads((ROOT / "models" / "final_model_meta.json").read_text())

    mismatches = check_versions(meta)
    if mismatches:
        message = (
            "Library versions differ from the ones that fitted the model:\n  "
            + "\n  ".join(mismatches)
            + "\nInstall the pinned versions to reproduce the documented "
              "results:\n  pip install -r requirements-demo.txt\n"
              "(in Colab, restart the runtime after installing)"
        )
        if strict:
            raise RuntimeError(message)
        print("WARNING: " + message)
    return model, meta


def validate_input(student: dict) -> list[str]:
    """Return a list of problems (empty list = input is usable)."""
    errors = []
    for field, (ftype, lo, hi, required) in INPUT_SCHEMA.items():
        if field not in student or student[field] is None:
            if required:
                errors.append(f"missing required field: '{field}'")
            continue
        value = student[field]
        if ftype is float:
            try:
                value = float(value)
            except (TypeError, ValueError):
                errors.append(f"'{field}' must be a number, got {value!r}")
                continue
            if lo is not None and value < lo:
                errors.append(f"'{field}' = {value} is below the minimum {lo}")
            if hi is not None and value > hi:
                errors.append(f"'{field}' = {value} is above the maximum {hi}")
    if student.get("code_module") not in VALID_MODULES:
        errors.append(f"'code_module' must be one of {sorted(VALID_MODULES)}")
    subs = student.get("early_tma_submitted_count") or 0
    due = student.get("early_tma_due_count") or 0
    if float(subs) > float(due):
        errors.append("submitted TMA count cannot exceed the number due")
    return errors


def _signals(student: dict) -> list[str]:
    """Simple rule-based explanations an advisor can act on."""
    out = []
    if float(student.get("early_total_clicks") or 0) == 0:
        out.append("no online activity at all in the early window")
    elif float(student.get("early_total_clicks") or 0) < 100:
        out.append("very low online activity (bottom quartile)")
    d = student.get("days_since_last_activity")
    if d is not None and float(d) > 21:
        out.append(f"inactive for {float(d):.0f} days at the check point")
    if float(student.get("early_tma_due_count") or 0) > 0 and \
            float(student.get("early_tma_submitted_count") or 0) == 0:
        out.append("has not submitted any assignment that was already due")
    score = student.get("early_tma_mean_score")
    if score is not None and float(score) < 55:
        out.append(f"low early assignment average ({float(score):.0f})")
    if float(student.get("num_of_prev_attempts") or 0) > 0:
        out.append("has previous unsuccessful attempts at this course")
    return out or ["no obvious warning signals - risk driven by weaker patterns"]


READABLE_NAMES = {
    "early_total_clicks": "clicks in the early window",
    "early_active_days": "days active in the early window",
    "early_clicks_per_active_day": "clicks per active day",
    "days_since_last_activity": "days since last activity",
    "early_tma_due_count": "assignments already due",
    "early_tma_submitted_count": "assignments submitted",
    "early_tma_any_submitted": "submitted at least one assignment",
    "early_tma_mean_score": "average early assignment score",
    "date_registration": "how early the student registered",
    "num_of_prev_attempts": "previous attempts at this course",
    "studied_credits": "credits being studied",
}


def _readable(feature: str) -> str:
    """Turn a pipeline feature name into something an advisor can read."""
    name = feature.split("__", 1)[-1]          # drop the 'num__'/'cat__' prefix
    if name.startswith("missingindicator_"):   # imputer indicator columns
        base = name[len("missingindicator_"):]
        # the score indicator is really "nothing submitted yet" - say that
        if base == "early_tma_mean_score":
            return "no assignment submitted yet"
        return f"{READABLE_NAMES.get(base, base)} (not recorded)"
    if name in READABLE_NAMES:
        return READABLE_NAMES[name]
    # one-hot columns look like 'imd_band_20-30%' -> 'imd_band = 20-30%'
    for column in ("code_module", "gender", "region", "highest_education",
                   "imd_band", "age_band", "disability"):
        if name.startswith(column + "_"):
            return f"{column} = {name[len(column) + 1:]}"
    return name


def top_factors(row_df, model, k: int = 3) -> list[dict]:
    """Per-student contributing factors from the model itself.

    Uses XGBoost's built-in TreeSHAP (`pred_contribs=True`), so these are the
    model's actual contributions for this one student, not a global ranking
    and not hand-written rules. Positive values push the risk up.
    """
    import xgboost as xgb

    preprocess = model.named_steps["preprocess"]
    booster = model.named_steps["model"].get_booster()

    matrix = preprocess.transform(row_df)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    names = list(preprocess.get_feature_names_out())

    # last column of the SHAP output is the base value, not a feature
    contributions = booster.predict(xgb.DMatrix(matrix), pred_contribs=True)[0][:-1]

    order = sorted(range(len(contributions)),
                   key=lambda i: abs(contributions[i]), reverse=True)
    factors = []
    for i in order[:k]:
        factors.append({
            "factor": _readable(names[i]),
            "direction": "increases risk" if contributions[i] > 0 else "lowers risk",
            "contribution": round(float(contributions[i]), 3),
        })
    return factors


def predict_risk(student: dict, model=None, meta=None) -> dict:
    """Validate the input and return the risk assessment for one student."""
    errors = validate_input(student)
    if errors:
        raise ValueError("invalid input: " + "; ".join(errors))
    if model is None or meta is None:
        model, meta = load_model()

    row = {f: student.get(f) for f in INPUT_SCHEMA}
    # derived field the caller may omit
    if row.get("early_clicks_per_active_day") is None:
        days = float(row["early_active_days"]) or 0
        row["early_clicks_per_active_day"] = (
            float(row["early_total_clicks"]) / days if days else 0.0
        )
    if row.get("early_tma_any_submitted") is None:
        row["early_tma_any_submitted"] = float(
            float(row["early_tma_submitted_count"]) > 0
        )

    row_df = pd.DataFrame([row])
    proba = float(model.predict_proba(row_df)[:, 1][0])
    threshold = meta["threshold"]
    band = "High" if proba >= 0.6 else ("Medium" if proba >= threshold else "Low")
    return {
        "risk_probability": round(proba, 3),
        "risk_band": band,
        "flagged_for_advisor": proba >= threshold,
        "decision_threshold": round(threshold, 3),
        # what the model itself weighted for this student (TreeSHAP)
        "top_factors": top_factors(row_df, model, k=3),
        # plain-language observations from the raw input, for the advisor
        "signals": _signals(student),
        "note": "Decision-support only: an advisor must review every flag.",
    }
