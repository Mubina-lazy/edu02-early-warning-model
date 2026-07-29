"""Build the early-window feature matrix for each split.

Every feature uses ONLY information available on/before the cutoff day
(25% of the module-presentation length) - the rules live in src/config.py
and are enforced here with explicit filters plus a final self-check.

No statistics are learned here (no imputation, no scaling, no encoding):
everything in this file is a row-wise transformation, so it cannot leak
information between splits. Learned preprocessing happens later, inside
the sklearn Pipeline that is fitted on the training data only.

Input : data/processed/split_{train,validation,test}.csv (from make_split.py)
Output: data/processed/features_{train,validation,test}.csv

Usage:  python src/features.py
"""

from pathlib import Path

import pandas as pd

import config

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

KEY = ["code_module", "code_presentation", "id_student"]

CATEGORICAL = [
    "code_module", "gender", "region", "highest_education",
    "imd_band", "age_band", "disability",
]
NUMERIC = [
    "early_total_clicks", "early_active_days", "early_clicks_per_active_day",
    "days_since_last_activity", "early_tma_due_count",
    "early_tma_submitted_count", "early_tma_any_submitted",
    "early_tma_mean_score", "date_registration",
    "num_of_prev_attempts", "studied_credits",
]


def load_cutoffs() -> pd.DataFrame:
    courses = pd.read_csv(RAW / "courses.csv")
    courses["cutoff_day"] = (
        (config.EARLY_WINDOW_FRACTION * courses.module_presentation_length)
        .round()
        .astype(int)
    )
    return courses[["code_module", "code_presentation", "cutoff_day"]]


def vle_features(cutoffs: pd.DataFrame) -> pd.DataFrame:
    """Click activity aggregated over days <= cutoff only."""
    vle = pd.read_csv(
        RAW / "studentVle.csv",
        dtype={"id_student": "int32", "id_site": "int32",
               "date": "int16", "sum_click": "int16"},
    )
    vle = vle.merge(cutoffs, on=["code_module", "code_presentation"])
    vle = vle[vle.date <= vle.cutoff_day]  # leakage filter: early window only

    agg = vle.groupby(KEY).agg(
        early_total_clicks=("sum_click", "sum"),
        early_active_days=("date", "nunique"),
        last_active_day=("date", "max"),
    ).reset_index()
    return agg


def tma_features(cutoffs: pd.DataFrame) -> pd.DataFrame:
    """Scores of TMAs that were both DUE and SUBMITTED by the cutoff."""
    assessments = pd.read_csv(RAW / "assessments.csv")
    assessments = assessments.merge(cutoffs, on=["code_module", "code_presentation"])
    early = assessments[
        assessments.assessment_type.isin(config.ALLOWED_ASSESSMENT_TYPES)
        & (assessments.date <= assessments.cutoff_day)  # due by cutoff
    ]

    due_count = (early.groupby(["code_module", "code_presentation"])
                 .size().rename("early_tma_due_count").reset_index())

    subs = pd.read_csv(RAW / "studentAssessment.csv")
    subs = subs.merge(
        early[["id_assessment", "code_module", "code_presentation", "cutoff_day"]],
        on="id_assessment",
    )
    if config.EXCLUDE_BANKED:
        subs = subs[~subs.is_banked.astype(bool)]
    # submitted by cutoff: a score the advisor could actually see that day
    subs = subs[subs.date_submitted <= subs.cutoff_day]

    per_student = subs.groupby(KEY).agg(
        early_tma_submitted_count=("id_assessment", "nunique"),
        early_tma_mean_score=("score", "mean"),
    ).reset_index()
    return per_student, due_count


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic label fixes decided in the audit (DQ-01, DQ-02)."""
    df = df.copy()
    df["imd_band"] = df.imd_band.replace({"10-20": "10-20%"}).fillna("Missing")
    df["disability"] = df.disability.astype(str)
    return df


def build_split(split_name: str, cutoffs, vle, tma, due_count) -> pd.DataFrame:
    members = pd.read_csv(PROCESSED / f"split_{split_name}.csv")
    info = clean_categoricals(pd.read_csv(RAW / "studentInfo.csv"))
    reg = pd.read_csv(RAW / "studentRegistration.csv")

    df = (members
          .merge(info.drop(columns=["final_result"]), on=KEY, validate="one_to_one")
          .merge(reg[KEY + ["date_registration"]], on=KEY)
          .merge(cutoffs, on=["code_module", "code_presentation"])
          .merge(vle, on=KEY, how="left")
          .merge(tma, on=KEY, how="left")
          .merge(due_count, on=["code_module", "code_presentation"], how="left"))

    # Students with no early clicks simply have no studentVle rows -> zeros.
    df["early_total_clicks"] = df.early_total_clicks.fillna(0).astype(int)
    df["early_active_days"] = df.early_active_days.fillna(0).astype(int)
    df["early_clicks_per_active_day"] = (
        df.early_total_clicks
        .div(df.early_active_days.where(df.early_active_days > 0))
        .fillna(0.0)
    )
    # NaN when the student was never active; the Pipeline imputes it later.
    df["days_since_last_activity"] = df.cutoff_day - df.last_active_day

    # DQ-03/DQ-05: "no TMA submitted" is a state, not a zero score.
    df["early_tma_due_count"] = df.early_tma_due_count.fillna(0).astype(int)
    df["early_tma_submitted_count"] = df.early_tma_submitted_count.fillna(0).astype(int)
    df["early_tma_any_submitted"] = (df.early_tma_submitted_count > 0).astype(int)
    # early_tma_mean_score stays NaN when nothing was submitted (imputed later).

    # code_module is both an identifier and a feature - select it only once.
    feature_cols = [c for c in CATEGORICAL if c not in KEY] + NUMERIC
    out = df[KEY + [config.TARGET_COLUMN] + feature_cols]

    # Self-check: no banned column may survive into the feature matrix.
    banned_present = config.BANNED_COLUMNS & set(out.columns) - {"id_student"}
    assert not banned_present, f"leakage: banned columns in output: {banned_present}"

    out.to_csv(PROCESSED / f"features_{split_name}.csv", index=False)
    print(f"features_{split_name}.csv  rows={len(out):,}  "
          f"features={len(CATEGORICAL) + len(NUMERIC)}  "
          f"at_risk_rate={out[config.TARGET_COLUMN].mean():.3f}")
    return out


def main() -> None:
    cutoffs = load_cutoffs()
    print("Aggregating VLE clicks (10.6M rows, early window only) ...")
    vle = vle_features(cutoffs)
    tma, due_count = tma_features(cutoffs)
    for split in ["train", "validation", "test"]:
        build_split(split, cutoffs, vle, tma, due_count)


if __name__ == "__main__":
    main()
