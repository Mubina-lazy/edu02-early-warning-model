"""Build the train/validation/test split by module-presentation.

Strategy (approved brief + docs/data_audit.md section 4):
  train      = 2013B + 2013J   (earlier cohorts)
  validation = 2014B
  test       = 2014J           (latest cohort - final unseen evaluation only)

This is deliberately deterministic - no random numbers. The "unseen data"
boundary is time: whole future course runs, exactly like real deployment.

Population rule (issue DQ-04): students who unregistered on or before the
cutoff day (25% of the presentation length) are excluded everywhere -
at prediction time their outcome is already known, so predicting them
would inflate the metrics.

Outputs:
  data/processed/split_train.csv / split_validation.csv / split_test.csv
      (key columns + at_risk target - membership lists, not features)
  reports/split_summary.csv (verification evidence for the Data Gate)

Usage:  python src/make_split.py
"""

from pathlib import Path

import pandas as pd

import config

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

KEY = ["code_module", "code_presentation", "id_student"]


def build_population() -> pd.DataFrame:
    """Students still enrolled at the cutoff, with the binary target."""
    students = pd.read_csv(RAW / "studentInfo.csv")
    registration = pd.read_csv(RAW / "studentRegistration.csv")
    courses = pd.read_csv(RAW / "courses.csv")

    courses["cutoff_day"] = (
        (config.EARLY_WINDOW_FRACTION * courses.module_presentation_length)
        .round()
        .astype(int)
    )

    df = students.merge(registration, on=KEY, validate="one_to_one")
    df = df.merge(
        courses[["code_module", "code_presentation", "cutoff_day"]],
        on=["code_module", "code_presentation"],
    )

    df[config.TARGET_COLUMN] = (
        df.final_result.isin(config.AT_RISK_RESULTS).astype(int)
    )

    if config.EXCLUDE_UNREGISTERED_BY_CUTOFF:
        before = len(df)
        already_gone = df.date_unregistration <= df.cutoff_day
        df = df[~already_gone.fillna(False)]
        print(
            f"Population rule DQ-04: excluded {before - len(df):,} students "
            f"already unregistered by the cutoff ({before:,} -> {len(df):,})"
        )

    return df


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for p in config.TRAIN_PRESENTATIONS:
        mapping[p] = "train"
    for p in config.VALIDATION_PRESENTATIONS:
        mapping[p] = "validation"
    for p in config.TEST_PRESENTATIONS:
        mapping[p] = "test"
    df = df.copy()
    df["split"] = df.code_presentation.map(mapping)
    assert df.split.notna().all(), "every presentation must belong to a split"
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Verification evidence: sizes, rates, ranges, overlap checks."""
    train_students = set(df.loc[df.split == "train", "id_student"])
    rows = []
    for split in ["train", "validation", "test"]:
        part = df[df.split == split]
        overlap = (
            "" if split == "train"
            else f"{len(set(part.id_student) & train_students):,} students also in train"
        )
        rows.append({
            "split": split,
            "rows": len(part),
            "target_rate_or_objective_summary": f"at_risk rate {part.at_risk.mean():.3f}",
            "date_min": part.code_presentation.min(),
            "date_max": part.code_presentation.max(),
            "unique_groups": part.groupby(["code_module", "code_presentation"]).ngroups,
            "notes": (
                f"{part.id_student.nunique():,} unique students; "
                f"modules {sorted(part.code_module.unique())}"
                + (f"; {overlap}" if overlap else "")
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = assign_split(build_population())

    OUT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        part = df.loc[df.split == split, KEY + [config.TARGET_COLUMN]]
        path = OUT / f"split_{split}.csv"
        part.to_csv(path, index=False)
        print(f"wrote {path.relative_to(ROOT)}  ({len(part):,} rows)")

    summary = summarize(df)
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS / "split_summary.csv", index=False)
    print(f"\nwrote {(REPORTS / 'split_summary.csv').relative_to(ROOT)}:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
