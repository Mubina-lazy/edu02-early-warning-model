# Project Status

## Project

Student Performance Early-Warning Model (EDU-02, Field-Based Scenario Track, OULAD dataset)

## Current stage

Complete — ready for submission preparation and defense rehearsal

## Completed

- **Scope** (course step 1): approved EDU-02 brief — binary at-risk target,
  early prediction point (25% of the course), presentation-based split,
  Recall/F1 primary metrics.
- **Repository** (step 2): 16-section README, reproducible scripts,
  pinned dependencies, no raw data in Git.
- **Data Gate** (step 3): full audit with written conclusions
  (`docs/data_audit.md`, `notebooks/01_data_audit_eda.ipynb`), issue log
  DQ-01…DQ-08, leakage register + `src/config.py` controls,
  presentation-based split (train 2013B+2013J / validation 2014B / test
  2014J). Mentor review: "a very strong and well-organized Data Gate" —
  Green after three applied corrections (CCC protocol, DQ-07 counts,
  pyreadr dependency).
- **Model Gate** (step 4): 18 leakage-safe early-window features
  (`src/features.py`); 7 MLflow runs — 2 baselines
  (`src/train_baseline.py`) + 5 hypothesis-driven experiments
  (`src/train_experiments.py`); final model = XGBoost with operating
  threshold 0.327 (validation F1 0.716, PR-AUC 0.810); selection evidence
  in `reports/experiments_results.md`.
- **Final evaluation** (one-shot, frozen test 2014J):
  recall 0.760 / precision 0.583 / F1 0.660 / PR-AUC 0.714 at the operating
  threshold; DQ-06 protocol honored (with/without CCC + per-module table);
  fairness slices and false-negative error analysis in
  `reports/final_evaluation.md`.
- **Inference & demo** (step 5): `src/predict.py` (validated inputs, risk
  bands, advisor signals) + executed `demo.ipynb` (Colab-ready, no dataset
  download needed — the <1 MB model artifact ships with the repo).
- **Verification pass**: pipeline re-run end-to-end — deterministic outputs,
  zero row overlap between splits, no banned columns, docs consistent with
  the data.
- **Clean-runtime reproduction test (passed)**: `demo.ipynb` opened from
  GitHub in a fresh Google Colab runtime and run top to bottom. It reproduced
  the documented predictions exactly — 0.889 (High) for the disengaged
  student, 0.122 (Low) for the engaged one, 0.964 (High) for the
  zero-activity edge case — and rejected all five invalid inputs with clear
  messages. Two defects this test surfaced were fixed: an unsatisfiable
  `requirements.txt` (mlflow requires pandas<3) and a setup cell that
  silently reused a stale clone.

## Current task

Submission preparation (course step 6): freeze the repository, prepare the
LMS submission file (full name, track, title, repo URL, short description),
rehearse the defense.

## Next

- Grant mentors repository access (repo is private) before the deadline.
- Prepare the LMS PDF/DOCX and defense slides; rehearse the live demo.
- Optional extensions (explicitly out of scope): periodic re-scoring at
  later checkpoints, advisor dashboard.

## Known problems / blockers

- None open.
- Sandbox note: the official OULAD URL is blocked from the cloud
  environment used during development, so data there was fetched with
  `--source mirror` (the dataset authors' own GitHub package) and validated
  against published row counts. In Colab/local runs the default official
  source works.

## Defense cheat-sheet (the five decisions to be able to explain)

1. **Target**: at risk = Withdrawn or Fail — advisors need one actionable
   binary signal; both outcomes are preventable failures.
2. **Prediction point**: 25% of each presentation (~day 60–67) — early
   enough to intervene; everything after that day is banned as leakage
   (`src/config.py`).
3. **Population (DQ-04)**: students already unregistered by the cutoff are
   excluded — their outcome is known at prediction time, "predicting" them
   would inflate recall.
4. **Split**: whole future cohorts held out (train 2013 → test 2014J) —
   mirrors real deployment; a random split would leak repeated students and
   cohort effects.
5. **Metrics & threshold**: Recall/F1 over accuracy (a do-nothing model
   gets 54–62% accuracy with zero recall); threshold 0.327 tuned on
   validation because missing an at-risk student costs more than an extra
   advisor check-in.
