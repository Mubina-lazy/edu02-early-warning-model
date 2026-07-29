# Project Status

## Project

Student Performance Early-Warning Model (EDU-02, Field-Based Scenario Track, OULAD dataset)

## Current stage

Model Gate (course route step 4) — features and baselines done; experiments (Random Forest, XGBoost) next

## Completed

- Scope and problem definition: approved Project Brief (EDU-02; binary
  at-risk target; early-course prediction point; presentation-based split;
  Recall/F1 primary metrics).
- Repository setup: 16-section README skeleton, AGENTS.md, requirements.txt,
  .gitignore, folder structure (`data/`, `notebooks/`, `src/`, `docs/`,
  `reports/`).
- Dataset acquisition: `src/download_data.py` (official site + authors'
  GitHub mirror fallback), all 7 tables' row counts validated against the
  OULAD paper; source, license, limitations in `data/README.md`.
- Data audit + EDA with written conclusions:
  `notebooks/01_data_audit_eda.ipynb` (executed, sections A–F) and
  `docs/data_audit.md`; figures in `reports/figures/`.
- Data-quality issue log: `docs/issue_log.csv` (DQ-01…DQ-08).
- Leakage register and controls: `docs/data_audit.md` §5 + machine-checkable
  constants in `src/config.py`.
- Preprocessing design (leakage-safe, fit-on-train-only):
  `docs/preprocessing_manifest.json`.
- Train/validation/test split by module-presentation:
  `src/make_split.py` → train 2013B+2013J (11,309 rows, 41.9% at risk),
  validation 2014B (6,186, 45.7%), test 2014J (8,746, 37.6%);
  verification in `reports/split_summary.csv`.

## Current task

Data Gate: **Green.** Mentor reviewed the evidence ("a very strong and
well-organized Data Gate") and approved moving on after three corrections,
all applied:

1. DQ-06 decided in advance — CCC is kept as a realistic cold-start course;
   headline metrics will be reported with and without CCC, plus a mandatory
   per-module table.
2. DQ-07 duplicate-student counts made consistent (raw 940/972 vs
   modeling-population 558/678; the latter, in `reports/split_summary.csv`,
   are authoritative).
3. `pyreadr` added to `requirements.txt` for the mirror download path.

Model Gate progress (step 1 done):

- Early-window feature matrices built (`src/features.py`, 18 features,
  leakage-safe row-wise transforms with a banned-column self-check).
- Baselines trained and logged to MLflow (`src/train_baseline.py`):
  Dummy majority = 0.0 at-risk recall at 54.3% accuracy (the accuracy trap,
  now measured); Logistic Regression = 0.565 recall / 0.821 precision /
  0.670 F1 / 0.803 PR-AUC on validation (2014B). Details in
  `reports/baseline_results.md`.

## Next

- Model Gate experiments: Random Forest and XGBoost, class weighting /
  threshold tuning to raise at-risk recall, several documented MLflow runs,
  then model selection with evidence.
- Test set (2014J) stays untouched until the final evaluation.

## Known problems / blockers

- None open.
- Sandbox note: the official OULAD URL is blocked by this cloud
  environment's network policy, so data here was fetched with
  `--source mirror` (the dataset authors' own GitHub package) and validated
  against published row counts. In Colab/local runs the default official
  source works.

## Data Gate checklist (M8C3)

| Done | Gate condition | Evidence |
|------|----------------|----------|
| ✅ | Data source and usage conditions documented | `data/README.md` |
| ✅ | Target/objective clear and matches project scope | `docs/data_audit.md` §1 |
| ✅ | EDA/data audit has written conclusions | `docs/data_audit.md` §2, `notebooks/01_data_audit_eda.ipynb` |
| ✅ | Data-quality issue log complete | `docs/data_audit.md` §3, `docs/issue_log.csv` |
| ✅ | Split strategy matches real use, visible proof | `src/make_split.py`, `reports/split_summary.csv` |
| ✅ | Leakage risks identified and controlled | `docs/data_audit.md` §5, `src/config.py` |
| ✅ | Preprocessing reusable or explicitly planned | `docs/preprocessing_manifest.json` |
| ✅ | Model-ready inputs exist or named blocker recorded | `docs/data_audit.md` §6/§8 (split files regenerable; features = first Model Gate step) |
| ✅ | PROJECT_STATUS.md and repository evidence current | this file |
| ✅ | Verified Data Gate commit visible | git history (`data: complete Data Gate audit and leakage-safe pipeline`) |
