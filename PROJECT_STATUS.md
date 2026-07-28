# Project Status

## Project

Student Performance Early-Warning Model (EDU-02, Field-Based Scenario Track, OULAD dataset)

## Current stage

Data Gate complete — awaiting mentor review before Model Gate (course route step 3 → 4 boundary)

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

Data Gate review with mentor. Gate self-assessment: **Yellow** — evidence is
complete; one named open decision (DQ-06: module CCC has no 2013 training
data — keep with per-module reporting vs exclude from headline metric).

## Next

- Review Data Gate evidence with mentor; resolve DQ-06.
- On Green: start Model Gate (course Class 4) — early-window feature
  building (`src/features.py`), baseline DummyClassifier + Logistic
  Regression, MLflow tracking, then Random Forest and XGBoost.
- **No modeling before the mentor review** (explicit project rule).

## Known problems / blockers

- DQ-06 (mentor decision pending) — see `docs/issue_log.csv`.
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
