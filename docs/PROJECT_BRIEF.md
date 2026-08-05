# Project Brief (approved scope)

Field-Based Scenario Track, scenario **EDU-02: Student Performance
Early-Warning Model** (EdTech). The official scenario brief is the project
specification; this file records the technical formulation I proposed and had
approved before implementation, so a reviewer can check the delivered project
against the scope it was meant to fill.

## Client background and business problem

An education provider wants to identify students who may need support before
the end of a course. Staff can see early attendance, LMS activity, quizzes and
assignments, but intervention usually happens only after final performance has
already declined. The business problem is an earlier signal of academic risk,
while there is still time to intervene - and the central challenge is choosing
a useful prediction point without using information that would not exist yet.

## Data and problem discovery

| Decision | Answer |
|---|---|
| Selected dataset and source | OULAD (Open University Learning Analytics Dataset): 7 modules, 22 module-presentations, 32,593 student enrolments. CC-BY 4.0, `analyse.kmi.open.ac.uk/open_dataset` |
| What one record represents | One student's data for one module-presentation: demographics, VLE activity and early assessment results for that specific run of a course |
| Proposed target | Binary risk classification: at risk (Withdrawn or Fail) against not at risk (Pass or Distinction), simplified from the four-class `final_result` for actionability |
| Information available at prediction time | Demographics (age band, gender, region, highest education, disability), registration date, VLE click activity in the first ~25% of the course, and already-due TMA scores |
| Main data quality issues | Missing `imd_band`; missing scores for unsubmitted assessments; class imbalance; students who register but never engage |
| Potential leakage risks | `final_result`-derived features, TMA scores submitted after the cutoff, whole-course VLE click totals - all excluded |
| Privacy, fairness, licensing | Anonymized and released by the Open University under CC-BY 4.0. Fairness risk: disability and region could cause systematic flagging of certain groups; subgroup metrics checked separately |

## Technical proposal

| Decision | Answer |
|---|---|
| ML problem formulation | Supervised binary classification, using only early-course information |
| Proposed baseline | DummyClassifier (majority class) and Logistic Regression on early features |
| Main approaches to investigate | Random Forest and Gradient Boosting (XGBoost), compared against each other and the baseline |
| Data splitting strategy | Split by module-presentation (train on earlier presentations, test on a later one) rather than randomly, to avoid mixing the same student across sets and to reflect time-based evaluation |
| Primary metrics and why | Recall and F1, with PR-AUC supporting - missing an at-risk student is costlier than an unnecessary check-in |
| Expected inference input | A student's demographics plus activity and early-assessment data for the first weeks of a course |
| Expected inference output | Risk probability (0-1), a risk band (Low/Medium/High), and the top 2-3 contributing factors via feature importance, for academic advisors |
| Main risks and assumptions | UK Open University data from 2013-2014; it does not transfer directly to Uzbekistani universities. A methodology-demonstrating prototype, not a production system - stated clearly in the README |

## Delivered against the scope

| Scope commitment | Where it is delivered |
|---|---|
| Binary at-risk target | `src/config.py`, README section 5 |
| Early prediction point, no post-cutoff data | `src/config.py` (25% cutoff, banned columns), `src/features.py` |
| Presentation-based split | `src/make_split.py`, `reports/split_summary.csv` |
| Dummy + Logistic Regression baselines | `src/train_baseline.py`, `reports/baseline_results.md` |
| Random Forest and XGBoost compared | `src/train_experiments.py`, `reports/experiments_results.md` |
| Recall and F1 primary, PR-AUC supporting | README section 9, `reports/final_evaluation.md` |
| Risk probability, band, and top factors | `src/predict.py` (`predict_risk`, `top_factors` via TreeSHAP) |
| Limitation stated in the README | README section 14, `docs/RESPONSIBLE_AI_AND_LIMITATIONS.md` |

## Controlled changes to the scope

One decision was added after the audit and endorsed by the mentor before the
Model Gate: **DQ-06**, how to treat module CCC, which exists only in the 2014
presentations and therefore has no training data. Agreed protocol: keep CCC as
a realistic cold-start case, and report headline metrics both with and without
it plus a per-module table. Recorded in `docs/issue_log.csv`.
