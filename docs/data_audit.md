# Data Audit and Leakage-Safe Pipeline

## 1. Project and data context
- **Project:** Student Performance Early-Warning Model (EDU-02, Field-Based Scenario)
- **Project type:** tabular (with time/group structure across module-presentations)
- **Dataset source:** [data/README.md](../data/README.md) — OULAD, CC-BY 4.0
- **One row/example represents:** one student registered in one module-presentation
  (key: `code_module` + `code_presentation` + `id_student`)
- **Target or objective:** binary — **at risk** (`final_result` ∈ {Withdrawn, Fail})
  vs **not at risk** ({Pass, Distinction})
- **Prediction/evaluation moment:** the day 25% of the module-presentation has
  elapsed (day 58–67 depending on the run). Only information dated on/before that
  day may be used. Population: students **still enrolled** at that moment.

## 2. Audit summary

Evidence notebook: [`notebooks/01_data_audit_eda.ipynb`](../notebooks/01_data_audit_eda.ipynb)
(sections A–F, executed outputs included).

| Area | Finding | Evidence path/output | Decision consequence |
|---|---|---|---|
| Structure | 7 tables, clean keys; 22 module-presentations, 234–269 days; all dates relative to course start (day 0). Not every module runs every presentation. | notebook §A | Early window definable per presentation as 25% of its length; split must respect module availability (see DQ-06) |
| Missingness | `imd_band` 1,111 (3.4%); `date_registration` 45; `score` 173; 11 assessments without date (all final Exams) | notebook §B | `imd_band` → explicit "Missing" category (informative); exams excluded anyway; unsubmitted TMAs need a "not submitted" indicator |
| Duplicates/groups | 0 duplicate keys in any table; 3,538 students repeat across module-presentations | notebook §B, §E | No dedup needed; repeated students make a random split dishonest → presentation-based split |
| Distribution/balance | At-risk 52.8% of all rows; 41.4% of the still-enrolled-at-cutoff population; varies by module (AAA 29% → CCC 62%) | notebook §C, §F | Moderate imbalance → Recall/F1 primary (per brief); per-module metrics required at evaluation |
| Time/order | Presentations are ordered cohorts (2013B → 2014J); 43.8% of VLE clicks fall in the early window; zero early clicks → 98.7% at risk | notebook §E, §F | Time-aware split (train 2013, validate 2014B, test 2014J); early engagement features carry real signal |
| Fairness/privacy | Anonymized by publisher. Base rates differ: disability 62% vs 52%; imd_band gradient 65%→43%; missing imd_band group lowest (34%) | notebook §D | Subgroup metrics (disability, imd_band) mandatory at evaluation; advisor-support use only, no punitive use |

## 3. Data-quality issue log

Machine-readable copy: [`docs/issue_log.csv`](issue_log.csv).

| ID | Finding | Evidence | Risk | Decision | Action or accepted limitation | Status |
|---|---|---|---|---|---|---|
| DQ-01 | `imd_band` missing for 1,111 rows (3.4%) and missingness is informative (at-risk 34% vs 53% overall) | notebook §B, §D | Silent imputation would erase signal and could distort fairness slices | Keep rows; encode missing explicitly | "Missing" becomes its own category in preprocessing | Controlled |
| DQ-02 | `imd_band` label `10-20` lacks the `%` suffix (all other bands have it) | notebook §B | Category treated as distinct from a hypothetical `10-20%`; ugly slices | Normalize label | Rename to `10-20%` in preprocessing | Controlled |
| DQ-03 | 173 submitted assessments have no `score`; unsubmitted TMAs simply have no row | notebook §B | Early-TMA features would silently become NaN/0 and mix "not submitted" with "scored 0" | Explicit indicator | Early-TMA features get a paired "submitted yes/no" flag; missing score ≠ 0 | Controlled |
| DQ-04 | 6,352 students (19.5%) unregistered on/before the cutoff day; their outcome is already known at prediction time | notebook §F | Keeping them inflates recall with "predictions" of already-known outcomes (answer-key effect) | Exclude from modeling population | Population = still enrolled at cutoff; implemented in `src/make_split.py`; at-risk rate becomes 41.4% | Resolved |
| DQ-05 | GGG-2014B has **zero** TMAs due by the cutoff (all other module-presentations have 1–3) | notebook §F | Early-TMA features undefined for one course run | Tolerate absence | Features must handle "no TMA due yet" as a valid state (indicator + neutral value), not an error | Controlled |
| DQ-06 | Module CCC exists only in 2014B/2014J → no CCC data in the 2013 training era | notebook §E | Model sees CCC students in validation/test with zero CCC training examples; per-module test scores will be confounded | Flag for mentor at gate review | Recommended default: keep CCC, report per-module metrics, discuss excluding CCC from headline metric | **Open (mentor)** |
| DQ-07 | 972 students (~9% of test) appear in both training-era and test-era presentations | notebook §E | Weak group leakage: same person on both sides (different course runs) | Accept with evidence | Returning students also exist in real deployment; features are presentation-specific; documented, not removed | Accepted |
| DQ-08 | 45 rows missing `date_registration`; 8 rows unregistered before cutoff but marked Fail (not Withdrawn) | notebook §B, §F | Negligible volume; odd records could confuse features | Keep with defaults | Missing registration date → impute + flag; the 8 odd rows are excluded by the DQ-04 rule anyway | Controlled |

## 4. Split decision
- **Chosen strategy:** chronological, by module-presentation (time-aware, group-preserving):
  **train = 2013B + 2013J, validation = 2014B, test = 2014J.**
- **What must remain genuinely unseen:** future cohorts — an entire later course run,
  never individual rows sampled from a run the model trained on.
- **Why this matches real use:** in deployment the university trains on completed past
  presentations and predicts for the cohort currently running. A random split would put
  the same presentation (and often the same student, see DQ-07) in both train and test,
  overstating performance.
- **Implementation path:** [`src/make_split.py`](../src/make_split.py)
  (deterministic — no randomness needed; population rule from DQ-04 applied first).
- **Verification output:** [`reports/split_summary.csv`](../reports/split_summary.csv) —
  row counts, at-risk rates, presentation date ranges, student-overlap counts per split;
  also printed by the script on every run.

## 5. Leakage risks and controls

Banned columns/filters are encoded once in [`src/config.py`](../src/config.py) and
reused by all later feature-building code.

| Risk | Why it could leak | Control | Verification evidence | Severity |
|---|---|---|---|---|
| `final_result` (or anything derived from it) used as a feature | It *is* the target | Used only to build the label; listed in `BANNED_COLUMNS` | `src/config.py`; feature code reads only allowed tables | Critical |
| `date_unregistration` as a feature | Directly reveals withdrawal (target proxy) | Never a feature; used **only** to apply the DQ-04 population rule with unregistrations known by the cutoff (`date_unregistration <= cutoff`) | `src/make_split.py`, `src/config.py` | Critical |
| Assessment scores due or submitted after the cutoff (incl. all Exams) | Future performance information | Double date filter: `assessments.date <= cutoff` AND `date_submitted <= cutoff` | filter constants in `src/config.py`; notebook §F shows early-TMA availability | High |
| VLE clicks after the cutoff / whole-course click totals | Future engagement information; totals encode course completion | All VLE aggregation filtered to `date <= cutoff`; no whole-course totals ever computed | notebook §F (early-window share); `src/config.py` | High |
| `is_banked` assessment results | Banked results are carried over from a previous presentation — not earned in this run's early window | Banked submissions excluded from early-TMA features | `src/config.py` (`EXCLUDE_BANKED`) | Medium |
| Fit-boundary leakage (imputer/encoder/scaler fitted on val/test) | Test statistics contaminate preprocessing | All learned preprocessing inside a sklearn `Pipeline`, fitted on train only (Model Gate) | `docs/preprocessing_manifest.json` (`fit_boundary`) | Medium |
| Same entity on both split sides | Repeated students across eras | Presentation split + overlap quantified (DQ-07) | notebook §E; `reports/split_summary.csv` | Medium |

## 6. Preprocessing design
- **Numerical handling:** early-window aggregates (clicks, active days, TMA scores,
  registration timing). Missing TMA score → indicator + neutral fill; median imputation
  for the rare missing `date_registration`. Scaling (for Logistic Regression) fitted on
  train only; tree models don't need it.
- **Categorical handling:** `gender`, `region`, `highest_education`, `imd_band`
  (normalized labels + "Missing" category), `age_band`, `disability`, `code_module` —
  one-hot encoding fitted on train only (`handle_unknown="ignore"` so unseen categories
  don't crash inference).
- **Text/image-specific handling:** n/a (tabular project).
- **Fit boundary:** every learned step (imputer, encoder, scaler) is fitted on the
  training presentations only, inside one sklearn `Pipeline`/`ColumnTransformer`;
  validation/test only get `transform`.
- **Reusable implementation path:** planned for Model Gate as `src/features.py`
  (early-window feature building from raw tables) + `src/preprocessing.py`
  (sklearn Pipeline). Manifest: [`docs/preprocessing_manifest.json`](preprocessing_manifest.json).
- **Model-ready output/status:** split membership files exist
  (`data/processed/split_*.csv`, regenerated by `src/make_split.py`); the feature
  matrix itself is intentionally **not built yet** — feature building is the first
  Model Gate step, after this audit is reviewed. No data blocker.

## 7. Project-type notes

Tabular with group/time structure. The dataset joins on
(`code_module`, `code_presentation`, `id_student`); all event tables
(`studentVle`, `studentAssessment`) are day-indexed relative to course start,
which makes the leakage cutoff a single reproducible filter rather than a
judgment call. The identifier `id_student` is never a feature. Class balance
(41.4% positive in the modeling population) is mild enough that resampling is
not planned; metric choice (Recall/F1, PR-AUC) addresses the remaining
imbalance.

## 8. Data Gate status
- **Status:** **Yellow** — evidence complete, awaiting the mentor review the
  student explicitly scheduled, with one named open decision (DQ-06: CCC
  cold-start handling).
- **Evidence links:** `notebooks/01_data_audit_eda.ipynb`, `docs/issue_log.csv`,
  `reports/split_summary.csv`, `reports/figures/`, `src/config.py`,
  `src/make_split.py`, `data/README.md`, `PROJECT_STATUS.md`
- **Named correction/blocker:** decide DQ-06 (keep CCC with per-module
  reporting vs exclude CCC from headline metric) with the mentor before
  baseline work.
- **Owner and due point:** Mubina + mentor, at the Data Gate review (before
  course Class 4 / Model Gate).
- **Next action:** review this document and the notebook with the mentor;
  on Green, start Model Gate (baseline DummyClassifier + Logistic Regression,
  MLflow tracking) per the approved brief.
