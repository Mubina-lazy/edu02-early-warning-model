# Student Performance Early-Warning Model (EDU-02)

> **Reviewing this project?** Start with [START_HERE.md](START_HERE.md) for the
> recommended route, or [RUBRIC_EVIDENCE_MATRIX.md](RUBRIC_EVIDENCE_MATRIX.md)
> to find the evidence for each grading criterion.

An AI/ML Fundamentals capstone project: an early-warning classifier that flags
students at risk of failing or withdrawing from an online course, using only
information available in the first weeks of the course.

> **Project status:** complete - data audit, experiments, final evaluation on
> unseen data, and a runnable demo. See [PROJECT_STATUS.md](PROJECT_STATUS.md).

---

## 1. Project Title

**Student Performance Early-Warning Model** - predicting academic risk from
early-course data (OULAD).

## 2. Problem Statement

Universities and online academies usually notice struggling students only after
final grades are in - when it is too late to help. This project builds a binary
classifier that estimates, early in a course, whether a student is **at risk**
(will withdraw or fail) so that academic support staff can intervene while
there is still time. Missing an at-risk student is costlier than an unnecessary
check-in, so the model prioritizes recall.

## 3. Selected Project Track

**Field-Based Scenario Track - EDU-02 "Student Performance Early-Warning Model"
(EdTech).** The official scenario brief serves as the project specification.

## 4. Dataset Source

**OULAD - Open University Learning Analytics Dataset**

- Source: <https://analyse.kmi.open.ac.uk/open_dataset>
- License: CC-BY 4.0 (Open University, UK)
- 7 modules (courses), 22 module-presentations (2013-2014), 32,593 student
  enrolments (28,785 distinct students; one row = one student in one presentation)
- Raw data is **not** committed to this repository - see
  [data/README.md](data/README.md) for the download script and instructions.

## 5. ML Task Type

**Supervised binary classification.**
Target: `at risk` (final result = Withdrawn or Fail) vs `not at risk`
(final result = Pass or Distinction), simplified from OULAD's 4-class
`final_result` field for actionability.

**Prediction point:** the day 25% of the course presentation has elapsed
(day 58-67 depending on the run). Only features
available at that point are used: demographics, registration timing, VLE
(virtual learning environment) click activity up to the cutoff day, and scores
of TMA assessments already due by the cutoff. Final exams, later assessments,
and whole-course activity totals are explicitly excluded (see
[docs/data_audit.md](docs/data_audit.md), leakage section).

## 6. Project Pipeline / Architecture

```
raw OULAD CSVs (downloaded, not committed)
        │  src/download_data.py
        ▼
data audit + EDA ──────────► docs/data_audit.md (conclusions, issue log)
        │  notebooks/01_data_audit_eda.ipynb
        ▼
presentation-based split ──► train (2013B+2013J) / val (2014B) / test (2014J)
        │  src/make_split.py
        ▼
early-window features ─────► 18 leakage-safe features per student
        │  src/features.py
        ▼
baselines + experiments ───► 7 MLflow runs, selection on validation
        │  src/train_baseline.py, src/train_experiments.py
        ▼
one-shot test evaluation ──► reports/final_evaluation.md
        │  src/evaluate_final.py
        ▼
inference + demo ──────────► src/predict.py, demo.ipynb (Colab-ready)
```

## 7. Models / Approaches Tested

Seven runs, all logged to MLflow and evaluated on the validation cohort
(2014B); full table in [reports/experiments_results.md](reports/experiments_results.md):

- Baselines: `DummyClassifier` (majority) and Logistic Regression
- Logistic Regression with class weighting
- Random Forest (regularized and deep variants)
- XGBoost (plain and class-weighted), plus decision-threshold tuning for
  every candidate

Key findings: class weighting is redundant once the threshold is tuned
explicitly; boosting gives the best-ranked risk scores (PR-AUC 0.810 on
validation).

## 8. Final Model and Justification

**XGBoost (400 trees, depth 4, learning rate 0.05) with an operating
threshold of 0.327**, chosen on validation - best PR-AUC (0.810) and best
tuned-threshold F1 (0.716) among all runs, catching ~84% of at-risk students
on validation. The full pipeline (preprocessing fitted on train only +
model) is saved as `models/final_model.joblib`; selection evidence is in
[reports/experiments_results.md](reports/experiments_results.md).

## 9. Evaluation Metrics and Results

> **Visual report:** [`reports/project_report.html`](reports/project_report.html)
> renders every figure and table below (class balance, all seven experiment
> runs, confusion matrix, PR curve, score distribution, per-module results,
> fairness slices, feature importance, error analysis). Regenerate it with
> `python src/build_report_data.py && python src/build_report_html.py`  -
> the numbers come from the model's own predictions, so the page cannot drift
> from the pipeline.

**Primary metrics: Recall and F1** for the "at risk" class - missing an
at-risk student is costlier than a false alarm. **Supporting metric: PR-AUC.**
Accuracy alone is not used as a headline metric because the classes are
imbalanced.

### Success criteria

The system is useful only if it beats what a university could do with no
model at all, so the bar is set against the baselines rather than against a
round number:

1. **At-risk recall must be far above the majority-class baseline.** That
   baseline scores 0.000 recall (it flags nobody), so any usable model must
   catch a clear majority of at-risk students: **recall >= 0.70**.
2. **Precision must stay meaningfully above the cohort's base rate.** Flagging
   at random would be right 37.6% of the time on the test cohort, so flags are
   only worth an advisor's time at **precision >= 0.50**.
3. **Ranking quality must beat the no-skill floor.** PR-AUC must exceed the
   base rate 0.376 by a wide margin, since advisors work from a ranked list.
4. **The model must beat plain logistic regression on recall**, otherwise the
   added complexity is not justified.

Decisions fixed *before* the test set was touched: the primary metrics (in the
approved brief), the population rule (DQ-04), the split, the leakage bans in
`src/config.py`, the decision threshold (tuned on validation only), and the
DQ-06 evaluation protocol of reporting with and without module CCC. The four
numeric bars above are derived from the baselines and the cohort base rate,
and were written into this README after the experiments were run - they are
stated here so the result can be judged against a rule rather than against
hindsight, not as a claim of pre-registration.

Final results on the frozen test cohort (2014J, evaluated exactly once  -
full report in [reports/final_evaluation.md](reports/final_evaluation.md)):

| Model | Recall (at-risk) | Precision | F1 | PR-AUC |
|---|---|---|---|---|
| Dummy (majority) | 0.000 | 0.000 | 0.000 | - |
| Logistic Regression | 0.584 | 0.672 | 0.625 | - |
| **XGBoost @ threshold 0.327** | **0.760** | 0.583 | 0.660 | 0.714 |

The model catches 76% of at-risk students on a completely unseen future
cohort (2,499 of 3,289), including the cold-start module CCC (recall 0.775
there despite zero CCC training data). Error analysis shows the missed
students were still engaged at the cutoff - their problems develop later
than the prediction point, an honest limitation of a single early check.

## 10. Installation Instructions

Two dependency sets are provided, so running the demo does not require the
full training stack:

```bash
git clone https://github.com/Mubina-lazy/edu02-early-warning-model.git
cd edu02-early-warning-model
python -m venv .venv && source .venv/bin/activate   # optional but recommended

# A) demo / inference only (pandas, scikit-learn, xgboost, joblib)
pip install -r requirements-demo.txt

# B) full reproduction: audit, training, MLflow experiment tracking
pip install -r requirements.txt
python src/download_data.py    # downloads OULAD (~430 MB unzipped) into data/raw/
```

`scikit-learn` and `xgboost` are pinned to the versions that fitted
`models/final_model.joblib`; those versions are recorded in
`models/final_model_meta.json`, and `src/predict.py` warns if the environment
differs. `pandas` is capped below 3 because MLflow requires it; the pipeline
was verified to produce identical metrics on pandas 2.3 and 3.0.

## 11. Training / Fine-Tuning Instructions

Reproduce the whole pipeline from raw data (each step is deterministic,
seed 42):

```bash
python src/make_split.py         # presentation-based train/val/test split
python src/features.py           # leakage-safe early-window features
python src/train_baseline.py     # baselines: Dummy + Logistic Regression
python src/train_experiments.py  # 5 experiments, selects + saves final model
python src/evaluate_final.py     # one-shot test evaluation + report
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect all runs
```

## 12. Demo / Inference Run Instructions

Open **[`demo.ipynb`](demo.ipynb)** from the repository in Google Colab (or
locally) and run all cells top to bottom. It needs **no dataset download**  -
the trained model ships with the repo (`models/`, <1 MB). The notebook shows
real unseen examples, input validation on bad inputs, and an edge case.
Programmatic use:

```python
import sys; sys.path.insert(0, "src")
from predict import load_model, predict_risk
model, meta = load_model()
print(predict_risk(student_dict, model, meta))
```

## 13. Example Input and Output

Input - one student's early-course data (a real 2014J student, later
Withdrawn):

```python
{"code_module": "AAA", "gender": "F", "region": "East Anglian Region",
 "highest_education": "A Level or Equivalent", "imd_band": "70-80%",
 "age_band": "0-35", "disability": "False",
 "early_total_clicks": 3, "early_active_days": 1,
 "days_since_last_activity": 57,
 "early_tma_due_count": 2, "early_tma_submitted_count": 0,
 "early_tma_mean_score": None,
 "date_registration": -144, "num_of_prev_attempts": 1, "studied_credits": 60}
```

Output:

```python
{"risk_probability": 0.889, "risk_band": "High", "flagged_for_advisor": True,
 "decision_threshold": 0.327,
 "signals": ["very low online activity (bottom quartile)",
             "inactive for 57 days at the check point",
             "has not submitted any assignment that was already due",
             "has previous unsuccessful attempts at this course"],
 "note": "Decision-support only: an advisor must review every flag."}
```

## 14. Known Limitations

- OULAD is from the UK Open University (2013-2014). Its patterns do **not**
  transfer directly to Uzbekistani universities (different education system,
  language, culture, and institutional structure). This project is a
  **methodology-demonstrating prototype, not a production-ready system**.
- The data is over a decade old and comes from distance-learning courses only.
- A single early prediction point misses students whose problems start later
  in the course (the dominant error mode - see the error analysis in
  [reports/final_evaluation.md](reports/final_evaluation.md)); periodic
  re-scoring would be the natural extension.
- Module CCC appears only in the 2014 presentations, so its test results are
  a cold-start case (documented and reported separately).
- Additional limitations found during the data audit are recorded in
  [docs/data_audit.md](docs/data_audit.md).

## 15. Responsible AI Considerations

- **Fairness:** demographics such as disability, socio-economic band
  (`imd_band`), and region could cause the model to systematically flag some
  groups as "at risk". Subgroup metrics on the test set are reported in
  [reports/final_evaluation.md](reports/final_evaluation.md): recall for
  students with a declared disability (0.827) is not below the overall level
  (0.760), i.e. the model does not systematically miss that group; their
  higher flag rate mirrors their genuinely higher base risk and is exactly
  why flags must lead to supportive outreach, never penalties.
- **Intended use:** a decision-support signal for academic advisors - never an
  automatic decision about a student. A human must review every flag.
- **Prohibited use:** admission decisions, grading, discipline, or any punitive
  action based on the model's output.
- **Privacy:** OULAD is anonymized and officially released under CC-BY 4.0;
  no personal data is added by this project.

## 16. Author

**Mubinabegim To'lqinjonova**

---

## Acknowledgements and sources

- **Dataset:** OULAD - Open University Learning Analytics Dataset. Kuzilek, J.,
  Hlosta, M. & Zdrahal, Z. *Open University Learning Analytics dataset.*
  Scientific Data 4, 170171 (2017). <https://doi.org/10.1038/sdata.2017.171>
  Licensed CC-BY 4.0. Fallback mirror: the authors' own R package,
  <https://github.com/jakubkuzilek/oulad>.
- **Libraries:** scikit-learn, XGBoost, pandas, NumPy, MLflow, joblib,
  matplotlib, pyreadr - versions pinned in `requirements.txt`. No pretrained
  model was used; the classifier was trained from scratch on the training
  presentations.
- **AI assistance:** an AI coding assistant (Claude) was used for scaffolding,
  refactoring and documentation support, as permitted by the Module 8 rules.
  Every change was reviewed and tested; the problem framing, modelling and
  evaluation decisions are the author's own, and the assistance is visible in
  the commit history (`Co-Authored-By` trailers). No external AI service or
  hosted API is presented as the project's model.
