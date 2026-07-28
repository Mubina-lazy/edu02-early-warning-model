# Student Performance Early-Warning Model (EDU-02)

An AI/ML Fundamentals capstone project: an early-warning classifier that flags
students at risk of failing or withdrawing from an online course, using only
information available in the first weeks of the course.

> **Project status:** Data Gate stage (data audit, EDA, split, leakage controls).
> Modeling has not started yet. See [PROJECT_STATUS.md](PROJECT_STATUS.md).

---

## 1. Project Title

**Student Performance Early-Warning Model** — predicting academic risk from
early-course data (OULAD).

## 2. Problem Statement

Universities and online academies usually notice struggling students only after
final grades are in — when it is too late to help. This project builds a binary
classifier that estimates, early in a course, whether a student is **at risk**
(will withdraw or fail) so that academic support staff can intervene while
there is still time. Missing an at-risk student is costlier than an unnecessary
check-in, so the model prioritizes recall.

## 3. Selected Project Track

**Field-Based Scenario Track — EDU-02 "Student Performance Early-Warning Model"
(EdTech).** The official scenario brief serves as the project specification.

## 4. Dataset Source

**OULAD — Open University Learning Analytics Dataset**

- Source: <https://analyse.kmi.open.ac.uk/open_dataset>
- License: CC-BY 4.0 (Open University, UK)
- 7 modules (courses), 22 module-presentations (2013–2014), 32,593 students
- Raw data is **not** committed to this repository — see
  [data/README.md](data/README.md) for the download script and instructions.

## 5. ML Task Type

**Supervised binary classification.**
Target: `at risk` (final result = Withdrawn or Fail) vs `not at risk`
(final result = Pass or Distinction), simplified from OULAD's 4-class
`final_result` field for actionability.

**Prediction point:** after the first ~25–30% of the course. Only features
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
[NEXT GATE] leakage-safe preprocessing + feature building (early window only)
        ▼
[NEXT GATE] baseline → candidate models → evaluation → demo
```

Stages marked **[NEXT GATE]** begin after the Data Gate review.

## 7. Models / Approaches Tested

Planned (not yet started — Data Gate must pass first):

- Baselines: `DummyClassifier` (majority class) and Logistic Regression
- Main candidates: Random Forest and Gradient Boosting (XGBoost)

_This section will be filled with actual experiment results at the Model Gate._

## 8. Final Model and Justification

_To be completed after experiments (Model Gate)._

## 9. Evaluation Metrics and Results

**Primary metrics: Recall and F1** for the "at risk" class — missing an
at-risk student is costlier than a false alarm. **Supporting metric: PR-AUC.**
Accuracy alone is not used as a headline metric because the classes are
imbalanced.

_Results will be reported on the held-out 2014J test presentation only, once,
at the final evaluation._

## 10. Installation Instructions

```bash
git clone <this-repository-url>
cd <repository-folder>
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python src/download_data.py    # downloads OULAD (~430 MB unzipped) into data/raw/
```

## 11. Training / Fine-Tuning Instructions

_To be completed at the Model Gate. Currently available:_

```bash
python src/make_split.py   # builds the presentation-based train/val/test split
```

## 12. Demo / Inference Run Instructions

_To be completed after the final model exists (Colab-first demo notebook)._

## 13. Example Input and Output

_To be completed with the demo._ Planned shape:

- **Input:** one student's demographics + first-weeks VLE activity + early TMA
  scores for a course presentation.
- **Output:** risk probability (0–1), risk band (Low/Medium/High), and the top
  contributing factors for academic advisors.

## 14. Known Limitations

- OULAD is from the UK Open University (2013–2014). Its patterns do **not**
  transfer directly to Uzbekistani universities (different education system,
  language, culture, and institutional structure). This project is a
  **methodology-demonstrating prototype, not a production-ready system**.
- The data is over a decade old and comes from distance-learning courses only.
- Additional limitations found during the data audit are recorded in
  [docs/data_audit.md](docs/data_audit.md).

## 15. Responsible AI Considerations

- **Fairness:** demographics such as disability, socio-economic band
  (`imd_band`), and region could cause the model to systematically flag some
  groups as "at risk". Subgroup-level metrics will be checked before any
  conclusion is drawn (base-rate differences are already documented in the
  audit).
- **Intended use:** a decision-support signal for academic advisors — never an
  automatic decision about a student. A human must review every flag.
- **Prohibited use:** admission decisions, grading, discipline, or any punitive
  action based on the model's output.
- **Privacy:** OULAD is anonymized and officially released under CC-BY 4.0;
  no personal data is added by this project.

## 16. Author

**Mubina — _[TODO: replace with your full name as registered in the LMS]_**
