# Baseline Results (Model Gate, step 1)

Evaluated on the **validation split only** (2014B, 6,186 students, 45.7% at
risk). The test split (2014J) remains untouched until the final evaluation.
Runs logged to MLflow (`mlflow.db`, local; see `src/train_baseline.py`).

| Model | Recall (at-risk) | Precision (at-risk) | F1 (at-risk) | PR-AUC | Accuracy |
|---|---|---|---|---|---|
| DummyClassifier (majority: "nobody is at risk") | 0.000 | 0.000 | 0.000 | 0.457 | 0.543 |
| Logistic Regression (defaults, fit-on-train pipeline) | **0.565** | **0.821** | **0.670** | **0.803** | 0.745 |

Confusion matrix, Logistic Regression (validation):

|  | predicted not at risk | predicted at risk |
|---|---|---|
| **actually not at risk** | 3,010 (TN) | 348 (FP) |
| **actually at risk** | 1,229 (FN) | 1,599 (TP) |

## Interpretation

- The dummy baseline confirms the accuracy trap documented in the audit: it
  reaches 54.3% accuracy while catching **zero** at-risk students. Accuracy
  alone is meaningless for this task.
- Logistic Regression on early-window features alone already catches **56.5%**
  of at-risk students with high precision (82.1%): when it flags a student,
  it is usually right. The whole pipeline (features -> preprocessing -> model
  -> honest evaluation) works.
- The main cost right now is the **1,229 false negatives** - at-risk students
  the model misses. For an early-warning system this is the expensive error,
  so the next experiments target recall: class weighting, decision-threshold
  tuning, and stronger models (Random Forest, XGBoost) per the approved brief.

## Reproduce

```bash
python src/features.py        # build early-window feature matrices
python src/train_baseline.py  # train + log both baselines to MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs
```
