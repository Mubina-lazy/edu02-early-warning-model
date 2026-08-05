# Experiment Log

Each row answers: what changed, why, and what was learned. All runs were
tracked in MLflow; the exported log is `reports/mlflow_runs.csv`, and the
scripts are `src/train_baseline.py` and `src/train_experiments.py`
(random seed 42 throughout).

**The test split was not used for any decision in this table.**

| Run ID | Approach | Controlled change | Data split | Primary metric | Supporting metrics | Artifact / run link | Conclusion |
|---|---|---|---|---|---|---|---|
| baseline-001 | Baseline | none - reference floor | train 2013B+2013J -> validation 2014B | recall 0.000 | F1 0.000, PR-AUC 0.457, F1@tuned - | `reports/mlflow_runs.csv` | A do-nothing model still scores 54% accuracy on validation with 0.000 recall: accuracy cannot be the headline metric. |
| baseline-002 | Simple linear model | linear model on the 18 early-window features | train 2013B+2013J -> validation 2014B | recall 0.565 | F1 0.670, PR-AUC 0.803, F1@tuned - | `reports/mlflow_runs.csv` | Early-window features alone already carry real signal (PR-AUC 0.803); the pipeline works end to end. |
| exp-E1 | Logistic Regression | class_weight=balanced | train 2013B+2013J -> validation 2014B | recall 0.639 | F1 0.693, PR-AUC 0.803, F1@tuned 0.713 | `reports/mlflow_runs.csv` | Weighting raises recall 0.565 -> 0.639 but leaves PR-AUC unchanged: it moves the operating point, it does not rank better. |
| exp-E2 | Random Forest | non-linear, min_samples_leaf=5, balanced subsample | train 2013B+2013J -> validation 2014B | recall 0.756 | F1 0.703, PR-AUC 0.788, F1@tuned 0.710 | `reports/mlflow_runs.csv` | Non-linearity buys recall (0.756) at the cost of precision; PR-AUC drops slightly below the linear model. |
| exp-E3 | Random Forest | min_samples_leaf=1 (unconstrained) | train 2013B+2013J -> validation 2014B | recall 0.693 | F1 0.698, PR-AUC 0.786, F1@tuned 0.711 | `reports/mlflow_runs.csv` | Deeper is not better - mild overfitting, PR-AUC 0.786 against 0.788 for the regularized forest. |
| exp-E4 | XGBoost | gradient boosting, 400 trees, depth 4, lr 0.05 | train 2013B+2013J -> validation 2014B | recall 0.698 | F1 0.707, PR-AUC 0.810, F1@tuned 0.716 | `reports/mlflow_runs.csv` | Best-ordered risk scores of all runs (PR-AUC 0.810) and best tuned-threshold F1 (0.716). Selected. |
| exp-E5 | XGBoost | boosting + scale_pos_weight | train 2013B+2013J -> validation 2014B | recall 0.756 | F1 0.711, PR-AUC 0.809, F1@tuned 0.716 | `reports/mlflow_runs.csv` | Ties E4 on tuned F1 (0.716): once the threshold is tuned explicitly, class weighting adds nothing. |

## Final selection

- **Selected run:** `exp-E4`
- **Selected model:** XGBoost (400 trees, depth 4, learning rate 0.05),
  operating threshold **0.327** tuned on validation.
- **Evidence-based reason:** best PR-AUC of all seven runs (0.810) and best F1
  at the tuned threshold (0.716, recall 0.839) on the validation cohort.
  Ranking quality matters most because advisors consume a ranked list.
- **Trade-off accepted:** XGBoost weighted ties it on tuned F1. I preferred the
  unweighted model for marginally better ranking and one fewer moving part -
  a preference, not a decisive gap, and I say so when asked.
- **One-shot test result:** recall 0.76,
  precision 0.583, F1 0.66,
  PR-AUC 0.714 (`reports/final_evaluation.md`).

## Determinism

The MLflow log contains repeated executions of the same run names, from the
verification pass. Every repeat reproduces **identical** metrics, which is the
determinism evidence for this pipeline.
