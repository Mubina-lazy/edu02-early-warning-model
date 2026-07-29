# Experiment Results (Model Gate)

All runs trained on 2013B+2013J, evaluated on **validation (2014B)** only,
logged to MLflow (`mlflow.db`; run `mlflow ui --backend-store-uri
sqlite:///mlflow.db`). Implementation: `src/train_baseline.py`,
`src/train_experiments.py`. Random seed 42 everywhere.

"Tuned thr" = decision threshold maximizing F1 on validation (the default
0.5 is just a convention; an early-warning system should pick its operating
point deliberately).

| Run | Hypothesis | Recall@0.5 | Prec@0.5 | F1@0.5 | PR-AUC | Tuned thr | Recall@thr | F1@thr |
|---|---|---|---|---|---|---|---|---|
| B1 Dummy (majority) | reference floor | 0.000 | 0.000 | 0.000 | 0.457 | — | — | — |
| B2 Logistic Regression | simplest real model | 0.565 | 0.821 | 0.670 | 0.803 | — | — | — |
| E1 LogReg balanced | class weights raise recall? | 0.639 | 0.757 | 0.693 | 0.803 | 0.365 | 0.760 | 0.713 |
| E2 Random Forest | non-linear interactions help? | 0.756 | 0.656 | 0.703 | 0.788 | 0.426 | 0.825 | 0.710 |
| E3 RF deep (leaf=1) | does deeper overfit? | 0.693 | 0.702 | 0.698 | 0.786 | 0.360 | 0.841 | 0.711 |
| **E4 XGBoost** | boosting beats forest? | 0.698 | 0.716 | 0.707 | **0.810** | 0.327 | 0.839 | **0.716** |
| E5 XGBoost weighted | boosting + weights for recall | 0.756 | 0.671 | 0.711 | 0.809 | 0.440 | 0.808 | 0.716 |

## What the experiments showed

1. **Class weighting (E1, E5) is a threshold in disguise.** It shifts the
   trade-off toward recall but does not improve the ranking quality
   (PR-AUC stays ~0.80). Once we tune the threshold explicitly, weighting
   adds nothing (E4 ≈ E5 at their tuned points).
2. **Non-linearity helps recall, boosting ranks best.** The forest (E2)
   trades precision for recall; XGBoost (E4) has the best PR-AUC (0.810)
   — the best-ordered risk scores — which matters because advisors will
   consume a ranked risk list.
3. **Deeper forest (E3) did not generalize better** than the regularized
   one (E2) — mild overfitting, as hypothesized.

## Final model selection

**E4 — XGBoost (400 trees, depth 4, lr 0.05), operating threshold 0.327**
(chosen on validation, F1-optimal). Justification: best PR-AUC and best
tuned-threshold F1, with recall 0.839 / precision 0.625 on validation —
it catches ~5 of every 6 at-risk students while ~5 of every 8 flags are
correct. Simpler LogReg keeps higher precision but misses far more at-risk
students even when rebalanced.

Artifacts: `models/final_model.joblib` (full preprocessing + model
pipeline, fitted on train only) + `models/final_model_meta.json`
(threshold and validation metrics). The test split (2014J) was not used
for any decision above; it is evaluated exactly once in
`src/evaluate_final.py` (see `reports/final_evaluation.md`).
