# Eight-Block Defense Deck Map

One core idea per slide. The deck follows the argument, not the notebook order.

| Block | Main message | Repository evidence | Planned slide(s) |
|---|---|---|---|
| 1. Problem, user, objective | Universities find out too late; advisors need the signal at 25% of the course, while help still changes the outcome | `docs/PROJECT_BRIEF.md`, README sections 2-5 | 1-2 |
| 2. Data and risks | 32,593 student enrolments (OULAD, CC-BY 4.0); the honest population is the 26,241 still enrolled at the cutoff, because the rest already have a known outcome | `data/README.md`, `docs/data_audit.md`, `notebooks/01_data_audit_eda.ipynb` | 3-4 |
| 3. End-to-end pipeline | Leakage is prevented in code, not by intention: banned columns in one config, a cutoff filter on every aggregation, preprocessing fitted on train only, split by whole course presentation | `src/config.py`, `src/features.py`, `src/make_split.py`, `reports/split_summary.csv` | 5-6 |
| 4. Baseline, experiments, final choice | A do-nothing model scores 62% accuracy with zero recall - that is why recall and F1 lead; seven tracked runs, and class weighting turned out to be a threshold in disguise | `reports/experiments_results.md`, `reports/mlflow_runs.csv`, `reports/results/experiment_summary.csv` | 7-9 |
| 5. Final evaluation and errors | One shot on the unseen 2014J cohort: recall 0.760, precision 0.583, F1 0.660, PR-AUC 0.714; the 790 missed students looked healthy at the checkpoint | `reports/final_evaluation.md`, `reports/results/final_metrics.json`, `reports/figures/final_confusion.png` | 10-12 |
| 6. Demo and user journey | Raw student dict -> validation -> pipeline -> risk band + the model's own top factors; invalid input is refused, never scored | `demo.ipynb`, `src/predict.py`, `tests/test_inference.py` | 13-14 |
| 7. Limitations and responsible use | UK data from 2013-2014, a methodology prototype; support signal only, never a verdict; no recall gap for students with a declared disability | `docs/RESPONSIBLE_AI_AND_LIMITATIONS.md`, fairness slices in `reports/final_evaluation.md` | 15-16 |
| 8. Conclusion and one controlled next step | The early-engagement signal generalises even to a course with no training data (CCC recall 0.775); the one controlled next step is re-scoring at later checkpoints | `reports/project_report.html`, README section 14 | 17 |

## Slide discipline

- Every slide carries one number or one claim, not a list of both.
- The four headline metrics appear once, on the evaluation slide - not repeated.
- No slide shows code; the demo shows code.
- The visual report `reports/project_report.html` is the source of every chart,
  so a screenshot from it always matches the repository.
