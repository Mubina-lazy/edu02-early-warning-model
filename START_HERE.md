# Start Here

**Student Performance Early-Warning Model (EDU-02)** - flags students at risk
of failing or withdrawing after the first 25% of an online course, using only
what a university knows at that point.

**One-line result:** on a cohort the model never saw, it catches **76%** of
at-risk students (recall 0.760, precision 0.583, F1 0.660, PR-AUC 0.714),
against **0.000** recall for a majority-class baseline.

## Fastest route: run the demo (2 minutes, no data download)

[Open `demo.ipynb` in Google Colab](https://colab.research.google.com/github/Mubina-lazy/edu02-early-warning-model/blob/main/demo.ipynb)
and run all cells. The trained model ships with the repository.

## Recommended reviewer route

1. Read `README.md` (16 sections, the full project write-up).
2. Check `RUBRIC_EVIDENCE_MATRIX.md` - every criterion mapped to its evidence path.
3. Confirm the approved scope in `docs/PROJECT_BRIEF.md`.
4. Review data decisions in `data/README.md`, `docs/data_audit.md` and
   `notebooks/01_data_audit_eda.ipynb` (audit with written conclusions and the
   DQ-01...DQ-08 issue log).
5. Review modelling evidence in `docs/EXPERIMENT_LOG.md`,
   `reports/experiments_results.md` and `reports/mlflow_runs.csv`.
6. Review final results and errors in `reports/final_evaluation.md` and
   `reports/results/`.
7. Run `demo.ipynb`; `tests/` covers the same guarantees automatically.
8. Check `docs/REPRODUCTION_TEST.md` (clean-Colab test: PASS, two defects found
   and fixed).
9. Review `docs/RESPONSIBLE_AI_AND_LIMITATIONS.md`.
10. Confirm the current state in `PROJECT_STATUS.md`.

## Everything at a glance

Open **`reports/project_report.html`** - one page with every chart and table:
class balance, all seven runs, confusion matrix, PR curve, score distribution,
per-module results, fairness slices, feature importance and error analysis.

## Repository map

| Path | What it holds |
|---|---|
| `README.md` | The 16-section project documentation |
| `RUBRIC_EVIDENCE_MATRIX.md` | Criterion to evidence path |
| `PROJECT_STATUS.md` | Current stage, what is done, defence cheat-sheet |
| `data/README.md` | Dataset source, licence, data dictionary, limitations |
| `docs/` | Brief, data audit, issue log, experiment log, reproduction test, responsible AI |
| `notebooks/01_data_audit_eda.ipynb` | Executed audit and EDA with conclusions |
| `demo.ipynb` | The canonical demo (Colab-ready) |
| `src/` | Download, split, features, training, evaluation, inference, reporting |
| `models/` | The final pipeline artifact and its metadata |
| `reports/` | Evaluation reports, figures, results, MLflow run log |
| `tests/` | Delivery checks for the inference path |
| `presentation/` | Defence deck map, speaker flow, Q&A bank, fallback evidence |
| `submission/` | The LMS submission file |

Folder names follow the course's example repository where it made sense;
`models/` plays the role of `artifacts/`, and the demo lives at the repository
root so the Colab link is short.
