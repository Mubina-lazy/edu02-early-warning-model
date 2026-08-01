# Model Artifacts

| File | What it is |
|---|---|
| `final_model.joblib` | The complete final pipeline (preprocessing + XGBoost classifier), fitted on the training presentations (2013B + 2013J) only. Committed because it is small (<1 MB) and lets `demo.ipynb` run without downloading the dataset. |
| `final_model_meta.json` | The selected run's name, the operating decision threshold (0.327, tuned on validation 2014B), and the validation metrics recorded at selection time. |

## How the artifact was produced

```bash
python src/download_data.py      # get raw OULAD data
python src/make_split.py         # presentation-based train/val/test split
python src/features.py           # early-window feature matrices
python src/train_baseline.py     # baselines (Dummy, Logistic Regression)
python src/train_experiments.py  # 5 experiments -> selects + saves this model
```

Selection evidence: `reports/experiments_results.md`. Final unseen-data
results: `reports/final_evaluation.md`. Every run is also logged in MLflow
(`mlflow.db`, local, git-ignored).

## How to use it

```python
import sys; sys.path.insert(0, "src")
from predict import load_model, predict_risk

model, meta = load_model()
result = predict_risk(one_student_dict, model, meta)
```

See `demo.ipynb` for full worked examples, the input schema, and validation
behavior. The model is decision support for academic advisors only - never
an automatic decision system.
