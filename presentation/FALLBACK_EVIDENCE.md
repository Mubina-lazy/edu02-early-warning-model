# Demo Fallback Evidence

The live demo is the primary route. This is what to have ready in case the
runtime, the network or Colab itself does not cooperate.

## Prepare before the defense

- [ ] Screenshot: the setup cell showing `Loaded final model: E4_xgboost
      (decision threshold 0.327, tuned on validation)` and the `Fitted with:`
      line - this is the reproducibility evidence.
- [ ] Screenshot: the disengaged student -> `risk_probability: 0.889`,
      `risk_band: High`, with the `top_factors` list visible.
- [ ] Screenshot: the five refused inputs, showing that invalid data never
      reaches the model.
- [ ] Screenshot: the zero-activity edge case -> `0.964`, `High`.
- [ ] Optional: a 60-second screen recording of the notebook running top to
      bottom.
- [ ] Open in a second browser tab: `reports/project_report.html` (all charts
      and tables) and `reports/results/final_metrics.json`.

## Saved evidence already in the repository

Even with no screenshots, these are committed and can be opened live:

| Evidence | Path |
|---|---|
| Demo notebook with its outputs stored | `demo.ipynb` |
| A real scored student, input and output | `reports/results/example_output.json` |
| Final metrics, machine-readable | `reports/results/final_metrics.json` |
| Baseline comparison | `reports/results/baseline_comparison.csv` |
| Every experiment run | `reports/mlflow_runs.csv` |
| Confusion matrix and PR curve | `reports/figures/` |
| Clean-runtime test record | `docs/REPRODUCTION_TEST.md` |

## What to say

"The live environment did not complete as expected, so I will use the prepared
fallback evidence and explain the same verified inference route transparently."

Then walk the same route with the saved outputs: raw input -> validation ->
preprocessing -> model -> risk band and factors. Say clearly that this is
recorded evidence of the real pipeline, not a substitute demo, and offer to run
it again after the session.
