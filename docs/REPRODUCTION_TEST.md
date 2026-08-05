# Clean-Runtime Reproduction Test

## Test information

- Tester: Mubinabegim To'lqinjonova
- Date: 2026-08-01
- Environment: Google Colab, fresh runtime (Runtime > Disconnect and delete
  runtime, then Run all), notebook opened directly from GitHub
- Commit SHA at the time of the test: `58a1a81`
- Route tested: `demo.ipynb` - the canonical demo

## Steps tested

- [x] Open the repository (public, no access steps required).
- [x] Install dependencies - `requirements-demo.txt`, four packages, seconds.
- [x] Obtain data using documented instructions - **not needed for the demo**;
      the model artifact is under 1 MB and ships with the repository.
- [x] Obtain model/preprocessing artifacts - `models/final_model.joblib` and
      `models/final_model_meta.json`, loaded by `src/predict.py`.
- [x] Restart runtime/kernel and run from a clean session.
- [x] Run the canonical demo from top to bottom.
- [x] Test one valid unseen input - the disengaged 2014J student.
- [x] Test invalid and edge inputs - five refusals plus the zero-activity case.
- [x] Confirm output matches the documented format.

## Result

- **Status: PASS**
- Time to first result: under two minutes from opening the Colab link.
- Outputs reproduced **exactly** as documented:

| Case | Expected | Observed |
|---|---|---|
| Disengaged student | 0.889, High, flagged | 0.889, High, flagged |
| Engaged student | 0.122, Low, not flagged | 0.122, Low, not flagged |
| Zero-activity edge case | 0.964, High, flagged | 0.964, High, flagged |
| Five invalid inputs | refused with a readable message | all five refused |

## Issues found and fixed

This test earned its place - it found two real defects that no amount of local
testing had surfaced:

1. **`requirements.txt` could not be installed.** MLflow requires
   `pandas<3`, but pandas was pinned to `3.0.5`, so pip failed with
   `ResolutionImpossible`. It had only ever "worked" locally because the
   packages were already installed.
   *Fix:* pandas capped below 3, and the whole pipeline re-verified on
   pandas 2.3.3 - every documented metric reproduced exactly. A separate
   `requirements-demo.txt` was added so the demo installs four packages
   instead of the full training stack.

2. **The setup cell silently reused a stale clone.** Re-running the notebook in
   a Colab session that already held a clone kept the old checkout, so new
   files were missing and the metadata lookup raised `KeyError`.
   *Fix:* the setup cell now detects Colab and resets the clone to the
   published `main` (fetch + reset, because the history had been rewritten),
   so the demo can never score with a stale artifact.

## Remaining limitation

Colab's preinstalled scikit-learn can differ from the version that fitted the
artifact. `load_model()` compares them and prints **one** readable warning
naming the versions and the fix, instead of scikit-learn's six blocks of
per-step warnings. Predictions were verified to be correct in that state; to
remove the warning entirely, install `requirements-demo.txt` and restart the
runtime.
