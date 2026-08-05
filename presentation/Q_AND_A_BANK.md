# Technical Q&A Bank

Answer structure: **Decision -> Evidence -> Limitation.**
Every answer below points at a file, a number, or a run you can open.

---

### 1. Why did you choose this problem and this ML task?

**Decision.** EDU-02 asks for an earlier signal of academic risk. I framed it as
binary classification - at risk (withdrawn or failed) against not at risk
(passed or distinction) - because the advisor's decision is binary: reach out
or not. The four-class original would not change what the advisor does, and
splitting the data four ways leaves fewer examples per class.

**Evidence.** `docs/PROJECT_BRIEF.md`; README sections 2 and 5;
`src/config.py` defines the target in one place.

**Limitation.** Collapsing Withdrawn and Fail hides *why* a student is at risk;
an advisor still has to look at the case.

---

### 2. How did you prevent data leakage?

**Decision.** Four controls. (a) A fixed prediction point: 25% of each course
presentation. (b) Banned columns listed once in `src/config.py` - final result,
unregistration date, and by rule anything dated after the cutoff. (c) All
learned preprocessing - imputer, encoder, scaler - is fitted inside a scikit-learn
Pipeline on the training split only; validation and test only get `transform`.
(d) The split is by whole course presentation, so no run and no cohort appears
on both sides.

**Evidence.** `docs/data_audit.md` section 5 (the leakage register with
severities); `src/config.py`; `src/features.py` filters every aggregation with
`date <= cutoff_day` and ends with an assertion that no banned column survived.

**Limitation.** 678 students (8.0% of the test set) appear in both the training
era and the test era, in *different* course runs. I kept them - returning
students exist in real deployment too - and documented it as DQ-07 rather than
hiding it.

---

### 3. Why are these metrics appropriate?

**Decision.** Recall and F1 on the at-risk class, with PR-AUC supporting.
The two errors are not equally expensive: a false negative means a struggling
student gets no help, a false positive costs an advisor a few minutes. Accuracy
treats them as equal, so it is the wrong headline.

**Evidence.** Measured, not asserted: the majority-class baseline scores 62%
accuracy on the test cohort with **0.000** recall
(`reports/results/baseline_comparison.csv`). PR-AUC matters because advisors
work from a ranked list, and it summarises ranking quality across all
thresholds.

**Limitation.** F1 weighs precision and recall equally, which is not exactly the
real cost ratio. I do not know the true ratio, so I did not invent one; I report
both numbers and the confusion matrix so a reader can apply their own.

---

### 4. What is the baseline and why is it meaningful?

**Decision.** Two baselines. `DummyClassifier(strategy="most_frequent")` is the
floor - it is what "no model" looks like. Logistic regression is the "would a
simple model have been enough?" test.

**Evidence.** `src/train_baseline.py`; `reports/baseline_results.md`. On test:
dummy 0.000 recall; logistic regression 0.584 recall / 0.672 precision; the
final model 0.760 recall / 0.583 precision.

**Limitation.** Logistic regression keeps higher *precision*. If an institution
had very little advisor capacity, that trade might actually be preferable - the
choice depends on how many check-ins they can afford.

---

### 5. What changed between your experiments?

**Decision.** One controlled change per run: class weighting on the linear
model; a regularized random forest; the same forest unconstrained; gradient
boosting; boosting plus class weighting. Every run also had its decision
threshold tuned on validation.

**Evidence.** `reports/experiments_results.md` (hypothesis per run);
`reports/results/experiment_summary.csv`; `reports/mlflow_runs.csv` (the exported
MLflow log).

**Limitation.** I did not run a wide hyper-parameter search - this is a
comparison of approaches, not a tuning competition. A larger search might gain a
little, and would risk overfitting the validation cohort.

---

### 6. Why was the final model selected?

**Decision.** XGBoost at threshold 0.327, selected on validation by F1 at the
tuned threshold, with PR-AUC as the tie-break.

**Evidence.** Validation: PR-AUC 0.810 (best of the seven), tuned-threshold F1
0.716, recall 0.839. `reports/experiments_results.md`, "Final model selection".

**Limitation.** XGBoost weighted ties it on tuned F1 (0.716). I chose the
unweighted model because its ranking is marginally better and it has one fewer
moving part - a defensible preference, not a decisive gap. The most interesting
finding was that class weighting is a threshold in disguise.

---

### 7. Which error is most costly?

**Decision.** The false negative: a genuinely at-risk student the model does not
flag, who therefore gets no outreach. That is why the threshold was pushed down
from 0.5 to 0.327, deliberately buying recall with precision.

**Evidence.** Confusion matrix on test: 790 false negatives against 1,784 false
positives (`reports/results/final_metrics.json`). At 0.5 the model would miss
more: recall 0.624 instead of 0.760.

**Limitation.** 1,784 false positives is real advisor time. At 8,746 students
the model flags about half the cohort, which only works if outreach is cheap -
an email or a nudge, not a formal meeting.

---

### 8. How does the demo load the final model and preprocessing?

**Decision.** One artifact holds both. `models/final_model.joblib` is the entire
scikit-learn Pipeline: the ColumnTransformer with the fitted imputer, encoder
and scaler, plus the trained XGBoost classifier. `src/predict.py` loads it with
`joblib.load` and reads the threshold from `models/final_model_meta.json`.

**Evidence.** `src/predict.py` (`load_model`); `models/README.md`;
`tests/test_inference.py::test_artifact_loads_with_recorded_versions`.

**Limitation.** A pickled scikit-learn object is version-sensitive, so the
metadata records the versions that fitted it and `load_model` warns - once, in
plain language - if the environment differs. That is why `requirements-demo.txt`
pins scikit-learn and xgboost exactly.

---

### 9. What happens with invalid or edge input?

**Decision.** Validation runs before the model sees anything.
`validate_input` checks required fields, numeric ranges, the course code, and
the logical rule that submitted assignments cannot exceed those due. A failure
raises `ValueError` naming the field and the bound - it never returns a
prediction.

**Evidence.** `demo.ipynb` shows five refusals; `tests/test_inference.py` covers
each. The zero-activity student - registered but never opened the course - is
treated as a valid and very alarming state, scoring 0.964.

**Limitation.** These are schema and range checks, not fraud detection. A
plausible-looking but wrong value (say 400 clicks that never happened) passes,
because nothing in the input can contradict it.

---

### 10. Who could be harmed by a wrong output?

**Decision.** The student. A false negative denies help to someone who needed
it; a false positive can feel like being labelled. So the output is framed as a
support signal with contributing factors, never a verdict, and the note "an
advisor must review every flag" travels with every prediction.

**Evidence.** `reports/final_evaluation.md`, fairness slices. Students with a
declared disability are flagged more often (flag rate 0.629 against 0.476), but
their recall is 0.827 - above the overall 0.760 - and their genuine base risk is
higher (0.490 against 0.376). So the model is not missing that group; it is
responding to a real difference.

**Limitation.** Equal recall is not the only fairness definition. Under a
different definition - say equal flag rates - this model would look worse, and
the disparity would be a problem rather than an explanation.

---

### 11. Where should the system not be used?

**Decision.** Not for admission, grading, funding or disciplinary decisions.
Not as an automated action of any kind. Not outside the population it was
trained on.

**Evidence.** `docs/RESPONSIBLE_AI_AND_LIMITATIONS.md`; README section 15.

**Limitation.** The dataset is UK Open University distance learning, 2013-2014.
It is over a decade old and does not represent Uzbekistani universities - a
different education system, language and culture. This is a
methodology-demonstrating prototype; deploying it elsewhere would require
retraining on local data and a fresh fairness audit.

---

### 12. How can another person reproduce the main result?

**Decision.** Two paths. The demo needs no data at all: open `demo.ipynb` in
Colab and run all cells - the model artifact is under 1 MB and ships with the
repository. Full reproduction from raw data is five scripts in order.

**Evidence.** README sections 10-12; `docs/REPRODUCTION_TEST.md` records the
clean-Colab run that passed. The pipeline is deterministic: every repeated run
in `reports/mlflow_runs.csv` reproduces identical metrics.

**Limitation.** The official OULAD download was blocked from my development
environment, so I added `--source mirror` (the dataset authors' own GitHub
package) and the script validates every table's row count against the published
paper either way.

---

## Questions I should expect to find harder

**"Your success criteria were written after the results. Isn't that circular?"**
Yes, and I say so in README section 9 rather than claiming pre-registration.
The bars are derived from the baselines and the base rate - not from the score I
got. What genuinely was fixed in advance is listed in the same section.

**"Why is precision only 0.583?"**
Because I chose it. At threshold 0.5 precision is 0.654 and recall drops to
0.624 - 128 more at-risk students missed. For an early-warning system that
trade is the wrong way round, so I moved the threshold. The PR curve in the
report shows the whole trade-off.

**"You used AI to build this. What did you actually do?"**
I defined the problem, made every modelling decision, reviewed and tested the
code, and found real defects myself - the requirements file that could not
install, and the Colab cell that silently reused a stale clone. The assistance
is declared in the submission and visible in the commit history.
