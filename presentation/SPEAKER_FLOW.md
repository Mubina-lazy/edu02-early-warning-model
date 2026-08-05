# Speaker Flow

## Core story (target: 4 minutes)

**1. The problem (30s)**
"Universities usually find out that a student is struggling when the final
results come in - too late to do anything. My project gives that signal after
the first quarter of the course, while an advisor can still help."

*Transition:* "So the first decision was: what exactly do we predict, and when?"

**2. The ML task and the prediction point (40s)**
"I framed it as binary classification: at risk, meaning the student will
withdraw or fail, against not at risk. The prediction point is the day 25% of
the course has elapsed - day 58 to 67 depending on the run. Everything dated
after that day is banned, because a real advisor would not have it."

*Transition:* "That ban is the part that decides whether the numbers can be
believed at all."

**3. Leakage and the split (50s)**
"Three things protect the result. First, the banned columns are in one config
file that every script imports - final result, unregistration date, later
assignment scores, whole-course click totals. Second, I removed the 6,352
students who had already unregistered before the prediction day, because their
outcome is known at that moment and keeping them would inflate recall. Third,
the split is by course presentation, not random: I train on the 2013 cohorts,
validate on 2014B, and test once on 2014J. A random split would put the same
student and the same course run on both sides."

*Transition:* "With an honest split, the next question is what counts as good."

**4. Metrics, not accuracy (40s)**
"62% of the test cohort passed. So a model that predicts 'everyone passes'
scores 62% accuracy and catches zero at-risk students - I measured that as my
dummy baseline. Missing a struggling student costs more than an unnecessary
check-in, so recall and F1 on the at-risk class are my primary metrics, with
PR-AUC as support."

*Transition:* "Then I compared seven models under exactly that rule."

**5. Experiments and the final choice (40s)**
"Seven runs tracked in MLflow: dummy, logistic regression with and without
class weighting, two random forests, two XGBoost variants, and I tuned the
decision threshold for each. One thing I learned: class weighting is a
threshold in disguise - it moves the trade-off but does not improve the
ranking, and once I tune the threshold explicitly it adds nothing. XGBoost had
the best-ordered scores, so I selected it at threshold 0.327."

*Transition:* "Then, once, on the cohort I had never touched:"

**6. The result (30s)**
"Recall 0.760, precision 0.583, F1 0.660, PR-AUC 0.714. It catches 2,499 of
3,289 at-risk students. It also holds up on module CCC, which has no training
data at all - recall 0.775 there."

*Transition:* "It is worth being clear about who it misses."

**7. Errors and limits (30s)**
"It misses 790 at-risk students, and the error analysis shows why: at the
prediction day they were still clicking, still submitting, still scoring around
84. Their problems start after the point where the model looks. That is the
honest limit of a single early checkpoint, and periodic re-scoring is the
obvious next step."

*Transition:* "Let me show it running."

## Demo handoff sentence

"Now I will show the final inference path using a new input and explain what
the output means."

## Demo route

Raw student dict -> schema validation -> the same preprocessing pipeline that
was fitted on training data -> loaded XGBoost model -> risk probability -> band
and flag at threshold 0.327 -> top contributing factors from TreeSHAP.

Files: `demo.ipynb` calls `src/predict.py`, which loads
`models/final_model.joblib` and `models/final_model_meta.json`.

Show, in order:
1. the disengaged student -> 0.889, High, flagged, factors name the inactivity;
2. the engaged student -> 0.122, Low, not flagged, factors are negative;
3. an invalid input -> refused with a readable message, no prediction;
4. the zero-activity edge case -> 0.964, High.

## 60-second emergency version

"The problem: universities notice struggling students too late. The data: OULAD,
32,593 student enrolments from the UK Open University, CC-BY 4.0. I predict at
25% of the course, using only what is known then, with the banned columns
enforced in code. The final model is XGBoost at threshold 0.327, chosen from
seven MLflow-tracked runs on a validation cohort. On the unseen 2014J cohort it
reaches recall 0.760 and precision 0.583, against zero recall for a
majority-class baseline. The demo scores a raw student dict and returns a risk
band with the model's own top factors. The main limitation: it is UK data from
2013-2014, it is a methodology prototype rather than a deployable system, and a
single early checkpoint misses students whose problems start later."

## Fallback sentence

"The live environment did not complete as expected, so I will use the prepared
fallback evidence and explain the same verified inference route transparently."

## Things to say carefully (be accurate, not impressive)

- The numeric success bars in README section 9 were written **after** the
  experiments, derived from the baselines. What was fixed **before** the test
  set was touched: the metrics, the population rule, the split, the leakage
  bans, the threshold (tuned on validation), and the CCC reporting protocol.
- An AI coding assistant was used, as the Module 8 rules permit; it is declared
  in the submission and acknowledged in the README, and the commit history
  shows it. Say so plainly if asked - the work and the decisions are yours.
