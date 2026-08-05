# Responsible AI and Limitations

## Who could be harmed

The student, in two different ways.

A **false negative** - an at-risk student the model does not flag - denies
support to the person who most needed it. On the test cohort this happens to
790 students. A **false positive** - 1,784 students - risks making someone feel
labelled, and spends advisor time that could have gone elsewhere. The system is
therefore built to produce an *offer of help*, never a judgement: the output
carries contributing factors and the standing note that an advisor must review
every flag.

## Bias and representativeness

Base rates genuinely differ between groups in this data, and the audit measured
them before any modelling (`docs/data_audit.md`, section 2):

- students with a declared disability: 62% at risk against 52% overall;
- a clear socio-economic gradient by `imd_band`: 65% in the most deprived band
  down to 43% in the least;
- rows with a *missing* `imd_band` have the **lowest** risk (34%), so
  missingness is informative and is encoded as its own category rather than
  imputed away.

The fairness question that matters for an early-warning system is **recall
parity**: are struggling students missed more often in some groups? Measured on
the test cohort (`reports/final_evaluation.md`):

| Group | n | base risk | recall | flag rate |
|---|---|---|---|---|
| disability = True | 792 | 0.490 | **0.827** | 0.629 |
| disability = False | 7,954 | 0.365 | 0.751 | 0.476 |
| overall | 8,746 | 0.376 | 0.760 | 0.500 |

Students with a declared disability are flagged more often, but their recall is
*above* the overall level, and their genuine base risk is higher. The model is
not missing that group - it is responding to a real difference. That is exactly
why a flag must lead to an offer of support and never to a penalty.

**Honest caveat:** equal recall is one fairness definition, not the only one.
Under equalised flag rates this model would look worse. I report the slices so a
reader can apply their own definition rather than take mine on trust.

## Privacy

OULAD is anonymised and released by The Open University under CC-BY 4.0. No
personal data is added by this project, no re-identification is attempted, and
the raw data is not redistributed - `src/download_data.py` fetches it from the
publisher. Sensitive attributes (disability, socio-economic band, region, age,
gender) are used only as model inputs and for the fairness audit above, never
to justify a decision about an individual.

## Human oversight

Required, always. The output is a ranked support signal:

- `risk_probability` and a Low/Medium/High band, not a label;
- `top_factors` from the model itself, so an advisor can see *why*;
- the note "Decision-support only: an advisor must review every flag" is
  returned with every prediction, in code, not just in documentation.

## Prohibited and inappropriate use

- **Not** for admission, grading, funding, or disciplinary decisions.
- **Not** as an automated action of any kind - no automatic emails, holds, or
  referrals without a human in the loop.
- **Not** for ranking or comparing staff, courses, or institutions.
- **Not** outside the population it was trained on without retraining and a
  fresh fairness audit.

## Limitations

1. **It is a prototype, not a deployable system.** OULAD is UK Open University
   distance-learning data from 2013-2014. It is over a decade old and does not
   represent Uzbekistani universities - different education system, language,
   culture and institutional structure. This project demonstrates a
   methodology; deploying it elsewhere would require local data.
2. **One prediction point is not enough.** The dominant error mode is students
   whose problems begin after the first quarter: the 790 missed students were
   still clicking (median 461 clicks), still submitting (median 2 assignments)
   and scoring around 84 at the checkpoint. Re-scoring at later checkpoints is
   the natural extension and is out of scope here.
3. **Precision is 0.583 by choice.** The threshold was moved from 0.5 to 0.327
   to buy recall. That means about half the cohort is flagged, which only works
   if outreach is cheap. An institution with scarce advisor time should re-tune
   the threshold for its own capacity.
4. **Module CCC has no training data**, because it only ran in 2014. It is kept
   as a realistic cold-start case and reported separately (recall 0.775 there);
   the headline is also given without it.
5. **The success bars in README section 9 were written after the experiments**,
   derived from the baselines rather than from the observed score. What was
   fixed before the test set was touched is listed in the same section.
6. **The model explains correlation, not cause.** "No assignment submitted yet"
   being the strongest feature does not mean submitting an assignment would
   make a student safe.
