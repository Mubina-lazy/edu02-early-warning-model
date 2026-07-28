# Data: OULAD — Open University Learning Analytics Dataset

## Source and access

- **Publisher:** The Open University (UK), Knowledge Media Institute.
- **Official page:** <https://analyse.kmi.open.ac.uk/open_dataset>
- **Citation:** Kuzilek, J., Hlosta, M. & Zdrahal, Z. *Open University Learning
  Analytics dataset.* Scientific Data 4, 170171 (2017).
  <https://doi.org/10.1038/sdata.2017.171>
- **License:** CC-BY 4.0 — free to use and share with attribution.
- **Fallback mirror:** the dataset authors' own GitHub package
  <https://github.com/jakubkuzilek/oulad> (same license, same data).

## How to get the data

Raw data is **never committed** to this repository (`data/raw/` is
git-ignored). Download it with:

```bash
python src/download_data.py                  # from the official site
python src/download_data.py --source mirror  # from the authors' GitHub mirror
```

Either way, the script verifies every table's row count against the numbers
published in the dataset paper before declaring success. Expect ~470 MB of
CSV files in `data/raw/` after extraction.

## Files and what one row means

| File | Rows | One row represents |
|------|------|--------------------|
| `courses.csv` | 22 | one module-presentation (a specific run of a course), with its length in days |
| `studentInfo.csv` | 32,593 | **one student registered in one module-presentation** (the unit of analysis of this project) — demographics + `final_result` |
| `studentRegistration.csv` | 32,593 | registration (and possibly unregistration) dates for a student in a presentation |
| `assessments.csv` | 206 | one assessment (TMA/CMA/Exam) in a presentation, with its due date and weight |
| `studentAssessment.csv` | 173,912 | one student's submission of one assessment (submission date, score) |
| `vle.csv` | 6,364 | one VLE material/activity (id_site) in a presentation, with its type |
| `studentVle.csv` | 10,655,280 | one student's clicks on one VLE material on one day |

Key identifiers: `code_module` (7 courses AAA–GGG), `code_presentation`
(2013B, 2013J, 2014B, 2014J — B = February start, J = October start),
`id_student`. Dates are measured in **days relative to the presentation
start** (day 0 = course start; negative = before start).

## Target

`final_result` in `studentInfo.csv`: Distinction / Pass / Fail / Withdrawn.
This project's binary target: **at risk = Withdrawn or Fail**,
**not at risk = Pass or Distinction**.

## Known limitations and sensitive-data notes

- Data is anonymized by the publisher; no names or contact details exist.
  We add no personal data.
- Contains sensitive demographic attributes (disability, socio-economic
  `imd_band`, region, age band, gender) — used with fairness checks, never
  for punitive decisions.
- UK distance-learning context, 2013–2014 only. Findings do not transfer
  directly to other countries (including Uzbekistan) or to present-day
  student behavior. This is a methodology prototype.
- `imd_band` (socio-economic deprivation index) has missing values; details
  in `docs/data_audit.md`.
