# PROJECT STATUS

**Project:** Student Performance Early-Warning Model (EDU-02, Field-Based Scenario)
**Student:** Mubina
**Current stage:** Data Gate (course route step 3 — data audit, EDA, split, leakage prevention)
**Gate self-assessment:** _pending review with mentor_

---

## Course route progress

| # | Stage | Status |
|---|-------|--------|
| 1 | Scope and problem definition | ✅ Done — approved Project Brief (EDU-02, OULAD, binary risk target) |
| 2 | GitHub repository and project organization | ✅ Done — structure, README skeleton, requirements, .gitignore |
| 3 | Data audit, EDA, preprocessing, leakage prevention | 🟡 In progress — this is the current gate |
| 4 | Baseline, experiments, MLflow, model selection | ⬜ Not started (blocked until Data Gate review) |
| 5 | Final inference/demo workflow | ⬜ Not started |
| 6 | Reproducibility, documentation, submission, defense prep | ⬜ Not started |

## Data Gate checklist (from M8C3)

| Done | Gate condition | Evidence |
|------|----------------|----------|
| ⬜ | Data source and usage conditions documented | `data/README.md` |
| ⬜ | Target/objective clear and matches project scope | `docs/data_audit.md` §1 |
| ⬜ | EDA/data audit has written conclusions | `docs/data_audit.md` §2–§6, `notebooks/01_data_audit_eda.ipynb` |
| ⬜ | Data-quality issue log complete enough to guide decisions | `docs/data_audit.md` issue log |
| ⬜ | Split strategy matches real use, visible proof | `src/make_split.py`, `docs/data_audit.md` split section |
| ⬜ | Leakage risks identified and controlled | `docs/data_audit.md` leakage register, `src/config.py` |
| ⬜ | Preprocessing reusable or explicitly planned | `docs/data_audit.md` preprocessing plan |
| ⬜ | Model-ready inputs exist or named blocker recorded | `docs/data_audit.md` model-ready status |
| ⬜ | PROJECT_STATUS.md and repository evidence current | this file |
| ⬜ | Verified Data Gate commit visible | git history |

## Next action

Complete the Data Gate items above, then **stop and review the evidence with
the mentor before starting Model Gate work** (baseline, MLflow, experiments —
course Class 4). Do not begin modeling before that review.

## Decisions log (short)

- Target = binary: at risk (Withdrawn/Fail) vs not at risk (Pass/Distinction).
- Prediction point = first ~25–30% of each module-presentation.
- Split = by module-presentation (time-aware), not random.
- Primary metrics = Recall and F1 (at-risk class); PR-AUC supporting.
- Raw OULAD data is downloaded by script, never committed.
