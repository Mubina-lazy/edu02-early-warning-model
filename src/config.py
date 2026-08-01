"""Single source of truth for the project's scope and leakage rules.

Every script and notebook imports these constants instead of re-typing
module lists or cutoff logic, so a leakage rule can never be "forgotten"
in one place while enforced in another.
"""

# --- Target definition (approved brief) -----------------------------------
# at risk = 1 (needs support), not at risk = 0.
AT_RISK_RESULTS = {"Withdrawn", "Fail"}
NOT_AT_RISK_RESULTS = {"Pass", "Distinction"}
TARGET_COLUMN = "at_risk"

# --- Prediction point ------------------------------------------------------
# The cutoff day for each module-presentation is this fraction of its
# length (courses.module_presentation_length), rounded to whole days.
# Only information dated on/before the cutoff may become a feature.
EARLY_WINDOW_FRACTION = 0.25

# --- Split by module-presentation (time-aware, approved brief) -------------
TRAIN_PRESENTATIONS = ["2013B", "2013J"]
VALIDATION_PRESENTATIONS = ["2014B"]
TEST_PRESENTATIONS = ["2014J"]

# --- Leakage controls (see docs/data_audit.md section 5) -------------------
# Columns that must NEVER be used as model features.
BANNED_COLUMNS = {
    "final_result",          # the target itself
    "date_unregistration",   # target proxy (reveals withdrawal); only used
                             # for the population rule below
    "id_student",            # identifier, not a feature
}

# Population rule (issue DQ-04): a student whose unregistration date is on
# or before the cutoff is NOT part of the modeling population - their
# outcome is already known at prediction time.
EXCLUDE_UNREGISTERED_BY_CUTOFF = True

# Assessment rules: a score may be used only if the assessment was DUE by
# the cutoff AND actually SUBMITTED by the cutoff. Exams are always after
# the prediction point (many have no date at all) and are never used.
ALLOWED_ASSESSMENT_TYPES = {"TMA"}
REQUIRE_DUE_BY_CUTOFF = True
REQUIRE_SUBMITTED_BY_CUTOFF = True
EXCLUDE_BANKED = True  # banked results come from a previous presentation

# VLE rule: click data is aggregated only for days <= cutoff. Whole-course
# totals are never computed.
