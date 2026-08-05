"""Delivery checks for the inference path.

These are the guarantees the demo depends on, so they are worth failing a
build over: the saved artifact loads, a valid unseen student gets a
well-formed result, invalid input is refused with a readable message, and
the documented example still reproduces its documented score.

Run:  python -m pytest tests/ -v      (or: python tests/test_inference.py)
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import (  # noqa: E402
    INPUT_SCHEMA, load_model, predict_risk, validate_input,
)

# A real student from the unseen 2014J cohort, described only by what was
# known at the prediction point. True outcome: Withdrawn.
AT_RISK_STUDENT = {
    "code_module": "AAA", "gender": "F", "region": "East Anglian Region",
    "highest_education": "A Level or Equivalent", "imd_band": "70-80%",
    "age_band": "0-35", "disability": "False",
    "early_total_clicks": 3, "early_active_days": 1,
    "days_since_last_activity": 57,
    "early_tma_due_count": 2, "early_tma_submitted_count": 0,
    "early_tma_mean_score": None,
    "date_registration": -144, "num_of_prev_attempts": 1, "studied_credits": 60,
}

ENGAGED_STUDENT = {
    **AT_RISK_STUDENT,
    "imd_band": "60-70%", "age_band": "35-55",
    "early_total_clicks": 1101, "early_active_days": 45,
    "days_since_last_activity": 8,
    "early_tma_submitted_count": 2, "early_tma_mean_score": 85.0,
    "date_registration": -38, "num_of_prev_attempts": 0,
}


@pytest.fixture(scope="module")
def model_and_meta():
    """The artifact must load from a cold start, exactly as the demo does."""
    return load_model()


def test_artifact_loads_with_recorded_versions(model_and_meta):
    model, meta = model_and_meta
    assert model is not None
    assert meta["run_name"] == "E4_xgboost"
    assert 0 < meta["threshold"] < 1
    # the versions that fitted the pipeline must travel with it
    assert "scikit-learn" in meta["fitted_with"]


def test_valid_input_returns_the_documented_schema(model_and_meta):
    model, meta = model_and_meta
    result = predict_risk(AT_RISK_STUDENT, model, meta)

    assert set(result) >= {"risk_probability", "risk_band", "flagged_for_advisor",
                           "decision_threshold", "top_factors", "signals"}
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["risk_band"] in {"Low", "Medium", "High"}
    assert isinstance(result["flagged_for_advisor"], bool)
    assert len(result["top_factors"]) == 3
    for factor in result["top_factors"]:
        assert set(factor) == {"factor", "direction", "contribution"}
        assert factor["direction"] in {"increases risk", "lowers risk"}


def test_documented_example_still_reproduces(model_and_meta):
    """The README and demo quote these scores; they must stay true."""
    model, meta = model_and_meta
    assert predict_risk(AT_RISK_STUDENT, model, meta)["risk_probability"] == 0.889
    assert predict_risk(ENGAGED_STUDENT, model, meta)["risk_probability"] == 0.122


def test_disengaged_scores_higher_than_engaged(model_and_meta):
    """Sanity: the student with no activity must not look safer."""
    model, meta = model_and_meta
    at_risk = predict_risk(AT_RISK_STUDENT, model, meta)
    engaged = predict_risk(ENGAGED_STUDENT, model, meta)
    assert at_risk["risk_probability"] > engaged["risk_probability"]
    assert at_risk["flagged_for_advisor"] and not engaged["flagged_for_advisor"]


@pytest.mark.parametrize("field,value,expected", [
    ("early_tma_mean_score", 150, "above the maximum"),
    ("early_total_clicks", -5, "below the minimum"),
    ("code_module", "ZZZ", "must be one of"),
    ("early_tma_submitted_count", 5, "cannot exceed"),
])
def test_invalid_values_are_refused_with_a_readable_message(
        model_and_meta, field, value, expected):
    model, meta = model_and_meta
    with pytest.raises(ValueError) as err:
        predict_risk({**ENGAGED_STUDENT, field: value}, model, meta)
    assert expected in str(err.value)


def test_missing_required_field_is_refused(model_and_meta):
    model, meta = model_and_meta
    incomplete = {k: v for k, v in ENGAGED_STUDENT.items()
                  if k != "early_total_clicks"}
    with pytest.raises(ValueError) as err:
        predict_risk(incomplete, model, meta)
    assert "early_total_clicks" in str(err.value)


def test_zero_activity_edge_case_is_valid_and_alarming(model_and_meta):
    """A student who registered but never opened the course is a real state."""
    model, meta = model_and_meta
    ghost = {
        **ENGAGED_STUDENT,
        "code_module": "BBB", "imd_band": "Missing",
        "early_total_clicks": 0, "early_active_days": 0,
        "days_since_last_activity": None,
        "early_tma_submitted_count": 0, "early_tma_mean_score": None,
    }
    assert validate_input(ghost) == []
    result = predict_risk(ghost, model, meta)
    assert result["risk_band"] == "High"
    assert result["flagged_for_advisor"]


def test_optional_derived_fields_may_be_omitted(model_and_meta):
    """The caller should not have to compute derived features by hand."""
    model, meta = model_and_meta
    minimal = {k: v for k, v in ENGAGED_STUDENT.items()
               if k not in ("early_clicks_per_active_day",
                            "early_tma_any_submitted")}
    assert predict_risk(minimal, model, meta)["risk_probability"] == 0.122


def test_schema_matches_the_published_example():
    """reports/results/example_output.json must stay in step with the code."""
    example = json.loads(
        (ROOT / "reports" / "results" / "example_output.json").read_text())
    assert set(example["input"]) <= set(INPUT_SCHEMA)
    assert example["result"]["risk_band"] == "High"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
