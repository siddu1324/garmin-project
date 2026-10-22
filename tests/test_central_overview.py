"""Presentation checks protect denominators and matched-cohort comparisons."""

import numpy as np
import pandas as pd
from pathlib import Path

from garmin_central_overview.overview import missing_summary, existing_results


def test_missing_cells_are_not_missing_responses():
    dictionary = pd.DataFrame({"name": ["a", "b", "c"], "modality": ["hr", "hr", "steps"]})
    frame = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, np.nan], "c": [1, 2]})
    summary = missing_summary(dictionary, frame, ["modality"]).set_index("modality")
    assert summary.loc["hr", "missing_cells"] == 3
    assert summary.loc["hr", "feature_cells"] == 4
    assert summary.loc["hr", "missing_pct"] == 75
    assert summary.loc["steps", "missing_pct"] == 0


def test_saved_comparisons_keep_distinct_cohorts():
    new, old, paired = existing_results(Path.cwd())
    assert len(new) == 7 and new.responses.eq(3689).all()
    assert len(old) == 4 and old.responses.eq(1355).all()
    assert set(old.model) == {"dummy", "logistic", "xgboost", "lightgbm"}
    assert len(paired) == 3 and paired.responses.eq(3689).all()
    assert set(new.cohort) == {"all_valid"}
