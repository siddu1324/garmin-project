"""Synthetic retrospective boundaries, matched cohorts and grouped model checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from garmin_ema_baseline.models import paired_bootstrap
from garmin_hr_activity.features import CLOCK_FLAGS, prepare
from garmin_inventory.privacy import AuditError
from garmin_morning_recall.data import WINDOWS, columns_for, wake_features
from garmin_morning_recall.experiment import (
    cohort_membership,
    fit_matched,
    metrics,
    model_view,
    run_models,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/morning_recall.json").read_text())
HR_CONFIG = json.loads((ROOT / "config/hr_activity.json").read_text())


def synthetic_features():
    n = 20
    x = np.arange(n)
    f = pd.DataFrame(
        {
            "n_ema_id": ["synthetic_" + str(i) for i in x],
            "n_participant": ["synthetic_person_" + str(i // 2) for i in x],
            "n_partition": "development",
            "n_fold": (x // 2) % 5,
            "n_survey_category": "Morning",
            "pss_combo": 4.0 + x % 7,
            "local_time_sin": np.sin(x / 5),
            "local_time_cos": np.cos(x / 5),
            "context_available": True,
            "candidate_available": True,
            "scenario": CONFIG["scenarios"][0],
            "wake_time": x * 10000,
            "occupied_bin_fraction": 1.0,
            "PSS_COPE_raw": "forbidden_current_answer",
        }
    )
    for interval in WINDOWS:
        for j, col in enumerate(columns_for(interval)):
            f[col] = 60 + x + j * np.sin(x)
    return f


def test_wake_features_duplicates_conflicts_and_later_minute_support():
    f = pd.DataFrame(
        {
            "n_epoch_s": [0.0, 0.0, 60.0, 120.0, 120.0, 180.0, 240.0],
            "n_value": [60.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0],
            "n_logical_row": np.arange(7),
            "q_conflicting_timestamp_value": False,
            **{name: False for name in CLOCK_FLAGS},
        }
    )
    g, _ = prepare(f, "hr", HR_CONFIG)
    v, q, _ = wake_features(g, 0, 240, HR_CONFIG)
    assert q["valid_readings"] == 3 and q["conflicting_rows"] == 2
    assert np.isclose(v["hr_mean_bpm__wake"], np.mean([60.0, 70.0, 100.0]))
    assert np.isclose(v["hr_sd_bpm__wake"], np.std([60.0, 70.0, 100.0], ddof=1))
    assert v["hr_latest_bpm__wake"] == 100 and q["latest_age_seconds"] == 60
    later, q, _ = wake_features(g, 60, 240, HR_CONFIG)
    assert q["valid_readings"] == 2 and later["hr_slope_bpm_per_minute__wake"] is None
    assert q["status_slope_bpm_per_minute"] == "insufficient_observations"
    assert q["maximum_internal_gap_seconds"] == 120
    with pytest.raises(AuditError):
        wake_features(g, 240, 240, HR_CONFIG)


def test_partial_bins_and_absent_observations_remain_missing():
    f = pd.DataFrame(
        {
            "n_epoch_s": [0.0, 60.0, 120.0],
            "n_value": [60.0, 70.0, 80.0],
            "n_logical_row": [0, 1, 2],
            "q_conflicting_timestamp_value": False,
            **{name: False for name in CLOCK_FLAGS},
        }
    )
    g, _ = prepare(f, "hr", HR_CONFIG)
    _, q, _ = wake_features(g, 0, 121, HR_CONFIG)
    assert q["possible_minute_bins"] == 3 and q["occupied_bin_fraction"] == 1
    values, q, _ = wake_features(g, 200, 300, HR_CONFIG)
    assert all(v is None for v in values.values()) and q["recording_status"] == "no_recording"
    assert q["maximum_gap_including_edges_seconds"] == 100


def test_pairwise_and_common_cohorts_preserve_exclusions():
    f = synthetic_features()
    f.loc[0, columns_for("fixed_5")[0]] = np.nan
    f.loc[1, "pss_combo"] = np.nan
    f.loc[2, "candidate_available"] = False
    f.loc[2, columns_for("wake")] = np.nan
    cfg = {**CONFIG, "scenarios": [CONFIG["scenarios"][0]]}
    m = cohort_membership(f, cfg)
    assert len(m) == 80
    assert m.query("cohort=='pair_fixed_60'").included.sum() == 18
    assert m.query("cohort=='common'").included.sum() == 17
    assert "missing_outcome" in m.loc[m.n_ema_id.eq("synthetic_1"), "reason"].iloc[0]
    assert "wake_boundary_unavailable" in m.loc[m.n_ema_id.eq("synthetic_2"), "reason"].iloc[0]


def test_predictor_projection_excludes_wake_quality_answers_and_other_windows():
    f = synthetic_features()
    projected = model_view(f, "wake")
    assert not {"wake_time", "occupied_bin_fraction", "PSS_COPE_raw"} & set(projected)
    assert not any(c.endswith("__60m") or c.endswith("__wake") for c in projected)
    with pytest.raises(AuditError):
        columns_for("fixed_1440")


def test_matching_scaling_dummy_and_validation_label_independence():
    f = synthetic_features()
    train, valid = f.loc[f.n_fold.ne(0)], f.loc[f.n_fold.eq(0)]
    pred, audit = fit_matched(train, valid, ["wake", "fixed_60"], CONFIG)
    assert np.all(pred["median"] == train.pss_combo.median())
    assert np.all(pred["mean"] == train.pss_combo.mean())
    wake = next(a for a in audit if a["model"] == "wake")
    assert wake["training_categories"] == ["Morning"] and wake["training_count"] == len(train)
    expected = model_view(train, "wake")[wake["numeric_columns"]].mean().to_numpy()
    assert np.allclose(wake["training_mean"], expected)
    changed = valid.copy()
    changed.pss_combo = 999.0
    again, again_audit = fit_matched(train, changed, ["wake", "fixed_60"], CONFIG)
    assert audit == again_audit
    for name in pred:
        assert np.array_equal(pred[name], again[name])
    contaminated = train.copy()
    contaminated.loc[contaminated.index[0], "n_participant"] = valid.n_participant.iloc[0]
    with pytest.raises(AuditError):
        fit_matched(contaminated, valid, ["wake", "fixed_60"], CONFIG)


def test_infeasible_cohorts_skip_together_without_redrawing():
    f = synthetic_features().iloc[:2].copy()
    cfg = {**CONFIG, "scenarios": [CONFIG["scenarios"][0]]}
    m, pred, cohorts, folds, audit = run_models(f, cfg)
    assert pred.empty and not audit and len(m) == 8
    assert folds.status.ne("fitted").all() and cohorts.limited_interpretation.all()
    assert set(folds.fold) == set(range(5))


def test_participant_errors_and_paired_cluster_bootstrap():
    f = pd.DataFrame(
        {
            "n_participant": ["synthetic_a", "synthetic_a", "synthetic_b"],
            "pss_combo": [0.0, 0.0, 0.0],
            "a": [1.0, 3.0, 4.0],
            "b": [0.0, 0.0, 0.0],
        }
    )
    m = metrics(f, "a")
    assert m["mae"] == 8 / 3 and m["participant_mae"] == 3
    assert np.isclose(m["participant_rmse"], (np.sqrt(5) + 4) / 2)
    boot = paired_bootstrap(f, "a", "b", 2000, 20261002)
    assert boot == paired_bootstrap(f, "a", "b", 2000, 20261002)
    assert boot["mae_difference"] == 8 / 3
    assert boot["participant_mae_difference"] == 3
    assert boot["mae_difference_low"] == 2 and boot["mae_difference_high"] == 4
