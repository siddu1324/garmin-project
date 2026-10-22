"""Synthetic recall boundaries and strict training-only model checks."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from garmin_inventory.privacy import AuditError
from garmin_recall_diagnostics.recall import boundaries, interval_quality, summaries, wake_boundary
from garmin_recall_diagnostics.training import fit_training, training_frame

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/recall_diagnostics.json").read_text())
HR_CONFIG = json.loads((ROOT / "config/hr_activity.json").read_text())


def test_wake_midnight_noon_offset_and_no_date_repair():
    actual = pd.Timestamp("2024-06-01 13:00:00")
    cutoff = pd.Timestamp("2024-06-01 17:00:00", tz="UTC").timestamp()
    midnight, reason = wake_boundary(actual, -240, cutoff, "12:00 AM")
    noon, _ = wake_boundary(actual, -240, cutoff, "12:00 PM")
    assert reason == "candidate" and noon - midnight == 43200
    assert pd.Timestamp(midnight, unit="s", tz="UTC").hour == 4
    later, reason = wake_boundary(actual, -240, cutoff, "2:00 PM")
    assert later > cutoff and reason == "wake_not_before_cutoff"
    assert wake_boundary(actual, -240, cutoff, "13:00 AM")[1] == "unparseable_wake_response"
    assert wake_boundary(actual, -240, cutoff, "")[1] == "missing_wake_response"
    assert wake_boundary(actual, -240, cutoff, "1:00 AM", True)[1] == "invalid_ema_clock"


def test_current_response_retrospective_and_buffer_uncertainty():
    cutoff = pd.Timestamp("2024-01-01 08:00:30", tz="UTC").timestamp()
    ema = pd.DataFrame(
        [
            dict(
                n_ema_id="synthetic",
                n_participant="synthetic",
                n_partition="development",
                n_fold=0,
                n_survey_category="Morning",
                n_cutoff_epoch=cutoff,
                n_actual_local=pd.Timestamp("2024-01-01 08:01:30"),
                n_offset_minutes=0,
                SLEEP_WAKETIME="8:00 AM",
                n_invalid_clock=False,
            )
        ]
    )
    b = boundaries(ema, CONFIG)
    assert b.candidate_available.sum() == 1
    assert not b.available_at_prediction.any()
    assert b.end_epoch.eq(cutoff).all()
    assert "minute_uncertainty_reaches_cutoff" in b.boundary_reason.tolist()
    ema.n_survey_category = "Endday"
    b = boundaries(ema, CONFIG)
    assert not b.candidate_available.any() and b.start_epoch.isna().all()


def groups(times, accepted):
    return pd.DataFrame(
        dict(
            n_epoch_s=times,
            n_value=np.full(len(times), 70.0),
            a_acceptable=accepted,
            a_group_size=np.ones(len(times), int),
            a_conflict=~np.array(accepted),
            a_group_bad_clock=np.zeros(len(times), bool),
            a_group_bad_value=np.zeros(len(times), bool),
            n_logical_row=np.arange(len(times)),
        )
    )


def test_partial_bin_left_inclusive_right_exclusive_rejected_and_empty():
    g = groups([0.0, 60.0, 90.0, 120.0, 125.0], [True, True, False, True, True])
    q, _ = interval_quality(g, 0, 125, HR_CONFIG)
    assert q["valid_readings"] == 3 and q["possible_minute_bins"] == 3
    assert q["occupied_minute_bins"] == 3 and q["partial_final_bin_seconds"] == 5
    assert q["maximum_internal_gap_seconds"] == 60 and q["latest_age_seconds"] == 5
    q, _ = interval_quality(g, 90, 120, HR_CONFIG)
    assert q["recording_status"] == "all_readings_rejected"
    assert q["maximum_gap_including_edges_seconds"] == 30
    q, _ = interval_quality(g, 130, 160, HR_CONFIG)
    assert q["recording_status"] == "no_recording" and q["occupied_bin_fraction"] == 0
    with pytest.raises(AuditError):
        interval_quality(g, 10, 10, HR_CONFIG)


def synthetic_training():
    n = 40
    x = np.arange(n, dtype=float)
    return pd.DataFrame(
        dict(
            n_ema_id=["synthetic_" + str(i) for i in range(n)],
            n_participant=["synthetic_person_" + str(i // 4) for i in range(n)],
            n_partition="development",
            n_fold=x.astype(int) % 4,
            n_survey_category=np.where(x % 2, "Morning", "Midday"),
            local_time_sin=np.sin(x),
            local_time_cos=np.cos(x),
            hr_mean_bpm__5m=x + 60,
            hr_sd_bpm__5m=np.sqrt(x + 1),
            hr_latest_bpm__5m=x + 61,
            hr_slope_bpm_per_minute__5m=x / 40,
            pss_combo=4 + x / 10,
        )
    )


def test_training_fold_guard_arithmetic_variation_and_constant():
    f = synthetic_training()
    stats, params, pred = fit_training(f, 4, "hr", 0, CONFIG)
    assert stats["training_mae"] < 1e-7
    assert stats["training_prediction_sd"] > 0
    assert np.isclose(stats["training_objective"], 0.5 * stats["training_mae"])
    assert np.allclose(params["training_mean"], f[params["numeric_columns"]].mean())
    assert not pred.n_fold.eq(4).any()
    with pytest.raises(AuditError):
        fit_training(f, 0, "hr", 0.01, CONFIG)
    with pytest.raises(AuditError):
        fit_training(f, 4, "hr", 0.1, CONFIG)
    f.pss_combo = 6.0
    stats, _, _ = fit_training(f, 4, "context_hr", 0.01, CONFIG)
    assert stats["nonzero_coefficients"] == 0 and stats["training_prediction_sd"] == 0


def test_targets_read_only_after_fold_filter_and_no_heldout_labels_requested():
    f = synthetic_training()
    original = f.loc[f.n_fold.ne(0), ["n_ema_id", "pss_combo"]].copy()
    with patch("garmin_recall_diagnostics.training.pd.read_parquet", return_value=original) as read:
        train = training_frame(f.drop(columns="pss_combo"), Path("synthetic"), 0)
        assert read.call_args.kwargs["filters"] == [("n_fold", "!=", 0)]
        assert read.call_args.kwargs["columns"] == ["n_ema_id", "pss_combo"]
    assert not train.n_fold.eq(0).any()
    f.loc[f.n_fold.eq(0), "pss_combo"] = 999999
    with patch("garmin_recall_diagnostics.training.pd.read_parquet", return_value=original):
        again = training_frame(f.drop(columns="pss_combo"), Path("synthetic"), 0)
    pd.testing.assert_frame_equal(train, again)


def test_matched_coverage_retains_no_recording():
    b = pd.DataFrame(
        dict(
            n_ema_id=["synthetic_a", "synthetic_b"],
            n_participant=["synthetic_a", "synthetic_b"],
            n_fold=[0, 1],
            n_survey_category="Morning",
            scenario="reported_minute_start",
            boundary_reason="candidate",
            candidate_available=True,
        )
    )
    c = b.assign(
        valid_readings=[0, 2],
        occupied_bin_fraction=[0.0, 0.5],
        recording_status=["no_recording", "observed_valid"],
    )
    fixed = c.drop(columns="occupied_bin_fraction").assign(
        lookback_minutes=5, occupied_minute_bins=[0, 1], possible_minute_bins=5
    )
    result = summaries(b, c, fixed)
    assert result["matched_fixed_coverage"].responses.item() == 2
    assert result["paired_coverage"].neither_hr.item() == 1
