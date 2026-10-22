"""Synthetic tests; no participant sensor rows or outcomes are loaded."""

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from garmin_hr_activity.boundaries import expected_window, verify_grid
from garmin_hr_activity.features import CLOCK_FLAGS, empty_frame, prepare, window
from garmin_hr_activity.pipeline import dictionary
from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import verify_bundle, verify_archive_bundle, artifact_hashes
from garmin_preprocessing.access import load_eligibility_windows
from garmin_hrv_input.access import load_indexed_window


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "config/hr_activity.json").read_text())


def source(t, v):
    n = len(t)
    f = pd.DataFrame(
        dict(
            n_epoch_s=np.array(t, float),
            n_value=np.array(v, float),
            n_logical_row=np.arange(n),
            n_text_local=pd.to_datetime(np.array(t) + 43200, unit="s"),
            n_inferred_offset_minutes=np.zeros(n),
            n_source_sha256=["synthetic_hash"] * n,
            q_conflicting_timestamp_value=np.zeros(n, bool),
        )
    )
    for name in CLOCK_FLAGS:
        f[name] = False
    return f


def test_hr_arithmetic_and_right_boundary(config):
    f = source([0, 60, 120, 180], [60, 90, 120, 240])
    g, ledger = prepare(f, "hr", config)
    v, q, _ = window(g, 0, 180, "hr", config)
    assert v == {
        "mean_bpm": 90.0,
        "median_bpm": 90.0,
        "sd_bpm": 30.0,
        "minimum_bpm": 60.0,
        "maximum_bpm": 120.0,
        "q25_bpm": 75.0,
        "q75_bpm": 105.0,
        "latest_bpm": 120.0,
        "slope_bpm_per_minute": 30.0,
    }
    assert q["latest_age_seconds"] == 60 and q["valid_readings"] == 3
    assert len(ledger) == 4 and f.n_value.tolist() == [60, 90, 120, 240]


def test_feature_specific_support_and_missingness(config):
    g, _ = prepare(source([1], [70]), "hr", config)
    v, q, _ = window(g, 0, 300, "hr", config)
    assert v["latest_bpm"] == 70 and v["mean_bpm"] is None
    assert q["status_mean_bpm"] == "insufficient_observations"
    g, _ = prepare(source([1, 1.5, 2], [70, 71, 72]), "hr", config)
    v, q, _ = window(g, 0, 300, "hr", config)
    assert v["mean_bpm"] == 71 and v["slope_bpm_per_minute"] is None
    assert q["status_slope_bpm_per_minute"] == "insufficient_timestamp_span"
    g, _ = prepare(source([1], [0]), "hr", config)
    v, q, _ = window(g, 0, 300, "hr", config)
    assert all(x is None for x in v.values()) and q["recording_status"] == "all_readings_rejected"
    g, _ = prepare(empty_frame(), "hr", config)
    v, q, _ = window(g, 0, 300, "hr", config)
    assert all(x is None for x in v.values()) and q["recording_status"] == "no_recording"


def test_duplicate_collapse_conflict_rejection_and_equal_values(config):
    f = source([1, 1, 2, 3, 3, 4], [70, 70, 70, 80, 90, 70])
    g, ledger = prepare(f, "hr", config)
    v, q, _ = window(g, 0, 300, "hr", config)
    assert len(ledger) == 6 and len(g) == 4 and q["valid_readings"] == 3
    assert q["same_value_tie_excess"] == 1 and q["conflicting_rows"] == 2
    assert v["mean_bpm"] == 70 and v["sd_bpm"] == 0
    assert ledger.a_analysis_keep.tolist() == [True, False, True, False, False, True]


def test_counter_boundaries_and_no_proration(config):
    g, _ = prepare(source([-10, 0, 10, 90, 100], [90, 100, 105, 119, 900]), "steps", config)
    v, q, _ = window(g, 0, 100, "steps", config)
    assert v["added_supported"] == 19 and q["supported_pairs"] == 2
    assert q["supported_pair_seconds"] == 90 and q["pairs_crossing_left_boundary"] == 1
    v, q, _ = window(g, 20, 80, "steps", config)
    assert v["added_supported"] is None and q["recording_status"] == "no_recording"


def test_counter_resets_break_negative_pair_but_allow_new_baseline(config):
    g, _ = prepare(source([0, 10, 20, 30], [100, 110, 3, 8]), "steps", config)
    v, q, _ = window(g, 0, 40, "steps", config)
    assert v["added_supported"] == 15 and q["pairs_counter_decrease"] == 1
    assert q["supported_pair_seconds"] == 20


def test_counter_bad_reading_cannot_create_new_adjacent_pair(config):
    g, _ = prepare(source([0, 10, 20, 30], [100, -1, 120, 125]), "steps", config)
    v, q, _ = window(g, 0, 40, "steps", config)
    assert v["added_supported"] == 5 and q["supported_pairs"] == 1
    g, _ = prepare(source([0, 10, 10, 20, 30], [100, 105, 109, 120, 125]), "steps", config)
    v, q, _ = window(g, 0, 40, "steps", config)
    assert v["added_supported"] == 5 and q["conflicting_rows"] == 2


def test_zero_requires_supported_observation_pair(config):
    g, _ = prepare(source([0, 10], [100, 100]), "steps", config)
    assert window(g, 0, 20, "steps", config)[0]["added_supported"] == 0
    assert window(g, 5, 20, "steps", config)[0]["added_supported"] is None
    assert window(g, 20, 40, "steps", config)[0]["added_supported"] is None


@pytest.mark.parametrize("barrier", ["day", "offset", "source", "gap", "clock"])
def test_counter_barriers(config, barrier):
    f = source([0, 10], [100, 110])
    if barrier == "day":
        f.loc[1, "n_text_local"] += timedelta(days=1)
    if barrier == "offset":
        f.loc[1, "n_inferred_offset_minutes"] = 60
    if barrier == "source":
        f.loc[1, "n_source_sha256"] = "another_synthetic_source"
    if barrier == "gap":
        f.loc[1, "n_epoch_s"] = 301
    if barrier == "clock":
        f.loc[1, "q_clock_discrepancy"] = True
    g, _ = prepare(f, "steps", config)
    v, q, _ = window(g, 0, 400, "steps", config)
    assert v["added_supported"] is None and q["supported_pairs"] == 0


def test_unresolved_counters_remain_missing_and_not_predictors(config):
    for sensor in ["calories", "intensityMinutes"]:
        g, _ = prepare(source([0, 60], [100, 105]), sensor, config)
        v, q, _ = window(g, 0, 300, sensor, config)
        assert v["added_supported"] is None and "unverified" in q["status_added_supported"]
    d = dictionary(config)
    assert len(d["features"]) == 104 and len(d["predictor_columns"]) == 88
    assert not any(
        s in n
        for n in d["predictor_columns"]
        for s in ["calories", "intensity", "stress", "quality", "ibi"]
    )


def test_hr_slope_uses_actual_elapsed_time_without_filling_gaps(config):
    g, _ = prepare(source([0, 60, 600], [60, 63, 90]), "hr", config)
    v, q, _ = window(g, 0, 660, "hr", config)
    assert math.isclose(v["slope_bpm_per_minute"], 3)
    assert q["valid_readings"] == 3 and q["maximum_internal_gap_seconds"] == 540
    assert q["occupied_minute_bins"] == 3 and q["possible_minute_bins"] == 11


def test_independent_datetime_at_both_overflows_and_day_boundary():
    cutoff = int(datetime(2030, 1, 2, tzinfo=timezone.utc).timestamp())
    for step in [0, 546, 547, 1092, 1093, 1435]:
        a, b = expected_window(cutoff, step)
        assert int(cutoff - b) == int(step) * 60 and int(b - a) == 300
        assert a >= cutoff - 86400 and b <= cutoff
    assert datetime.fromtimestamp(expected_window(cutoff, 1435)[0], tz=timezone.utc) == datetime(
        2030, 1, 1, tzinfo=timezone.utc
    )
    rows = [
        dict(
            n_ema_id="SYNTHETIC",
            n_step_index=j,
            n_window_start_epoch=expected_window(cutoff, j)[0],
            n_window_end_epoch=expected_window(cutoff, j)[1],
        )
        for j in range(1436)
    ]
    f = pd.DataFrame(rows)
    assert verify_grid(f, {"SYNTHETIC": cutoff}, [5, 15, 30, 60, 120, 360, 720, 1440]) == 1436
    f.loc[547, "n_window_end_epoch"] += 65536
    with pytest.raises(AuditError):
        verify_grid(f, {"SYNTHETIC": cutoff}, [1440])


def test_all_loaders_reject_superseded_version_even_if_renamed(tmp_path):
    archive = tmp_path / "renamed_directory"
    archive.mkdir()
    (archive / "manifest.json").write_text(
        json.dumps(
            dict(
                version="6747223eb50269ba", complete=True, artifact_sha256=artifact_hashes(archive)
            )
        )
    )
    assert verify_archive_bundle(archive)["complete"]
    with pytest.raises(AuditError, match="Superseded"):
        verify_bundle(archive)
    with pytest.raises(AuditError, match="Superseded"):
        load_eligibility_windows(tmp_path, archive, "synthetic")
    with pytest.raises(AuditError, match="Superseded"):
        load_indexed_window(tmp_path, archive, "synthetic", 0, {}, "interval_end")


def test_supersession_privacy_count_exception_checks_schema_and_arithmetic(tmp_path):
    from garmin_inventory.privacy import public_scan_text

    p = tmp_path / "window_supersession.json"
    row = dict(
        policy="screen",
        scenario="interval_end",
        lookback_minutes=1440,
        survey_category="All",
        reason="synthetic_reason",
        windows=100,
        obsolete_windows=80,
        delta_windows=20,
    )
    p.write_text(json.dumps({"attrition_delta": [row], "other": "kept"}))
    masked = json.loads(public_scan_text(p))
    assert (
        masked["other"] == "kept"
        and masked["attrition_delta"][0]["windows"] == "[validated aggregate count]"
    )
    for changed in [
        dict(row, windows="100"),
        dict(row, delta_windows=21),
        dict(row, unexpected="identity"),
    ]:
        p.write_text(json.dumps({"attrition_delta": [changed]}))
        assert public_scan_text(p) == p.read_text()
