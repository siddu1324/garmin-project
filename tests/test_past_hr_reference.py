"""Synthetic tests of past-only history, UTC bins and blocked leakage paths."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from garmin_hr_activity.features import CLOCK_FLAGS, prepare
from garmin_inventory.privacy import AuditError
from garmin_past_hr_reference.diagnostics import decomposition
from garmin_past_hr_reference.models import feature_inputs, fit_predict, inner_splits, tune
from garmin_past_hr_reference.reference import earlier_reference, build_reference

ROOT = Path(__file__).parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "config/past_hr_reference.json").read_text())


def source(times, values):
    f = pd.DataFrame(
        dict(
            n_epoch_s=np.array(times, float),
            n_value=np.array(values, float),
            n_logical_row=np.arange(len(times)),
            n_source_sha256="synthetic_source",
            n_source_row=np.arange(len(times)) + 1,
            q_conflicting_timestamp_value=False,
            **{c: False for c in CLOCK_FLAGS},
        )
    )
    _, ledger = prepare(f, "hr", json.loads((ROOT / "config/hr_activity.json").read_text()))
    return ledger


def sufficient_times():
    return [3600 * h + 60 * m for h in [1, 2, 3] for m in range(10)]


def test_thresholds_both_required_and_missing_not_zero(config):
    t = sufficient_times()
    v, q, _ = earlier_reference(source(t, [70] * 30), 90000, config)
    assert (
        v == 70
        and q["reference_available"]
        and q["occupied_utc_minutes"] == 30
        and q["occupied_utc_hours"] == 3
    )
    v, q, _ = earlier_reference(source(t[:-1], [70] * 29), 90000, config)
    assert v is None and "fewer_than_30" in q["reason"]
    t = [3600 + 60 * m for m in range(30)]
    v, q, _ = earlier_reference(source(t, [70] * 30), 90000, config)
    assert v is None and "fewer_than_3_utc_hours" in q["reason"]
    v, q, _ = earlier_reference(source([], []), 90000, config)
    assert v is None and "no_recording" in q["reason"] and q["accepted_readings"] == 0


def test_left_inclusion_right_current_future_exclusion(config):
    t = [0] + sufficient_times() + [86400, 89999, 90000, 95000]
    v, q, _ = earlier_reference(source(t, [80] + [70] * 30 + [220] * 4), 90000, config)
    assert q["accepted_readings"] == 31 and v == pytest.approx((80 + 70 * 30) / 31)
    before = source(t[:-4], [80] + [70] * 30)
    assert earlier_reference(before, 90000, config)[0] == v


def test_utc_binning_and_partial_boundary_clipped_first(config):
    cutoff = 90030
    t = [0, 29, 30, 31] + sufficient_times() + [86429, 86430]
    v, q, _ = earlier_reference(
        source(t, [240, 240, 60, 80] + [70] * 30 + [100, 240]), cutoff, config
    )
    assert q["accepted_readings"] == 33 and q["occupied_utc_minutes"] == 32
    assert q["possible_utc_minute_bins"] == 1441
    assert v == pytest.approx((70 + 70 * 30 + 100) / 32)
    assert q["latest_reading_age_seconds"] == 3601


def test_observed_minutes_equal_weight_not_readings(config):
    t = sufficient_times()
    t = [t[0], t[0] + 1, t[0] + 2] + t[1:]
    values = [60, 60, 60] + [90] * 29
    v, q, _ = earlier_reference(source(t, values), 90000, config)
    assert v == 89 and q["accepted_readings"] == 32
    assert v != np.mean(values)


def test_duplicates_conflicts_and_repeated_values(config):
    t = sufficient_times()
    t = sorted(t + [t[0], t[1]])
    vals = [70] * len(t)
    vals[t.index(3660) + 1] = 90
    v, q, _ = earlier_reference(source(t, vals), 90000, config)
    assert v is None and q["occupied_utc_minutes"] == 29
    assert q["same_value_tie_excess"] == 1 and q["conflicting_rows"] == 2


def test_builder_rejects_answers_and_full_person_means(config):
    with pytest.raises(AuditError):
        build_reference(pd.DataFrame({"PSS_COPE": ["4"], "person_mean": [60]}), ROOT, config)


def test_variation_identity_and_retrospective_only():
    f = pd.DataFrame(
        {"n_participant": ["synthetic_a"] * 2 + ["synthetic_b"] * 2, "pss_combo": [3, 5, 9, 11]}
    )
    d = decomposition(f, "pss_combo")
    assert d["total_ss"] == 40 and d["within_ss"] == 4 and d["between_ss"] == 36
    assert d["within_ss_fraction"] == 0.1 and d["pooled_within_sd"] == pytest.approx(np.sqrt(2))
    assert all("person_mean" not in key or key == "person_mean_sd" for key in d)


def toy_frame():
    rng = np.random.default_rng(5)
    n = 90
    f = pd.DataFrame(
        dict(
            n_ema_id=[f"synthetic_{i}" for i in range(n)],
            n_participant=np.repeat(np.arange(15), 6),
            n_survey_category=np.tile(["Morning", "Midday", "Endday"], 30),
            pss_combo=rng.integers(3, 16, n),
        )
    )
    for view in ["B_context", "C_raw_hr", "D_reference", "E_reference_deviations"]:
        for col in feature_inputs(view)[0]:
            if col not in f:
                f[col] = rng.normal(size=n)
    return f


def test_exact_predictor_sets_and_training_only_transforms(config):
    f = toy_frame()
    tr = f[f.n_participant.lt(12)].copy()
    va = f[f.n_participant.ge(12)].copy()
    va["pss_combo"] = np.nan
    va["reference_mean_bpm"] = 9999
    tr["whole_person_mean"] = 123
    va["whole_person_mean"] = 321
    for view in ["B_context", "C_raw_hr", "D_reference", "E_reference_deviations"]:
        p, _, a = fit_predict(tr, va, view, "tuned_ridge", {"alpha": 1}, config)
        assert np.isfinite(p).all() and a["training_count"] == len(tr)
        assert (
            "whole_person_mean" not in a["input_columns"]
            and "n_participant" not in a["input_columns"]
        )
        assert np.allclose(a["training_mean"], tr[feature_inputs(view)[0]].mean())
    assert "hr_mean_bpm__60m" not in feature_inputs("E_reference_deviations")[0]
    with pytest.raises(AuditError):
        feature_inputs("whole_person_mean")
    with pytest.raises(AuditError):
        fit_predict(tr, tr, "D_reference", "tuned_ridge", {"alpha": 1}, config)


def test_nested_tuning_no_outer_outcomes_and_grid_unchanged(config):
    train = toy_frame()
    splits = inner_splits(train)
    setting, score, audits = tune(train, "D_reference", "tuned_ridge", config, splits)
    expected = [np.mean([a["mae"] for a in audits if a["candidate"] == i]) for i in range(4)]
    assert (
        score == min(expected) and setting["alpha"] == config["ridge_alphas"][np.argmin(expected)]
    )
    old = json.loads((ROOT / "config/window_benchmark.json").read_text())
    assert config["hgb_grid"] == old["hgb_grid"] and config["hgb_fixed"] == old["hgb_fixed"]


def test_between_and_within_are_distinct_retrospective_diagnostics():
    from garmin_past_hr_reference.diagnostics import association_summary

    f = pd.DataFrame(
        {
            "n_participant": np.repeat(["synthetic_a", "synthetic_b", "synthetic_c"], 2),
            "hr": [58, 62, 78, 82, 98, 102],
            "pss_combo": [11, 13, 7, 9, 3, 5],
        }
    )
    before = f.copy(deep=True)
    between, within = association_summary(f, "hr", 200, 5)
    assert between["correlation"] == pytest.approx(-1)
    assert within["correlation"] == pytest.approx(1)
    assert between["slope"] == pytest.approx(-0.2) and within["slope"] == pytest.approx(0.5)
    pd.testing.assert_frame_equal(f, before)
    assert set(between) == set(within) and "n_participant" not in between


def test_reference_join_rejects_changed_fold():
    from garmin_past_hr_reference.pipeline import link_reference

    f = pd.DataFrame(
        {
            "n_ema_id": ["synthetic_ema"],
            "n_participant": ["synthetic_person"],
            "n_partition": ["development"],
            "n_fold": [0],
            "n_survey_category": ["Morning"],
            "n_cutoff_epoch": [90000.0],
        }
    )
    ref = f.copy()
    ref["n_fold"] = 1
    with pytest.raises(AuditError):
        link_reference(f, ref, pd.DataFrame())
