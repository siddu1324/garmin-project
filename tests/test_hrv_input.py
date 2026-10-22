"""Synthetic tests only; no real HRV values or participant records are loaded."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from garmin_hrv_input.contract import window_input
from garmin_hrv_input.synthetic import (
    fixtures,
    independent_reference,
    native_probe,
    package_run,
    validate_all,
    validate_run,
)


@pytest.fixture
def policy():
    return json.loads((Path(__file__).parents[1] / "config/ibi_eligibility.json").read_text())[
        "policies"
    ][1]


def hydrated(case, policy):
    f = case.frame()
    return window_input(
        f,
        case.window_end,
        policy,
        case.scenario,
        excluded=f.n_logical_row.isin(case.excluded_rows).to_numpy(),
    )


def test_hand_calculated_example_and_denominators():
    r = independent_reference([800, 900, 1000, 950, 850])
    assert r["mean_ibi_ms"] == r["median_ibi_ms"] == 900
    assert math.isclose(r["sdnn_sample_ms"], math.sqrt(6250))
    assert math.isclose(r["sd_population_ms"], math.sqrt(5000))
    assert math.isclose(r["rmssd_ms"], math.sqrt(8125))
    assert r["pnn50_pairs_percent"] == 75 and r["nk_native_pnn50_percent"] == 60
    assert independent_reference([800, 850, 800])["pnn50_pairs_percent"] == 0
    assert independent_reference([800])["sdnn_sample_ms"] is None
    assert independent_reference([800])["rmssd_ms"] is None


@pytest.mark.parametrize("case", fixtures(), ids=lambda c: c.name)
def test_contract_never_creates_pairs_or_mutates_clocks(case, policy):
    original = case.frame()
    w = hydrated(case, policy)
    assert (w.records.n_epoch_s >= w.start).all() and (w.records.n_epoch_s < w.end).all()
    assert w.records.n_epoch_s.isin(original.n_epoch_s).all()
    assert np.all(w.records.c_run_id.eq(-1) == ~w.records.c_acceptable)
    for _, run in w.runs.iterrows():
        r = w.records[w.records.c_run_id == run.run_id]
        assert np.all(np.diff(r.n_logical_row) == 1)
        assert r.c_link_to_previous_original_record.iloc[1:].all()
        assert run.allowed_pair_count == len(r) - 1
        assert not run.real_hrv_authorized
    assert original.equals(case.frame())


@pytest.mark.parametrize("name", ["recording_gap", "removed_interval", "rejected_interval"])
def test_gap_and_removal_cannot_be_concatenated(name, policy):
    case = next(c for c in fixtures() if c.name == name)
    w = hydrated(case, policy)
    assert len(w.runs) >= 2
    with pytest.raises(ValueError):
        validate_run(w.records[w.records.c_acceptable])
    assert w.runs.allowed_pair_count.sum() < max(0, w.records.c_acceptable.sum() - 1)


def test_boundary_scenarios_retain_rejected_records(policy):
    end = hydrated(next(c for c in fixtures() if c.name == "boundary_end"), policy)
    start = hydrated(next(c for c in fixtures() if c.name == "boundary_start"), policy)
    assert end.records.n_epoch_s.tolist() == [0.0, 0.5, 1.3, 299.5]
    assert end.records.c_boundary_excluded.tolist() == [True, True, False, False]
    assert start.records.c_boundary_excluded.tolist() == [False, False, False, True]


def test_screen_pass_does_not_mean_five_minute_uninterrupted_hrv(policy):
    from garmin_hrv_input.synthetic import SyntheticCase
    from garmin_preprocessing.eligibility import evaluate_windows, parameter_array

    t = np.arange(0.0, 300.0, 0.8)
    # A small clock interruption breaks a link without removing most recorded support.
    t[188:] += 0.3
    t = t[t < 300]
    rr = np.full(len(t), 0.8)
    z = np.zeros(len(t), bool)
    mask, metrics, _ = evaluate_windows(
        t, rr, z, z, z, np.array([300.0]), 0, parameter_array(policy)
    )
    assert mask[0] == 0 and 120 <= metrics[0, 4] < 180
    c = SyntheticCase("two_short_runs", tuple(t), tuple(rr * 1000), tuple(range(len(t))))
    w = hydrated(c, policy)
    assert len(w.runs) == 2 and w.runs.timestamp_span_seconds.max() < 180
    with pytest.raises(ValueError):
        package_run(c, w.records[w.records.c_acceptable], "heartpy")


def test_synthetic_package_adapters_match_independent_arithmetic(policy):
    report = validate_all(policy)
    assert report["cases"] == 13
    assert sum(r["status"] == "matched" for r in report["comparisons"]) > 20
    assert any(r["status"].startswith("blocked") for r in report["comparisons"])
    assert not report["real_data_hrv_calculated"]
    assert not report["interpolation"]


def test_neurokit_partial_vs_no_timestamp_agreement_is_not_a_quality_gate():
    partial = native_probe(next(c for c in fixtures() if c.name == "jitter_some_10ms_matches"))
    none = native_probe(next(c for c in fixtures() if c.name == "jitter_no_10ms_matches"))
    assert partial["native_missing_detected"]
    assert partial["effective_package_pairs"] == partial["native_10ms_matching_pairs"] < 4
    assert not none["native_missing_detected"]
    assert none["native_10ms_matching_pairs"] == 0 and none["effective_package_pairs"] == 4


def test_adapters_refuse_real_data_entry_and_changed_synthetic_clocks(policy):
    c = fixtures()[0]
    w = hydrated(c, policy)
    with pytest.raises(TypeError):
        package_run(None, w.records, "heartpy")
    w.records["n_epoch_s"] += 1000
    with pytest.raises(ValueError):
        package_run(c, w.records, "heartpy")


def test_neurokit_start_anchoring_is_explicitly_blocked(policy):
    c = next(c for c in fixtures() if c.name == "known_start")
    w = hydrated(c, policy)
    with pytest.raises(ValueError):
        package_run(c, w.records, "neurokit2")
    result = package_run(c, w.records, "heartpy")
    assert result["pnn50_pairs_percent"] == 75


def test_synthetic_validation_is_deterministic_across_import_state(policy):
    assert validate_all(policy) == validate_all(policy)
