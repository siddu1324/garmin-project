"""Synthetic-only pilot guard, selection and arithmetic regression tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from garmin_hrv_input.contract import window_input
from garmin_hrv_input.synthetic import SyntheticCase, fixtures, package_run
from garmin_ibi_pilot.adapters import FEATURES, guard_run, package_comparison, reference_run
from garmin_ibi_pilot.pilot import calculate
from garmin_ibi_pilot.selection import choose_runs, overlap_components
from garmin_inventory.privacy import AuditError


@pytest.fixture
def policy():
    policies = json.loads((Path(__file__).parents[1] / "config/ibi_eligibility.json").read_text())
    return next(p for p in policies["policies"] if p["name"] == "screen")


def view(person, first, last, scenario="interval_end", cutoff=1000):
    return dict(
        directory=person,
        participant=person,
        scenario=scenario,
        first_logical_row=first,
        last_logical_row=last,
        first_timestamp=first,
        last_timestamp=last,
        ema_cutoff=cutoff,
        window_end=cutoff,
        ema_id=str(cutoff),
    )


def test_overlap_is_transitive_but_never_merges_intervals():
    f = pd.DataFrame(
        [
            view("synthetic_a", 1, 5),
            view("synthetic_a", 5, 10),
            view("synthetic_a", 9, 15),
            view("synthetic_a", 16, 20),
            view("synthetic_b", 1, 5),
        ]
    )
    grouped, selected = choose_runs(f)
    assert grouped.component.iloc[0] == grouped.component.iloc[2]
    assert grouped.component.iloc[2] != grouped.component.iloc[3]
    assert grouped.component.nunique() == 3
    assert selected.first_logical_row.tolist() == [1, 1]
    assert selected.last_logical_row.tolist() == [5, 5]


def test_selection_deterministic_caps_and_hypotheses_independent():
    rows = [
        view(f"synthetic_{i}", i * 100, i * 100 + 10, s)
        for s in ["interval_end", "interval_start"]
        for i in range(14)
    ]
    rows += [view("synthetic_0", 0, 10, cutoff=1100)]
    f = pd.DataFrame(rows)
    _, chosen = choose_runs(f)
    _, shuffled = choose_runs(f.sample(frac=1, random_state=7).reset_index(drop=True))
    pd.testing.assert_frame_equal(chosen, shuffled)
    assert chosen.groupby("scenario").size().tolist() == [10, 10]
    assert not chosen.duplicated(["scenario", "participant"]).any()
    assert chosen.ema_cutoff.eq(1000).all()
    with pytest.raises(ValueError):
        choose_runs(f, 11)
    assert choose_runs(f.iloc[:0])[1].empty
    assert overlap_components(f.iloc[:0]).empty


def test_guard_rejects_unapproved_entry_before_any_reads(tmp_path):
    with pytest.raises(AuditError, match="explicit limited-pilot"):
        calculate(tmp_path, tmp_path, tmp_path)


def long_run(policy, spacing=0.8, scenario="interval_end"):
    t = np.arange(1.0, 299.0, spacing)
    c = SyntheticCase("long_synthetic", tuple(t), (800.0,) * len(t), tuple(range(len(t))), scenario)
    h = window_input(c.frame(), 300.0, policy, scenario)
    assert len(h.runs) == 1
    return h.records[h.records.c_run_id == 0].copy()


@pytest.mark.parametrize("scenario", ["interval_end", "interval_start"])
def test_duration_gate_and_clock_preservation(policy, scenario):
    r = long_run(policy, scenario=scenario)
    original = r.copy(deep=True)
    span, support = guard_run(r, scenario, screen_eligible=True, development=True)
    assert span >= 240 and support >= 240
    ref = reference_run(r)
    assert set(ref) == set(FEATURES)
    assert ref["mean_ibi_ms"] == ref["median_ibi_ms"] == 800
    assert ref["sdnn_sample_ms"] == ref["rmssd_ms"] == ref["pnn50_pairs_percent"] == 0
    assert package_comparison(r, scenario, "heartpy")["status"] == "compared"
    pd.testing.assert_frame_equal(r, original)
    for kw in [
        dict(screen_eligible=False, development=True),
        dict(screen_eligible=True, development=False),
        dict(screen_eligible=True, development=True, span_min=120),
    ]:
        with pytest.raises(ValueError):
            guard_run(r, scenario, **kw)


def test_span_and_support_are_separate_gates(policy):
    r = long_run(policy, spacing=1.0)
    assert r.n_epoch_s.iloc[-1] - r.n_epoch_s.iloc[0] >= 240
    with pytest.raises(ValueError, match="span/support"):
        guard_run(r, "interval_end", screen_eligible=True, development=True)
    r = long_run(policy).iloc[:200]
    with pytest.raises(ValueError, match="span/support"):
        guard_run(r, "interval_end", screen_eligible=True, development=True)


@pytest.mark.parametrize("change", ["remove", "link", "duplicate", "rejected"])
def test_no_new_pairs_across_rejections_or_removed_records(policy, change):
    r = long_run(policy)
    if change == "remove":
        r = r.drop(r.index[100])
    elif change == "link":
        r.loc[r.index[100], "c_link_to_previous_original_record"] = False
    elif change == "duplicate":
        r.loc[r.index[100], "n_epoch_s"] = r.n_epoch_s.iloc[99]
    else:
        r.loc[r.index[100], "c_acceptable"] = False
    with pytest.raises(ValueError):
        reference_run(r)
    with pytest.raises(ValueError):
        package_comparison(r, "interval_end", "heartpy")


@pytest.mark.parametrize("name", ["known_end", "known_start", "strict_50ms_threshold"])
def test_pilot_adapter_matches_original_synthetic_adapter(policy, name):
    c = next(c for c in fixtures() if c.name == name)
    r = window_input(c.frame(), c.window_end, policy, c.scenario).records
    reference = reference_run(r)
    for backend in ["heartpy", "neurokit2"]:
        result = package_comparison(r, c.scenario, backend)
        if backend == "neurokit2" and c.scenario == "interval_start":
            assert result["status"] == "blocked"
            continue
        old = package_run(c, r, backend)
        for feature, value in result["values"].items():
            if value is not None:
                assert value == pytest.approx(reference[feature], abs=1e-8)
                assert value == old[feature]
    if name == "strict_50ms_threshold":
        assert reference["pnn50_pairs_percent"] == pytest.approx(100 / 3)


def test_neurokit_blocks_incompatible_clock_before_call(policy, monkeypatch):
    import neurokit2 as nk

    def forbidden(*args, **kwargs):
        raise AssertionError("Package must not be called on incompatible inputs")

    monkeypatch.setattr(nk, "hrv_time", forbidden)
    r = long_run(policy, spacing=0.81)
    r["n_epoch_s"] = r.n_epoch_s + np.arange(len(r)) * 0.01
    assert package_comparison(r, "interval_end", "neurokit2")["status"] == "blocked"
    assert package_comparison(r, "interval_start", "neurokit2")["status"] == "blocked"


def test_corrected_long_lookback_grid_never_wraps_or_crosses_cutoff(policy):
    from garmin_ibi_pilot.windows import corrected_windows

    root = Path(__file__).parents[1]
    cfg = json.loads((root / "config/ibi_eligibility.json").read_text())
    # Reproduce historical overflow; expected clocks are hand-specified seconds before cutoff.
    steps = np.array([0, 546, 547, 1092, 1093, 1435], dtype=np.int16)
    w = pd.DataFrame(
        dict(
            n_ema_id=["synthetic_event"] * 6,
            n_step_index=steps,
            n_window_end_epoch=100000.0 - 60 * steps,
        )
    )
    for scenario in ["interval_end", "interval_start"]:
        w[f"screen__{scenario}__eligible"] = False
        w[f"screen__{scenario}__longest_sequence_seconds"] = 0.0
    ema = pd.DataFrame(dict(n_cutoff_epoch=[100000.0]), index=["synthetic_event"])
    corrected = corrected_windows(w, ema, None, policy, cfg)
    assert corrected.n_window_end_epoch.tolist() == [
        100000.0,
        67240.0,
        67180.0,
        34480.0,
        34420.0,
        13900.0,
    ]
    assert corrected.historical_window_clock_corrected.tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert corrected.n_window_start_epoch.min() == 13600.0
    assert corrected.n_window_end_epoch.max() == 100000.0
    assert not corrected["screen__interval_end__eligible"].any()
    pd.testing.assert_frame_equal(corrected, corrected_windows(w, ema, None, policy, cfg))
