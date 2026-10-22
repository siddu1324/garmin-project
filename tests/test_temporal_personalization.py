import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from garmin_temporal_personalization.history import (
    build_history,
    eligible_history,
    chronological_population,
    online_baseline,
    personalized_probability,
    HISTORY,
)
from garmin_temporal_personalization.data import clock_reference, block_rate
from garmin_temporal_personalization.models import specifications, nested


def frame():
    return pd.DataFrame(
        dict(
            n_ema_id=list("abcdef"),
            n_participant=["one"] * 6,
            n_actual_epoch=np.arange(6) * 100.0,
            n_cutoff_epoch=np.arange(6) * 100.0 - 1,
            n_available_epoch=np.arange(6) * 100.0 + 10,
            n_availability_verified=True,
            pss_combo=[3.0, 4.0, 5.0, 7.0, 10.0, 12.0],
            item_nervous=2.0,
            item_cope=3.0,
            item_control=4.0,
        )
    )


def test_unknown_availability_never_assumed_and_current_future_hidden():
    f = frame()
    f.n_available_epoch = np.nan
    f.n_availability_verified = False
    h, lineage = build_history(f)
    assert h[HISTORY].isna().all().all() and len(lineage) == 0
    f = frame()
    a, _ = build_history(f)
    f.loc[4:, "pss_combo"] = 15
    b, _ = build_history(f)
    pd.testing.assert_frame_equal(a.iloc[:5], b.iloc[:5])


def test_strict_cutoff_late_answers_ties_and_wrong_person():
    f = frame()
    f.loc[0, "n_available_epoch"] = f.loc[4, "n_cutoff_epoch"]
    f.loc[1, "n_available_epoch"] = 900
    f.loc[2, "n_participant"] = "other"
    h = eligible_history(f, next(f.iloc[[4]].itertuples()))
    assert h.n_ema_id.tolist() == ["d"]
    f.loc[2, "n_participant"] = "one"
    f.loc[2, "n_actual_epoch"] = 300
    f.loc[2, "n_available_epoch"] = 310
    assert len(eligible_history(f, next(f.iloc[[4]].itertuples()))) == 0


def test_history_arithmetic_and_chronological_population():
    f = frame()
    h, lineage = build_history(f)
    r = h.iloc[-1]
    assert r.previous_pss == 10 and r.previous2_pss == 7 and r.previous3_pss == 5
    assert r.historical_mean == pytest.approx(np.mean([3, 4, 5, 7, 10]))
    assert r.historical_sd == pytest.approx(np.std([3, 4, 5, 7, 10], ddof=1))
    assert (lineage.source_available_epoch < lineage.cutoff_epoch).all()
    other = f.copy()
    other.n_participant = "other"
    other.n_ema_id += "x"
    pop = chronological_population(pd.concat([f, other]), {"one"}, 150)
    assert len(pop) == 2 and pop.n_participant.eq("other").all()
    assert personalized_probability(0.2, [0, 1], 5) == pytest.approx(2 / 7)


def test_prequential_calibration_no_future_population_answers():
    f = frame()
    other = f.copy()
    other.n_participant = "other"
    other.n_ema_id += "x"
    both = pd.concat([f, other], ignore_index=True)
    p = online_baseline(both, {"one"}, 5)
    assert len(p) == 1 and p.historical_responses.iloc[0] == 5
    both.loc[both.n_actual_epoch >= 500, "pss_combo"] = 3
    q = online_baseline(both, {"one"}, 5)
    assert p.probability.iloc[0] == q.probability.iloc[0]
    both.n_availability_verified = False
    assert online_baseline(both, {"one"}, 5).empty


def test_clock_reference_bounds_midnight_and_support():
    cfg = json.loads(Path("config/temporal_personalization.json").read_text())
    c = 8 * 86400 + 10
    t = np.concatenate([c - d * 86400 + np.arange(10) * 60 for d in [1, 2, 3]])
    v = np.arange(30) + 60.0
    # current hour and left-outside boundary cannot influence the reference.
    mu, q = clock_reference(np.r_[t, c, c - 7 * 86400 - 1], np.r_[v, 1000, 1000], c, cfg)
    assert mu == pytest.approx(v.mean()) and q["recorded_minutes"] == 30
    assert q["selected_last"] < c - 3600 and q["selected_first"] >= c - 7 * 86400
    assert np.isnan(clock_reference(t[:20], v[:20], c, cfg)[0])


def test_counter_boundaries_rejection_and_supported_duration():
    p = pd.DataFrame(
        dict(
            a_pair_start=[0, 10, 20, 30, 40],
            n_epoch_s=[10, 20, 30, 40, 50],
            a_pair_delta=[2, 3, -100, 4, 999],
            a_pair_duration=10,
            a_pair_accepted=[True, True, False, True, True],
        )
    )
    rate, seconds = block_rate(p, 5, 50, 20)
    assert seconds == 20 and rate == pytest.approx(21)
    assert np.isnan(block_rate(p, 5, 50, 21)[0])


def test_actual_nested_pipeline_recovers_synthetic_signal():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(19)
    x = rng.normal(size=240)
    f = pd.DataFrame(
        dict(
            n_participant=np.repeat(np.arange(40), 6),
            n_fold=np.repeat(np.arange(40) % 5, 6),
            n_survey_category=np.tile(["Morning", "Midday"], 120),
            local_time_sin=0.5,
            local_time_cos=0.7,
            physiology=x,
            target_A=(x > 0).astype(int),
        )
    )
    spec = dict(
        name="synthetic",
        columns=["physiology"],
        context=False,
        missing=False,
        cohort="all",
        family="logistic",
        permutation=None,
    )
    cfg = json.loads(Path("config/temporal_personalization.json").read_text())
    train, valid = f[f.n_fold.ne(0)], f[f.n_fold.eq(0)]
    fitted, _, _, splits = nested(train, "A", spec, cfg)
    assert roc_auc_score(valid.target_A, fitted.probability(valid.drop(columns="target_A"))) > 0.99
    for a, b in splits:
        assert not set(train.iloc[a].n_participant) & set(train.iloc[b].n_participant)
    assert fitted.transform.names == ["physiology"]


def test_fixed_registry_no_history_targets_or_quality_in_physiology():
    specs = specifications()
    assert len(specs) == 21 and len({s["name"] for s in specs}) == 21
    assert all(not set(s["columns"]) & set(HISTORY) for s in specs)
    assert all(not c.startswith(("item_", "target_", "n_")) for s in specs for c in s["columns"])
