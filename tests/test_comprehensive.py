"""Independent arithmetic, boundary, broken adjacency and train-only checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from garmin_comprehensive.features import observed_features, activity_features, coobserved

CFG = json.loads(Path("config/comprehensive_features.json").read_text())


def readings(t, v, good=None):
    return pd.DataFrame(
        dict(
            n_epoch_s=np.array(t, float),
            n_value=np.array(v, float),
            a_acceptable=np.ones(len(t), bool) if good is None else good,
        )
    )


def test_statistics_and_original_pair_barrier():
    g = readings([0, 30, 60, 90, 120], [60, 80, 100, 120, 140], [True, True, False, True, True])
    f, q, _, _ = observed_features(g, 0, 180, CFG)
    v = np.array([60, 80, 120, 140.0])
    assert f["mean"] == 100 and f["median"] == 100
    assert f["variance"] == pytest.approx(4000 / 3)
    assert f["mad"] == 30 and f["net_change"] == 80
    assert f["mean_absolute_change"] == 20  # 80 -> 120 is never invented
    assert f["rms_change"] == 20 and q["valid_adjacent_pairs"] == 2
    assert f["slope"] == pytest.approx(40)
    assert f["cv"] == pytest.approx(v.std(ddof=1) / 100)


def test_gap_and_half_support():
    f, q, _, _ = observed_features(readings([0, 10, 200, 210], [60, 70, 80, 90]), 0, 300, CFG)
    assert q["valid_adjacent_pairs"] == 2 and f["rms_change"] == 10
    assert f["half_mean_change"] == 20
    assert q["maximum_internal_gap_seconds"] == 190
    assert np.isnan(f["skewness"])


@pytest.mark.parametrize("minutes", [5, 15, 30, 60, 120, 360, 720, 1440])
def test_boundaries_python_datetime(minutes):
    from datetime import datetime, timedelta, timezone

    cutoff = datetime(2030, 1, 2, 12, 0, tzinfo=timezone.utc)
    start = (cutoff - timedelta(minutes=minutes)).timestamp()
    end = cutoff.timestamp()
    g = readings([start - 1, start, start + 1, end - 1, end], [240, 60, 70, 80, 240])
    f, _, _, bounds = observed_features(g, start, end, CFG)
    assert f["mean"] == 70 and f["first"] == 60 and f["latest"] == 80
    assert bounds == (1, 4)


def test_constant_and_missing_not_zero():
    f, _, reasons, _ = observed_features(readings([], []), 0, 300, CFG)
    assert all(np.isnan(v) for v in f.values())
    assert set(reasons.values()) == {"no_recording"}
    f, _, reason, _ = observed_features(readings(np.arange(10) * 10, [80] * 10), 0, 300, CFG)
    assert f["sd"] == 0 and np.isnan(f["skewness"]) and reason["skewness"] == "constant_values"


def test_pair_bounds_resets_and_observed_zero():
    g = readings([0, 60, 120, 180], [10, 20, 2, 2])
    g["a_pair_start"] = [-60, 0, 60, 120]
    g["a_pair_duration"] = [60] * 4
    g["a_pair_delta"] = [10, 10, -18, 0]
    g["a_pair_accepted"] = [True, True, False, True]
    g["a_pair_rejection_mask"] = [0, 0, 128, 0]
    f, q, _ = activity_features(g, 0, 180)
    assert f["added_supported"] == 10 and q["supported_seconds"] == 60
    f, q, _ = activity_features(g, 120, 240)
    assert f["added_supported"] == 0 and f["zero_change_supported_fraction"] == 1
    f, _, _ = activity_features(g, 61, 119)
    assert np.isnan(f["added_supported"])


def test_counter_overlap_not_future():
    h = readings([0, 30, 60, 90, 120, 150], [60, 70, 80, 90, 100, 110])
    p = readings([90, 180], [3, 5])
    p["a_pair_start"] = [0, 90]
    p["a_pair_accepted"] = [True, True]
    p["a_pair_delta"] = [0, 2]
    f = coobserved(h, p, 0, 180)
    assert f["hr_zero_step_mean"] == 70
    assert np.isnan(f["hr_positive_step_mean"])  # endpoint180 not in observation window


def test_observed_minutes_not_reading_weighted():
    f, _, _, _ = observed_features(readings([0, 1, 2, 60], [60, 60, 60, 120]), 0, 300, CFG)
    assert f["mean"] == 75 and f["observed_minute_mean"] == 90
    assert np.isnan(f["rolling5_mean"])


def test_train_only_imputation_and_correlation():
    from garmin_comprehensive.models import Transform

    tr = pd.DataFrame(
        dict(
            local_time_sin=[0, 1, 0, 1],
            local_time_cos=[1, 0, 1, 0],
            n_survey_category=["Morning"] * 4,
            a=[1, 2, 3, np.nan],
            b=[2, 4, 6, np.nan],
            empty=[np.nan] * 4,
        )
    )
    t = Transform(["a", "b", "empty"], correlation=0.95).fit(tr)
    assert t.medians["a"] == 2 and t.dropped["b"] == "training_correlation"
    assert t.dropped["empty"] == "all_missing"
    va = tr.copy()
    va["a"] = [10000, np.nan, 10000, np.nan]
    va["b"] = 999
    x = t.transform(va)
    assert t.medians["a"] == 2 and np.isfinite(x).all()
    assert "missing__a" in t.names and "empty" not in t.names
    with pytest.raises(Exception, match="Forbidden"):
        Transform(["pss_combo"])


def test_synthetic_grouped_pipeline_positive_control():
    from garmin_comprehensive.models import nested_fit

    rng = np.random.default_rng(4)
    rows = []
    for person in range(20):
        for i in range(6):
            target = i % 2
            g = readings([0, 60, 120], [60 + target * 40 + rng.normal() for _ in range(3)])
            values, _, _, _ = observed_features(g, 0, 300, CFG)
            rows.append(
                dict(
                    n_participant=f"SYNTHETIC_{person}",
                    n_fold=person % 5,
                    n_survey_category="Morning",
                    local_time_sin=0.0,
                    local_time_cos=1.0,
                    hr_mean=values["mean"],
                    target=target,
                )
            )
    f = pd.DataFrame(rows)
    train = f[f.n_fold != 0]
    valid = f[f.n_fold == 0]
    spec = dict(method="l2", block="hr", flags=False, handling="imputed")
    config = {**CFG, "C_grid": [0.1]}
    t, m, p, _, _, _, splits = nested_fit(
        train, valid.drop(columns="target"), "target", 2, spec, {"hr": ["hr_mean"]}, config
    )
    assert (p.argmax(axis=1) == valid.target).mean() > 0.95
    assert all(
        not set(train.iloc[a].n_participant) & set(train.iloc[b].n_participant) for a, b in splits
    )
    changed = valid.drop(columns="target").copy()
    changed["local_time_sin"] = 100
    _, m2, _, _, _, _, _ = nested_fit(
        train, changed, "target", 2, spec, {"hr": ["hr_mean"]}, config
    )
    np.testing.assert_allclose(m.coef_, m2.coef_)


def test_past_reference_excludes_current_and_future_and_zero_sd():
    from garmin_comprehensive.augment import past_reference

    cutoff = 100000.0
    m = 60
    end = cutoff - 3600
    t = np.r_[
        end - 10800 + np.arange(15) * 60,
        end - 7200 + np.arange(15) * 60,
        end - 3600 + np.arange(15) * 60,
        end,
        cutoff,
    ]
    v = np.r_[np.full(45, 80.0), 200.0, 240.0]
    out, q = past_reference(readings(t, v), cutoff, m, 90, 100)
    assert out["reference_mean"] == 80 and out["mean_deviation"] == 10
    assert out["reference_sd"] == 0 and np.isnan(out["standardized_mean_deviation"])
    assert q["occupied_minutes"] == 45 and q["reference_end"] == end
    for minutes in CFG["lookbacks_minutes"]:
        _, q = past_reference(readings(t, v), cutoff, minutes, 90, 100)
        assert q["reference_end"] == cutoff - minutes * 60
        assert q["reference_start"] == q["reference_end"] - 86400


def test_experiment_scheduler_never_duplicates_quality_control():
    from garmin_comprehensive.models import feature_blocks, specs
    from garmin_comprehensive.dictionary import dictionary

    blocks = feature_blocks(dictionary(CFG))
    blocks["quality"] = ["quality__hr__count"]
    plan = specs(blocks)
    assert len(plan) == len({x["name"] for x in plan})
    assert sum(x["name"] == "quality" for x in plan) == 1
