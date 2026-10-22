"""Synthetic-only tests for lineage, density, end-to-end linkage and nested isolation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from garmin_ema_baseline.models import feature_columns
from garmin_inventory.privacy import AuditError
from garmin_window_benchmark.information import compare, density
from garmin_window_benchmark.models import fit_predict, inner_splits, tune
from garmin_window_benchmark.synthetic import positive_control
from garmin_window_benchmark.trace import scalar_features

ROOT = Path(__file__).parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "config/window_benchmark.json").read_text())


def readings(times, values, rows=None):
    return pd.DataFrame(
        dict(
            n_epoch_s=times,
            n_value=values,
            n_source_sha256="synthetic_hash",
            n_source_row=np.arange(len(times)) if rows is None else rows,
        )
    )


def test_source_identity_is_not_repeated_value():
    a = readings([0, 60, 120], [70, 70, 70], [1, 2, 3])
    b = readings([60, 120], [70, 70], [2, 3])
    r = compare(a, b, [70, 0, 70, 0], [70, 0, 70, 0], [1, 1, 1, 0.1])
    assert not r["identical_sets"] and r["shared"] == 2 and r["only_a"] == 1
    assert r["identical_latest_identity"] and r["identical_vector"]
    c = readings([121], [70], [4])
    assert not compare(b, c, [70, 0, 70, 0], [70, 0, 70, 0], [1, 1, 1, 0.1])[
        "identical_latest_identity"
    ]


def test_density_empty_minutes_and_partial_boundary():
    f = readings([0, 1, 2, 60, 180], [60, 60, 60, 120, 90])
    d = density(f, 0, 181)
    assert d["occupied_minutes"] == 3 and d["possible_minutes"] == 4
    assert d["reading_mean"] == 78 and d["observed_minute_mean"] == 90
    assert d["dense_decile_reading_share"] == 0.6 and d["dense_decile_mean_contribution_bpm"] == 36
    assert density(f.iloc[:0], 0, 181)["readings"] == 0


def synthetic_frame():
    n = 90
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        dict(
            n_ema_id=[f"synthetic_{i}" for i in range(n)],
            n_participant=np.repeat(np.arange(15), 6),
            n_survey_category=np.tile(["Morning", "Midday", "Endday"], 30),
            local_time_sin=rng.uniform(-1, 1, n),
            local_time_cos=rng.uniform(-1, 1, n),
            pss_combo=rng.integers(3, 16, n),
        )
    )
    for col in feature_columns(60):
        frame[col] = rng.normal(size=n)
    return frame


def test_grouped_inner_splits_partition_outer_training():
    f = synthetic_frame()
    train = f[f.n_participant.lt(12)]
    splits = inner_splits(train)
    seen = []
    for a, b in splits:
        assert not set(train.iloc[a].n_participant) & set(train.iloc[b].n_participant)
        assert set(train.iloc[a].n_participant) | set(train.iloc[b].n_participant) == set(
            train.n_participant
        )
        seen.extend(b)
    assert sorted(seen) == list(range(len(train)))
    assert inner_splits(f[f.n_participant.lt(2)]) == []


def test_training_only_scaling_and_predictor_allowlist(config):
    f = synthetic_frame()
    train = f[f.n_participant.lt(12)].copy()
    valid = f[f.n_participant.ge(12)].copy()
    valid.loc[:, feature_columns(60)] = 1000
    valid["pss_combo"] = np.nan  # prediction API must not need validation outcomes
    train["forbidden_quality"] = 999
    valid["forbidden_quality"] = -999
    p, _, a = fit_predict(train, valid, "context_hr", "fixed_ridge", {"alpha": 1}, config)
    assert np.isfinite(p).all() and "forbidden_quality" not in a["input_columns"]
    assert a["training_count"] == len(train)
    assert np.allclose(a["training_mean"], train[a["input_columns"][:-1]].mean())
    with pytest.raises(AuditError):
        fit_predict(train, train, "context_hr", "fixed_ridge", {"alpha": 1}, config)
    valid.loc[valid.index[0], feature_columns(60)[0]] = np.nan
    with pytest.raises(AuditError):
        fit_predict(train, valid, "context_hr", "hgb", {"max_depth": 1, "max_iter": 50}, config)


def test_inner_selection_uses_mean_mae_and_fixed_grid(config):
    train = synthetic_frame()
    splits = inner_splits(train)
    setting, score, audits = tune(train, "context", "tuned_ridge", config, splits)
    scores = [np.mean([a["mae"] for a in audits if a["candidate"] == i]) for i in range(4)]
    assert (
        score == min(scores) and setting["alpha"] == config["ridge_alphas"][int(np.argmin(scores))]
    )
    assert len(audits) == 12


def test_independent_scalar_hr_reference():
    assert scalar_features([0, 60, 120], [60, 90, 120]) == [90, 30, 120, 30]


def test_actual_linkage_feature_model_positive_control(config):
    result = positive_control(json.loads((ROOT / "config/hr_activity.json").read_text()), config)
    assert result["synthetic_only"] and result["responses"] == 300
    assert result["fixed_ridge_mae"] < 0.1 and result["hgb_mae"] < 0.3
    assert result["median_mae"] > 3
