"""Synthetic leakage, split, estimator and matched regression checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from garmin_comprehensive_regression.models import (
    FAMILIES,
    estimator,
    fit_checked,
    grid,
    nested_fit,
    predict,
    prepare,
    specifications,
)
from garmin_comprehensive_regression.run import basic
from garmin_inventory.privacy import AuditError

CFG = json.loads(Path("config/comprehensive_regression.json").read_text())


def synthetic():
    rng = np.random.default_rng(14)
    rows = []
    for person in range(18):
        for j in range(12):
            x = rng.normal()
            rows.append(
                dict(
                    n_ema_id=f"synthetic_{person}_{j}",
                    n_participant=f"synthetic_{person}",
                    n_fold=person % 3,
                    n_survey_category=["Morning", "Midday", "Endday"][j % 3],
                    local_time_sin=0.0,
                    local_time_cos=1.0,
                    hr=x,
                    sparse=x if j % 3 else np.nan,
                    pss_combo=7 + 2 * x,
                )
            )
    return pd.DataFrame(rows)


def spec(**kw):
    return dict(view="full", block="full", cohort="all_valid", flags=True, family="ridge", **kw)


def test_schedule_and_inputs():
    s = specifications()
    assert len(s) == 98 and len({x["name"] for x in s}) == 98
    assert {x["family"] for x in s} == set(FAMILIES) | {"fixed_ridge"}
    assert all(x["block"] != "pulseox_full" or x["view"] == "pulseox_full" for x in s)


@pytest.mark.parametrize("family", FAMILIES)
def test_installed_regressor_interfaces(family):
    f = synthetic()
    x, y = f[["hr"]].to_numpy(), f.pss_combo.to_numpy()
    model, converged, _ = fit_checked(estimator(family, grid(family, CFG)[0], CFG), x, y)
    assert converged and np.isfinite(predict(model, x)).all()
    if family == "hgb":
        assert not model.do_early_stopping_


def test_grouped_synthetic_signal_and_validation_isolation():
    f = synthetic()
    train, valid = f[f.n_fold.ne(0)], f[f.n_fold.eq(0)].drop(columns="pss_combo")
    model, best, _, pairs, _ = nested_fit(train, valid, spec(), ["hr", "sparse"], CFG)
    assert np.abs(model.predict(valid) - f.loc[valid.index, "pss_combo"]).mean() < 0.1
    assert all(
        not set(train.iloc[a].n_participant) & set(train.iloc[b].n_participant) for a, b in pairs
    )
    poison = valid.copy()
    poison["hr"] = 1e6
    other, best2, _, _, _ = nested_fit(train, poison, spec(), ["hr", "sparse"], CFG)
    assert best == best2
    np.testing.assert_allclose(model.model.coef_, other.model.coef_)
    with pytest.raises(AuditError, match="validation outcomes"):
        nested_fit(train, f[f.n_fold.eq(0)], spec(), ["hr"], CFG)


def test_train_imputation_and_selector_context_preservation():
    f = synthetic()
    train, valid = f[f.n_fold.ne(0)].copy(), f[f.n_fold.eq(0)].copy()
    train["empty"] = np.nan
    valid["empty"] = 999
    s = spec()
    s["view"] = "full_l1select"
    transform, selector, keep, _, _ = prepare(train, valid, s, ["hr", "sparse", "empty"], CFG)
    assert "empty" in transform.dropped and "empty" not in transform.kept
    assert transform.medians["sparse"] == train["sparse"].median()
    assert selector is not None and keep.any()
    assert all(keep[i] for i, n in enumerate(transform.names) if n.startswith("n_survey_category_"))
    with pytest.raises(AuditError):
        prepare(train, valid, s, ["pss_combo"], CFG)


def test_metrics_constant_predictions_and_no_clipping():
    a = basic(np.array([3.0, 5.0, 10.0]), np.array([5.0, 5.0, 5.0]))
    assert a["mae"] == pytest.approx(7 / 3) and a["spearman"] is None
    b = basic(np.array([3.0, 15.0]), np.array([1.0, 17.0]))
    assert b["outside_target_range"] == 1 and b["rmse"] == 2


def test_nonfinite_estimator_default_is_metadata_not_imputation():
    from garmin_comprehensive_regression.run import json_parameters

    model = estimator("xgb", grid("xgb", CFG)[0], CFG)
    assert np.isnan(model.get_params()["missing"])
    encoded = json_parameters(model.get_params())
    assert encoded["missing"] == {"nonfinite_default": "nan"}
    json.dumps(encoded, allow_nan=False)
    assert np.isnan(model.get_params()["missing"])


def test_paired_cluster_interval_and_membership():
    from garmin_comprehensive_regression.evaluate import paired, cluster_values

    f = synthetic()
    f["prediction"] = f.pss_combo + 1
    g = f.copy()
    g["prediction"] = g.pss_combo + 2
    cfg = dict(bootstrap_draws=100, seed=14)
    result = paired(f, g, cfg)
    assert result["mae_difference"] == pytest.approx(-1)
    assert result["mae_low"] == pytest.approx(-1) and result["mae_high"] == pytest.approx(-1)
    assert result["full_membership_match"]
    assert not paired(f, g.iloc[10:], cfg)["full_membership_match"]
    bad = g.copy()
    bad["pss_combo"] += 1
    with pytest.raises(AssertionError):
        paired(f, bad, cfg)
    np.testing.assert_allclose(cluster_values(f, 100, 14)["mae"], np.ones(100))


def test_compiled_privacy_scan_preserves_token_boundaries():
    from garmin_comprehensive_regression.privacy import token_matcher
    from garmin_inventory.privacy import contains_private_token

    ids = ["1234", "12345", "alpha-23", "a.b", "abc"]
    names = ["Example Person", "record_file.csv", "a+b[1]"]
    check = token_matcher(ids, names)
    examples = [
        "x12345y",
        "12345",
        "(1234)",
        "hash12345ab",
        "x_1234_",
        "ALPHA-23",
        "alpha-23",
        "a.b",
        "Example Person archive",
        "Example PersonX",
        "record_file.csv",
        "a+b[1]",
        "none",
    ]
    assert all(check(x) == contains_private_token(x, ids, names) for x in examples)


def test_only_authorized_aggregate_notebook_accepts_executed_output():
    from garmin_inventory.privacy import validate_notebook_output

    n = dict(
        metadata=dict(garmin_public_aggregate_notebook=True),
        cells=[
            dict(
                cell_type="code",
                execution_count=1,
                outputs=[dict(output_type="display_data", data={"text/plain": "aggregate"})],
            )
        ],
    )
    validate_notebook_output(Path("14_comprehensive_pss_regression.ipynb"), n)
    with pytest.raises(AuditError):
        validate_notebook_output(Path("11_measurement_source_and_timestamp_audit.ipynb"), n)
