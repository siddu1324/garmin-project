import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from garmin_binary_6hour.metrics import bootstrap, score, weighted_ranking
from garmin_binary_6hour.models import (
    blocks,
    cohort,
    make_transform,
    nested_fit,
    targets,
)


def config():
    return json.loads(Path("config/binary_6hour.json").read_text())


def synthetic():
    rng = np.random.default_rng(11)
    n = 360
    x = rng.normal(size=n)
    return pd.DataFrame(
        dict(
            n_participant=np.repeat(np.arange(36), 10),
            n_fold=np.repeat(np.arange(36) % 5, 10),
            n_ema_id=np.arange(n),
            n_survey_category=np.tile(["Morning", "Midday", "Endday"], 120),
            local_time_sin=rng.normal(size=n),
            local_time_cos=rng.normal(size=n),
            hr__mean__360m=x,
            empty=np.nan,
            constant=1.0,
            target_A=(x > 0).astype(int),
        )
    )


def test_exact_target_boundaries_and_missing():
    y = pd.Series([3.0, 4.0, 5.0, 9.0, 10.0, 15.0, np.nan])
    np.testing.assert_equal(targets(y, 5).to_numpy(), [0, 0, 1, 1, 1, 1, np.nan])
    np.testing.assert_equal(targets(y, 10).to_numpy(), [0, 0, 0, 0, 1, 1, np.nan])


def test_window_allowlist_excludes_extended_support():
    from garmin_comprehensive.dictionary import dictionary
    from garmin_comprehensive.augment import HISTORY_METRICS, history_name

    cfg = json.loads(Path("config/comprehensive_features.json").read_text())
    entries = [d for d in dictionary(cfg) if d["modality"] != "history"]
    entries += [
        dict(name=history_name(metric, m), modality="history", lookback_minutes=m)
        for m in cfg["lookbacks_minutes"]
        for metric in HISTORY_METRICS
    ]
    b = blocks(entries)
    for m in [60, 120, 360]:
        assert len(b[f"full{m}"]) == 47
        assert not any(
            c.startswith(("history", "multiscale", "pulseox", "quality")) for c in b[f"full{m}"]
        )
    assert len(b["multiscale"]) == 438


def test_training_transform_and_availability():
    f = synthetic()
    f.loc[:250, "hr__mean__360m"] = np.nan
    spec = dict(mode="availability")
    t = make_transform(f, ["hr__mean__360m", "empty", "constant"], spec, config())
    assert "hr__mean__360m" not in t.kept
    assert t.dropped["hr__mean__360m"] == "training_availability_below_50pct"
    assert "empty" not in t.kept and "constant" not in t.kept
    fresh = synthetic()
    fresh["hr__mean__360m"] = 1e9
    assert np.isfinite(t.transform(fresh)).all()
    t2 = make_transform(synthetic(), ["hr__mean__360m"], dict(mode="main"), config())
    before = t2.medians.copy()
    t2.transform(fresh)
    pd.testing.assert_series_equal(before, t2.medians)


def test_nested_positive_control_and_validation_labels_excluded():
    f = synthetic()
    train, valid = f[f.n_fold != 0], f[f.n_fold == 0]
    cfg = {**config(), "C_grid": [0.01, 1], "ratios": [0.0, 0.5, 1.0]}
    spec = dict(mode="main", weighted=False)
    fitted, best, trials, pairs = nested_fit(train, "A", spec, ["hr__mean__360m"], cfg)
    p = fitted.probability(valid.drop(columns="target_A"))
    assert roc_auc_score(valid.target_A, p) > 0.95
    assert len(trials) == 6 and best["mean_auroc"] > 0.95
    for a, b in pairs:
        assert not set(train.iloc[a].n_participant) & set(train.iloc[b].n_participant)
    assert fitted.model.classes_.tolist() == [0, 1]


def test_ranking_bootstrap_matches_sklearn_with_ties_and_weights():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.2, 0.9, 0.9, 0.2])
    weights = np.array([[1, 1, 1, 1, 1, 1], [0, 2, 2, 1, 1, 3], [0, 1, 1, 0, 1, 0]])
    a, b = weighted_ranking(y, p, weights)
    for i in [0, 1]:
        assert np.isclose(a[i], roc_auc_score(y, p, sample_weight=weights[i]))
        assert np.isclose(b[i], average_precision_score(y, p, sample_weight=weights[i]))
    assert np.isnan(a[2]) and np.isnan(b[2])
    z = score(y, np.ones(6))
    assert z["auroc"] == 0.5 and z["ap"] == 0.5
    assert z["balanced_accuracy"] == 0.5 and z["specificity"] == 0


def test_common_complete_cohort_matches_windows():
    f = synthetic().assign(pss_combo=5.0, context_available=True)
    b = {}
    for m in [60, 120, 360]:
        c = f"value{m}"
        f[c] = 1.0
        b[f"original{m}"] = [c]
    f.loc[0, "value60"] = np.nan
    f.loc[1, "value360"] = np.nan
    assert len(cohort(f, b, "all")) == len(f)
    assert len(cohort(f, b, "complete")) == len(f) - 2


def test_cluster_draws_are_paired_and_response_order_independent():
    f = pd.DataFrame(
        dict(
            n_participant=[0, 0, 1, 1, 2, 2, 3, 3],
            target=[0, 1, 0, 1, 0, 1, 0, 1],
            probability=[0.1, 0.8, 0.3, 0.8, 0.4, 0.6, 0.6, 0.9],
        )
    )
    cfg = dict(seed=21, bootstrap_draws=200)
    a = bootstrap(f, cfg)
    b = bootstrap(f.iloc[::-1], cfg)
    np.testing.assert_allclose(a, b, atol=1e-14)
    np.testing.assert_array_equal(a - bootstrap(f, cfg), np.zeros_like(a))


def test_notebook15_authorization_rejects_errors():
    import nbformat
    import pytest
    from garmin_binary_6hour.privacy import validate_15
    from garmin_inventory.privacy import AuditError

    n = nbformat.v4.new_notebook(metadata=dict(garmin_public_aggregate_notebook=True))
    c = nbformat.v4.new_code_cell('print("aggregate")', execution_count=1)
    n.cells = [c]
    validate_15(n)
    c.outputs = [nbformat.v4.new_output("error", ename="Error", evalue="synthetic", traceback=[])]
    with pytest.raises(AuditError):
        validate_15(n)
