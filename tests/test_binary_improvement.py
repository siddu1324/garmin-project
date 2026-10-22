import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from garmin_binary_improvement.models import Prepared, nested_fit, threshold
from garmin_binary_improvement.sampling import prior_correct, resample
from garmin_binary_improvement.design import specifications


def cfg():
    return json.loads(Path("config/binary_improvement.json").read_text())


def synthetic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=240)
    return pd.DataFrame(
        dict(
            n_participant=np.repeat(np.arange(40), 6),
            n_fold=np.repeat(np.arange(40) % 5, 6),
            n_survey_category=np.tile(["Morning", "Midday"], 120),
            local_time_sin=0.5,
            local_time_cos=np.sqrt(0.75),
            a=x,
            b=x * 2,
            c=rng.normal(size=240),
            empty=np.nan,
            target_A=(x > 0.7).astype(int),
        )
    )


def spec(sampler="none", context=False):
    return dict(
        name="synthetic",
        mode="main",
        sampler=sampler,
        context=context,
        compact=True,
        flags=False,
        permutation=None,
    )


@pytest.mark.parametrize("method", ["none", "weighted", "over", "under", "smote", "conditional"])
def test_sampling_provenance_and_counts(method):
    f = synthetic()
    x = f[["a", "c", "local_time_sin", "local_time_cos"]].to_numpy()
    y = f.target_A.to_numpy()
    xr, yr, audit, lineage = resample(
        x,
        y,
        f.n_participant.to_numpy(),
        f.n_survey_category.to_numpy(),
        f[["a", "c"]].isna().to_numpy(),
        method,
        [0, 1],
        17,
    )
    assert np.isfinite(xr).all()
    assert audit["original_counts"] == np.bincount(y).tolist()
    if method in ["over", "under", "smote", "conditional"]:
        assert np.bincount(yr)[0] == np.bincount(yr)[1]
        assert lineage.base.between(0, len(f) - 1).all()
        assert lineage.neighbor.between(0, len(f) - 1).all()
    if method == "conditional":
        assert audit["cross_person_rows"] == audit["cross_category_rows"] == 0
        np.testing.assert_allclose(xr[:, 2] ** 2 + xr[:, 3] ** 2, 1)
    if method == "smote":
        assert audit["synthetic_rows"] > 0
        a, b = lineage.base.to_numpy(int), lineage.neighbor.to_numpy(int)
        z = x[a].copy()
        z[:, :2] += lineage.fraction.to_numpy()[:, None] * (x[b, :2] - x[a, :2])
        np.testing.assert_allclose(xr[len(f) :], z)


def test_compact_training_selection_and_no_categorical_interpolation():
    f = synthetic()
    t = Prepared(["a", "b", "c", "empty"], spec(), cfg()).fit(f)
    assert t.names == ["a", "c"]
    original = t.base.medians.copy()
    f.a = 100000
    assert np.isfinite(t.transform(f)).all()
    pd.testing.assert_series_equal(t.base.medians, original)


def test_threshold_and_prior_correction():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.15, 0.2, 0.3])
    assert threshold(y, p) == 0.2
    c = prior_correct(p, 0.1, 0.5)
    assert roc_auc_score(y, p) == roc_auc_score(y, c)
    assert np.all(c < p)
    np.testing.assert_allclose(prior_correct(p, 0.1, 0.1), p)


def test_nested_smote_positive_control_no_heldout_donors():
    f = synthetic()
    tr, va = f[f.n_fold != 0], f[f.n_fold == 0]
    config = cfg() | dict(C_grid=[0.1, 1], ratios=[0.0])
    fitted, _, _, splits, _, donors, _, _ = nested_fit(tr, "A", spec("smote"), ["a", "c"], config)
    assert roc_auc_score(va.target_A, fitted.probability(va.drop(columns="target_A"))) > 0.95
    for (a, b), lineage in zip(splits, donors):
        assert not set(tr.iloc[a].n_participant) & set(tr.iloc[b].n_participant)
        assert set(lineage.base) <= set(a) and set(lineage.neighbor) <= set(a)


def test_experiment_registry_fixed_and_outcome_free():
    rows = specifications()
    assert len(rows) == len({r["name"] for r in rows}) == 60
    assert sum(r["reuse"] is not None for r in rows) == 18
    assert all(r["compact"] and not r["context"] for r in rows if r["sampler"] == "smote")
    assert all(r["compact"] and r["context"] for r in rows if r["sampler"] == "conditional")


def test_inner_threshold_changes_decisions_not_ranking_or_brier():
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, precision_score
    from garmin_binary_improvement.evaluate import decision_metrics

    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.3, 0.2, 0.4, 0.7, 0.2])
    thresholds = np.array([0.25, 0.25, 0.25, 0.35, 0.35, 0.35])
    result = decision_metrics(y, p, thresholds)
    assert result["auroc"] == pytest.approx(roc_auc_score(y, p))
    assert result["brier"] == pytest.approx(brier_score_loss(y, p))
    assert result["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, p >= thresholds))
    assert result["precision"] == pytest.approx(precision_score(y, p >= thresholds))


@pytest.mark.parametrize("method", ["weighted", "over", "under", "conditional"])
def test_matched_context_augmentation_preserves_base_rows(method):
    f = synthetic()
    full = f[["a", "c", "local_time_sin", "local_time_cos"]].to_numpy()
    context = full[:, 2:]
    common = (
        f.target_A.to_numpy(),
        f.n_participant.to_numpy(),
        f.n_survey_category.to_numpy(),
    )
    x, y, audit, donors = resample(
        full, *common, f[["a", "c"]].isna().to_numpy(), method, [0, 1], 19
    )
    xc, yc, ac, dc = resample(context, *common, np.zeros((len(f), 0), dtype=bool), method, [], 19)
    np.testing.assert_array_equal(y, yc)
    np.testing.assert_allclose(x[:, 2:], xc)
    pd.testing.assert_series_equal(donors.base, dc.base)
    assert audit["resampled_counts"] == ac["resampled_counts"]
    assert ac["synthetic_rows"] == 0
