import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from garmin_root_cause.data import (
    aligned_features,
    recall_gate,
    independent_hr,
    interval_support,
    probe_window,
    HR,
)
from garmin_root_cause.models import OrdinalLogit, ordinal_objective, fit_model, specs
from garmin_root_cause.describe import random_intercept_reml


def test_independent_hr_and_boundaries():
    value = independent_hr([0, 60, 120], [60, 70, 80])
    assert list(value.values()) == [70, 10, 80, 10]
    assert np.isnan(independent_hr([0, 30], [60, 80])[HR[-1]])
    with pytest.raises(AssertionError):
        independent_hr([0, 0], [60, 80])


def test_interval_union_and_crossing():
    assert interval_support([(0, 50), (40, 100), (150, 240)], 0, 300) == (190, 60)
    assert interval_support([], 0, 300) == (0, 300)
    with pytest.raises(ValueError):
        interval_support([(-1, 60)], 0, 300)
    with pytest.raises(ValueError):
        interval_support([(240, 300)], 0, 300)


def test_recall_gate_and_no_post_data():
    row = dict(
        start_epoch=0.0,
        available_epoch=5.0,
        cutoff_epoch=180.0,
        boundary_source="verified_delivered_prompt",
    )
    contract = dict(
        sensor_acquisition_verified=True,
        utc_conversion_verified=True,
        boundary_semantics_verified=True,
        complete_prompt_log=True,
    )
    assert not recall_gate(row, contract)
    for change in [
        dict(available_epoch=np.nan),
        dict(available_epoch=180.0),
        dict(boundary_source="current_survey_answer"),
        dict(start_epoch=190.0),
    ]:
        assert recall_gate({**row, **change}, contract)
    assert recall_gate(row, {**contract, "complete_prompt_log": False})
    h = pd.DataFrame(
        dict(
            n_epoch_s=[-1, 0, 60, 120, 180, 999],
            n_value=[240, 60, 70, 80, 240, 240],
            a_first=True,
            a_acceptable=True,
        )
    )
    assert aligned_features(h, row, contract)[HR[0]] == 70


def test_ordinal_gradient_and_known_likelihood():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 2))
    y = rng.integers(0, 5, 30)
    w = np.array([0.2, -0.3, -1.0, 0.0, -0.3, 0.2])
    assert (
        check_grad(
            lambda a: ordinal_objective(a, x, y)[0], lambda a: ordinal_objective(a, x, y)[1], w
        )
        < 1e-4
    )
    z = np.zeros((5, 1))
    theta = np.log(np.array([0.2, 0.4, 0.6, 0.8]) / np.array([0.8, 0.6, 0.4, 0.2]))
    w = np.r_[0.0, theta[0], np.log(np.diff(theta))]
    assert ordinal_objective(w, z, np.arange(5))[0] == pytest.approx(-5 * np.log(0.2))


def test_ordinal_synthetic_recovery():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(1000, 1))
    latent = 2 * x[:, 0] + rng.logistic(size=1000)
    y = np.searchsorted([-1.5, -0.5, 0.5, 1.5], latent) + 1
    m = OrdinalLogit().fit(x[:800], y[:800])
    p = m.predict_proba(x[800:])
    assert np.allclose(p.sum(axis=1), 1) and (p >= 0).all()
    assert np.all(np.diff(m.theta) > 0) and 1.5 < m.beta[0] < 2.5
    assert np.corrcoef(p @ np.arange(1, 6), y[800:])[0, 1] > 0.6


def test_actual_grouped_pipeline_positive_control_and_training_transform():
    rng = np.random.default_rng(81)
    rows = []
    for person in range(30):
        for j in range(20):
            v = rng.normal()
            h = independent_hr([0, 60, 120], [80 + v * 10, 81 + v * 10, 82 + v * 10])
            rows.append(
                dict(
                    n_ema_id=f"synthetic{person}_{j}",
                    n_participant=f"synthetic{person}",
                    n_fold=person % 5,
                    n_survey_category="Synthetic",
                    local_time_sin=0.0,
                    local_time_cos=1.0,
                    target=int(rng.random() < expit(5 * v)),
                    **h,
                )
            )
    f = pd.DataFrame(rows)
    train, valid = f[f.n_fold != 0], f[f.n_fold == 0]
    spec = next(s for s in specs() if s["name"] == "hr")
    fitted, _, membership = fit_model(
        train, "target", spec, "binary", {"seed": 12, "C_grid": [0.1, 1.0]}
    )
    assert roc_auc_score(valid.target, fitted.predict(valid)) > 0.9
    assert not set(train.n_participant) & set(valid.n_participant)
    assert len(membership) == 6
    assert set(fitted.transform.names) <= set(HR)
    base = fitted.transform.base
    pd.testing.assert_series_equal(base.medians, train[base.kept].median())


def test_reml_matches_independent_dense_covariance():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(15), 6)
    x = np.column_stack([np.ones(len(groups)), rng.normal(size=len(groups))])
    y = x @ [2.0, 0.5] + rng.normal(size=15)[groups] + rng.normal(size=len(groups)) * 0.4
    fit = random_intercept_reml(x, y, groups)
    variance = fit["residual_variance"]
    covariance = variance * np.eye(len(y)) + fit["participant_variance"] * (
        groups[:, None] == groups[None, :]
    )
    inv = np.linalg.inv(covariance)
    dense_beta = np.linalg.solve(x.T @ inv @ x, x.T @ inv @ y)
    np.testing.assert_allclose(dense_beta, fit["beta"], atol=1e-10)
    np.testing.assert_allclose(np.linalg.inv(x.T @ inv @ x), fit["covariance"], atol=1e-10)
    assert fit["converged"] and fit["icc"] > 0.5


def test_activity_probe_no_missing_to_zero_or_boundary_assignment():
    import json
    from pathlib import Path

    cfg = json.loads(Path("config/root_cause_analysis.json").read_text())
    hr = pd.DataFrame(
        dict(
            n_epoch_s=[0, 60, 120, 180, 240, 300],
            n_value=[70, 75, 80, 75, 70, 240],
            a_first=True,
            a_acceptable=True,
        )
    )
    pairs = pd.DataFrame(
        dict(
            a_pair_start=[0, 60, 120, 180, 240],
            n_epoch_s=[60, 120, 180, 240, 300],
            a_pair_delta=[20, 20, 20, 20, 999],
            a_pair_accepted=True,
        )
    )
    result = probe_window(hr, pairs, 0, 300, cfg)
    assert result["eligible"] and result["target"] == 1 and result["supported_seconds"] == 240
    assert result["step_rate"] == 20 and result[HR[2]] == 70
    zero = probe_window(hr, pairs.assign(a_pair_delta=0), 0, 300, cfg)
    assert zero["eligible"] and zero["target"] == 0
    missing = probe_window(hr, pairs.iloc[:0], 0, 300, cfg)
    assert not missing["eligible"] and np.isnan(missing["target"])
    rejected = probe_window(hr, pairs.assign(a_pair_accepted=False), 0, 300, cfg)
    assert not rejected["eligible"] and rejected["supported_seconds"] == 0


def test_completed_seal_rejects_changed_source(tmp_path):
    import json
    from garmin_root_cause.preservation import completed

    path = tmp_path / "reports/root_cause_analysis"
    path.mkdir(parents=True)
    source = tmp_path / "source.py"
    source.write_text("changed")
    (path / "completion.json").write_text(
        json.dumps(
            dict(complete=True, source_sha256={"source.py": "incorrect"}, presentation_sha256={})
        )
    )
    with pytest.raises(ValueError, match="create an explicit new version"):
        completed(tmp_path)


def test_alignment_entrypoint_identity_evidence_and_frozen_cutoff():
    from garmin_root_cause.alignment import align_verified_metadata

    ema = dict(
        n_ema_id="synthetic",
        n_participant="synthetic_person",
        n_partition="development",
        n_fold=0,
        n_cutoff_epoch=180.0,
    )
    contract = dict(
        sensor_acquisition_verified=True,
        utc_conversion_verified=True,
        boundary_semantics_verified=True,
        complete_prompt_log=True,
        contract_sha256="a" * 64,
    )
    metadata = dict(
        n_ema_id="synthetic",
        n_participant="synthetic_person",
        boundary_source="verified_delivered_prompt",
        start_epoch=0.0,
        available_epoch=1.0,
        boundary_payload_sha256="b" * 64,
    )
    ledger = pd.DataFrame(
        dict(
            n_participant="synthetic_person",
            n_partition="development",
            n_epoch_s=[0, 60, 120, 180],
            n_value=[60, 70, 80, 240],
            a_first=True,
            a_acceptable=True,
        )
    )
    evidence = {"a" * 64, "b" * 64}
    assert align_verified_metadata(ema, ledger, metadata, contract, evidence)[HR[0]] == 70
    for change in [
        dict(n_participant="different"),
        dict(cutoff_epoch=181.0),
        dict(boundary_source="current_survey_answer"),
    ]:
        with pytest.raises(ValueError):
            align_verified_metadata(ema, ledger, {**metadata, **change}, contract, evidence)
    with pytest.raises(ValueError):
        align_verified_metadata(
            {**ema, "n_partition": "final_test"}, ledger, metadata, contract, evidence
        )
    with pytest.raises(ValueError):
        align_verified_metadata(ema, ledger, metadata, contract, set())
