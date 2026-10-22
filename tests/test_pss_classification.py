"""Synthetic boundary, leakage, scoring, and model integration checks."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from garmin_inventory.privacy import AuditError
from garmin_pss_classification.data import targets, feature_set, cohort_mask
from garmin_pss_classification.metrics import macro_f1, paired_bootstrap
from garmin_pss_classification.models import Transform, weights, fit_model, probabilities, inner_splits


def sample():
    rng = np.random.default_rng(32)
    n = 360
    y = np.tile(np.arange(3), n // 3)
    frame = pd.DataFrame(dict(n_participant=np.repeat(np.arange(60), 6),
        n_survey_category=np.tile(['Morning', 'Midday', 'Endday'], n // 3),
        local_time_sin=rng.uniform(-1, 1, n), local_time_cos=rng.uniform(-1, 1, n),
        target_three_class=y, context_available=True))
    for c in feature_set('context_hr_steps', 60)[1]:
        frame[c] = 10 * y + rng.normal(0, .1, n)
    return frame


def test_targets_and_missingness():
    s = [3, 4, 5, 9, 10, 15, np.nan]
    np.testing.assert_equal(targets(s, 'three_class'), [0, 0, 1, 1, 2, 2, np.nan])
    np.testing.assert_equal(targets(s, 'at_least_5'), [0, 0, 1, 1, 1, 1, np.nan])
    np.testing.assert_equal(targets(s, 'at_least_10'), [0, 0, 0, 0, 1, 1, np.nan])
    with pytest.raises(AuditError):
        targets([3.5], 'three_class')


def test_transforms_train_only_and_all_missing():
    f = sample()
    wear = feature_set('context_hr_steps', 60)[1]
    f.loc[:299, wear[0]] = np.nan
    f.loc[0, wear[1]] = np.nan
    transform = Transform('context_hr_steps', 60, 'imputed').fit(f.iloc[:300])
    assert wear[0] in transform.dropped
    before = transform.medians.copy()
    val = f.iloc[300:].copy()
    val[wear[1]] = 1e9
    val['n_survey_category'] = 'Unseen'
    assert np.isfinite(transform.transform(val)).all()
    pd.testing.assert_series_equal(before, transform.medians)
    assert len(transform.names) == transform.transform(val).shape[1]
    assert not any('target' in c or 'participant' in c or 'pss' in c for c in transform.names)
    with pytest.raises(AuditError):
        Transform('context_hr_steps', 60, 'complete').fit(f)


def test_grouped_folds_weights_and_matched_cohort():
    f = sample()
    splits = inner_splits(f, 'target_three_class', 3, dict(inner_splits=3, seed=42))
    for a, b in splits:
        assert not set(f.iloc[a].n_participant) & set(f.iloc[b].n_participant)
    y = np.array([0, 0, 0, 1, 1, 2])
    w = weights(y, 3, 'balanced')
    np.testing.assert_allclose([w[y == k].sum() for k in range(3)], [2, 2, 2])
    f.loc[0, 'steps_added_supported__60m'] = np.nan
    spec = dict(task='three_class', handling='complete', minutes=60, cohort='joint')
    assert cohort_mask(f, spec).sum() == len(f) - 1
    assert cohort_mask(f, dict(spec, handling='imputed')).sum() == len(f)


def test_metrics_and_paired_bootstrap():
    y, p = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 0, 2])
    assert macro_f1(y, p, 3) == pytest.approx(f1_score(y, p, average='macro'))
    result = paired_bootstrap(y, p, p, np.array(['a', 'a', 'b', 'b', 'c', 'c']), 3, draws=20)
    assert result['difference_low'] == result['difference_high'] == 0


@pytest.mark.parametrize('family,params', [('logistic', {'C': 1}),
    ('xgboost', {'n_estimators': 30, 'max_depth': 2}),
    ('lightgbm', {'n_estimators': 30, 'num_leaves': 4, 'min_child_samples': 10})])
def test_synthetic_signal_held_out_participants(family, params):
    f = sample()
    a, b = np.arange(300), np.arange(300, 360)
    # Remove survey-category shortcut; only injected HR values carry the synthetic signal.
    f['n_survey_category'] = 'Synthetic'
    t = Transform('context_hr', 60, 'complete').fit(f.iloc[a])
    model = fit_model(family, params, t.transform(f.iloc[a]), f.iloc[a].target_three_class.to_numpy(), 3, 'none', 42)
    p = probabilities(model, t.transform(f.iloc[b]), 3)
    assert macro_f1(f.iloc[b].target_three_class, p.argmax(axis=1), 3) > .95


def test_executed_notebook_exception_is_narrow():
    from pathlib import Path
    from garmin_inventory.privacy import validate_notebook_output
    cell = dict(cell_type='code', execution_count=1, outputs=[dict(output_type='stream', text='Aggregate')])
    nb = dict(metadata={'garmin_public_aggregate_notebook': True}, cells=[cell])
    validate_notebook_output(Path('12_pss_classification.ipynb'), nb)
    with pytest.raises(AuditError):
        validate_notebook_output(Path('11_measurement_source_and_timestamp_audit.ipynb'), nb)
    cell['outputs'] = [dict(output_type='error')]
    with pytest.raises(AuditError):
        validate_notebook_output(Path('12_pss_classification.ipynb'), nb)


@pytest.mark.parametrize('family', ['logistic', 'xgboost', 'lightgbm'])
def test_binary_probability_order_and_weights(family):
    f = sample()
    y = (f.target_three_class.to_numpy() >= 2).astype(int)
    t = Transform('context_hr', 60, 'complete').fit(f.iloc[:300])
    params = {'C': 1} if family == 'logistic' else {'n_estimators': 20}
    model = fit_model(family, params, t.transform(f.iloc[:300]), y[:300], 2, 'balanced', 42)
    p = probabilities(model, t.transform(f.iloc[300:]), 2)
    assert p.shape == (60, 2)
    assert macro_f1(y[300:], p.argmax(axis=1), 2) > .95
