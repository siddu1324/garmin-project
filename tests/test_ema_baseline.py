"""Synthetic-only tests for target linkage, leakage guards and paired uncertainty."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from garmin_ema_baseline.linkage import (META, check_development, link_outcomes,
                                        local_context, score_items)
from garmin_ema_baseline.models import (LOOKBACKS, cohort_masks, feature_columns, fit_fold,
                                       make_model, metrics, model_columns, paired_bootstrap,
                                       run_models)
from garmin_inventory.audit import PSS
from garmin_inventory.privacy import AuditError


def fixture():
    n = 20
    e = pd.DataFrame({'n_ema_id': ['synthetic_event_' + str(i) for i in range(n)],
                      'n_participant': ['synthetic_person_' + str(i // 2) for i in range(n)],
                      'n_partition': 'development', 'n_fold': [i // 4 for i in range(n)],
                      'n_survey_category': ['Morning', 'Midday'] * 10,
                      'n_cutoff_epoch': np.arange(n) + 100., 'n_start_precision': 'second',
                      'n_invalid_clock': False, 'n_actual_local': pd.Timestamp('2020-01-01 12:00:00'),
                      'n_source_sha256': 'synthetic', 'n_source_row': range(n),
                      'n_source_columns': json.dumps(list(PSS)),
                      'PSS_NERVOUS': ['2', '4'] * 10, 'PSS_COPE': '4', 'PSS_CONTROL': '3'})
    features = e[META].copy()
    for m in LOOKBACKS:
        for j, c in enumerate(feature_columns(m, True)):
            features[c] = np.arange(n) / 10 + j
    a = e[['n_participant', 'n_partition', 'n_fold']].drop_duplicates().rename(
        columns={'n_participant': 'participant', 'n_partition': 'partition', 'n_fold': 'fold'})
    return e, features, a


def test_scoring_reversal_range_missing_and_invalid():
    raw = pd.DataFrame({'PSS_NERVOUS': ['1', '5', '4', '', '-99', 'NA', '2.5', '6', 'inf', None, 'x'],
                        'PSS_COPE': ['5', '1', '4'] + ['3'] * 8,
                        'PSS_CONTROL': ['5', '1', '2'] + ['3'] * 8})
    out = score_items(raw)
    assert out.pss_combo.iloc[:3].tolist() == [3, 15, 10]
    assert out.pss_combo.iloc[3:].isna().all()
    assert out.PSS_NERVOUS_status.tolist()[3:] == ['missing', 'missing', 'missing', 'invalid', 'invalid', 'invalid', 'missing', 'invalid']
    assert out.PSS_NERVOUS_raw.equals(raw.PSS_NERVOUS)
    assert 'PSS_NERVOUS:missing' == out.outcome_reason.iloc[3]


def test_link_preserves_all_rows_missing_outcomes_and_reordered_features():
    e, f, a = fixture()
    e.loc[1, 'PSS_COPE'] = ''
    linked = link_outcomes(e, f.iloc[::-1], a, expected_rows=20)
    assert len(linked) == 20 and linked.n_ema_id.equals(e.n_ema_id)
    assert linked.n_fold.equals(f.n_fold) and pd.isna(linked.pss_combo.iloc[1])
    assert linked.pss_combo.notna().sum() == 19


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'participant', 'fold', 'cutoff', 'category'])
def test_join_rejects_identity_and_cardinality_failures(kind):
    e, f, a = fixture()
    if kind == 'duplicate':
        f.loc[1, 'n_ema_id'] = f.loc[0, 'n_ema_id']
    elif kind == 'missing':
        f.loc[1, 'n_ema_id'] = 'synthetic_unknown'
    else:
        col = {'participant': 'n_participant', 'fold': 'n_fold', 'cutoff': 'n_cutoff_epoch', 'category': 'n_survey_category'}[kind]
        f.loc[1, col] = {'participant': 'synthetic_stranger', 'fold': 4, 'cutoff': 2, 'category': 'Endday'}[kind]
    with pytest.raises(AuditError):
        link_outcomes(e, f, a, expected_rows=20)


def test_final_test_and_changed_fold_rejected():
    e, _, a = fixture()
    e.loc[0, 'n_partition'] = 'final_test'
    with pytest.raises(AuditError):
        check_development(e, a)
    e.loc[0, 'n_partition'] = 'development'
    e.loc[0, 'n_fold'] = 1
    with pytest.raises(AuditError):
        check_development(e, a)


def test_cyclic_actual_local_time_seconds_and_midnight():
    c = local_context(pd.Series(['2020-01-01 00:00:00', '2020-01-01 06:00:00',
                                  '2020-01-01 23:59:59', '2020-01-02 00:00:01', 'bad']))
    assert np.allclose(c.iloc[0], [0, 1]) and np.allclose(c.iloc[1], [1, 0])
    assert c.local_time_sin.iloc[2] < 0 < c.local_time_sin.iloc[3]
    assert np.isclose(c.local_time_cos.iloc[2], c.local_time_cos.iloc[3])
    assert c.iloc[4].isna().all()


def test_explicit_predictors_exclude_targets_quality_and_other_features():
    allowed = set(feature_columns(5, True) + ['local_time_sin', 'local_time_cos', 'n_survey_category'])
    for model in ['context', 'hr', 'context_hr', 'context_hr_steps']:
        numeric, category = model_columns(model, 5)
        assert set(numeric + category) <= allowed
    with pytest.raises(AuditError):
        model_columns('hrv', 5)
    with pytest.raises(AuditError):
        feature_columns(10)


def test_training_only_scaler_encoder_and_validation_target_independence():
    e, f, a = fixture()
    linked = link_outcomes(e, f, a, expected_rows=20)
    train, valid = linked.iloc[:16], linked.iloc[16:].copy()
    valid.loc[:, feature_columns(5)[0]] = 100000.
    valid.loc[:, 'n_survey_category'] = 'Endday'
    p1, audits, _ = fit_fold(train, valid, 5)
    valid.loc[:, 'pss_combo'] = 999.
    p2, _, _ = fit_fold(train, valid, 5)
    for name in p1:
        assert np.array_equal(p1[name], p2[name])
    hr = next(x for x in audits if x['model'] == 'hr')
    assert np.allclose(hr['training_mean'], train[feature_columns(5)].mean())
    assert np.allclose(hr['training_variance'], train[feature_columns(5)].var(ddof=0))
    context = next(x for x in audits if x['model'] == 'context')
    assert context['categories'] == ['Midday', 'Morning']
    assert np.allclose(p1['mean'], train.pss_combo.mean())
    assert np.allclose(p1['median'], train.pss_combo.median())
    with pytest.raises(AuditError):
        fit_fold(train, train.iloc[:1], 5)


def test_extra_columns_cannot_change_ridge_predictions():
    e, f, a = fixture()
    linked = link_outcomes(e, f, a, expected_rows=20)
    model = make_model('hr', 5).fit(linked, linked.pss_combo)
    expected = model.predict(linked)
    linked['pss_combo'] = 99999
    linked['quality_flag'] = -1000
    linked['floorsClimbed_added_supported__5m'] = 999
    assert np.array_equal(expected, model.predict(linked))


def test_common_intersection_and_matched_fold_predictions():
    e, f, a = fixture()
    f.loc[0, feature_columns(5)[0]] = np.nan
    f.loc[1, 'steps_added_supported__15m'] = np.nan
    linked = link_outcomes(e, f, a, expected_rows=20)
    masks = cohort_masks(linked)
    assert masks[('native', 'hr', 15)].sum() == 20
    assert masks[('common', 'hr', 15)].sum() == 19
    assert masks[('common', 'hr_steps', 15)].sum() == 18
    config = json.loads(Path('config/ema_baseline.json').read_text())
    p, counts, _, _, cohorts, _ = run_models(linked, config)
    assert counts.status.eq('fitted').all()
    assert cohorts.loc[cohorts.scope.eq('common'), 'comparative_interpretation_limited'].all()
    for _, g in p.groupby(['scope', 'block', 'lookback_minutes']):
        assert not g.n_ema_id.duplicated().any()
        assert g[[c for c in p if c.startswith('prediction_') and c != 'prediction_context_hr_steps']].notna().all().all()
        if g.block.iloc[0] == 'hr_steps':
            assert g.prediction_context_hr_steps.notna().all()


def test_paired_cluster_bootstrap_determinism_zero_and_weighting():
    f = pd.DataFrame({'n_participant': ['synthetic_a'] * 3 + ['synthetic_b'],
                      'pss_combo': [0.] * 4, 'model': [1., 1., 1., 3.], 'baseline': [0.] * 4})
    result = paired_bootstrap(f, 'model', 'baseline', draws=200)
    assert result == paired_bootstrap(f, 'model', 'baseline', draws=200)
    assert result['mae_difference'] == 1.5 and result['participant_mae_difference'] == 2
    assert set(paired_bootstrap(f, 'model', 'model', draws=200).values()) == {0.}
    assert metrics(f, 'model')['rmse'] == np.sqrt(3)
    assert metrics(f, 'model')['participant_mae'] == 2
