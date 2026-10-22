"""Synthetic tests; never inspect participant response rows."""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor

from garmin_ema_baseline.linkage import link_outcomes
from garmin_ema_baseline.models import LOOKBACKS, run_models
from garmin_hr_support_loss.experiment import fit_quantiles, interval, run_comparisons, small_flag
from garmin_hr_support_loss.support import accepted_occupancy, support_masks
from garmin_inventory.privacy import AuditError
from test_ema_baseline import fixture


def data():
    e, f, a = fixture()
    linked = link_outcomes(e, f, a, expected_rows=20)
    quality = pd.DataFrame([{'n_ema_id': key, 'lookback_minutes': m, 'occupied_minute_bins': m,
                            'latest_age_seconds': 1., 'accepted_occupancy_fraction': 1., 'valid_readings': 10,
                            'observed_span_seconds': 100., 'maximum_internal_gap_seconds': 5.,
                            'maximum_gap_including_edges_seconds': 10.}
                           for m in LOOKBACKS for key in linked.n_ema_id])
    return linked, quality


def test_occupancy_boundaries_duplicates_and_unaccepted_points():
    # Caller passes accepted points only; a rejected reading in bin 2 contributes nothing.
    times = np.array([99., 100., 100., 159.999, 160., 399.999, 400.])
    assert accepted_occupancy(times, 100., 5) == 3
    assert accepted_occupancy(np.array([]), 100., 5) == 0
    assert accepted_occupancy(np.array([100., 400.]), 100., 5) == 1
    assert accepted_occupancy(np.array([100., 100.+1439*60]), 100., 1440) == 2


def test_fixed_policy_thresholds_inclusive_and_common_intersection():
    linked, q = data()
    selected = q.lookback_minutes.eq(5)
    q.loc[selected, 'occupied_minute_bins'] = 4
    q.loc[selected, 'latest_age_seconds'] = 60
    q.loc[selected & q.n_ema_id.eq(linked.n_ema_id.iloc[0]), 'latest_age_seconds'] = 60.000001
    q.loc[selected & q.n_ema_id.eq(linked.n_ema_id.iloc[1]), 'occupied_minute_bins'] = 3
    m = support_masks(linked, q)
    assert m[('A','native',5)].sum() == 20
    assert m[('B','native',5)].sum() == 19
    assert m[('C','native',5)].sum() == 18
    assert m[('C','common',1440)].sum() == 18
    assert m[('C','native',1440)].sum() == 20


def test_quantile_train_only_and_constant_predictions():
    linked, _ = data()
    train, valid = linked.iloc[:16].copy(), linked.iloc[16:].copy()
    train['pss_combo'] = 7.
    pred, audit = fit_quantiles(train, valid, 5)
    for values in pred.values():
        assert np.allclose(values, 7)
    assert all(a['converged'] and a['validation_unique_predictions_8dp'] == 1 for a in audit)
    valid['pss_combo'] = 1000.
    valid['n_survey_category'] = 'Endday'
    pred2, audit2 = fit_quantiles(train, valid, 5)
    assert all(np.allclose(v, 7) for v in pred2.values())
    assert all('Endday' not in a['training_categories'] for a in audit2)
    with pytest.raises(AuditError):
        fit_quantiles(train, train.iloc[:2], 5)


def test_convergence_warning_is_failure(monkeypatch):
    linked, _ = data()
    original = QuantileRegressor.fit
    def warned(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        warnings.warn('synthetic convergence warning', ConvergenceWarning)
        return result
    monkeypatch.setattr(QuantileRegressor, 'fit', warned)
    with pytest.raises(AuditError, match='convergence'):
        fit_quantiles(linked.iloc[:16], linked.iloc[16:], 5)


def test_small_empty_and_uninformative_bootstrap():
    linked, _ = data()
    cfg = json.loads(Path('config/hr_support_loss.json').read_text())
    assert small_flag(linked.iloc[:0], cfg)
    assert small_flag(linked, cfg)
    frame = linked.iloc[:1].copy()
    frame['pred'] = 7.
    frame['base'] = 6.
    assert interval(frame, 'pred', 'base', cfg)['interval_status'] == 'fewer_than_two_participants'


def test_saved_subset_refit_and_loss_do_not_mix(monkeypatch):
    linked, q = data()
    oldconfig = json.loads(Path('config/ema_baseline.json').read_text())
    saved = run_models(linked, oldconfig)[0]
    q['occupied_minute_bins'] = 0  # B/C empty: preserve all fold reasons, no relaxation.
    cfg = json.loads(Path('config/hr_support_loss.json').read_text())
    # This test exercises bookkeeping; fitting itself is tested above.
    def quick(train, valid, minutes):
        return {f'q_{name}': np.full(len(valid), train.pss_combo.median()) for name in ['context','hr','context_hr']}, []
    monkeypatch.setattr('garmin_hr_support_loss.experiment.fit_quantiles', quick)
    out = run_comparisons(linked, q, saved, cfg)
    p = out['predictions']
    assert set(p.policy) == {'A'}
    assert set(p.stage) == {'saved_subset','support_refit','loss_only'}
    assert out['fold_counts'].loc[out['fold_counts'].policy.ne('A'),'status'].eq('no_validation_responses').all()
    assert len(out['membership']) == 20 * 48
    assert not p.loc[p.stage.ne('loss_only'),'prediction_q_hr'].notna().any()
    assert out['membership'].groupby(['policy','scope','lookback_minutes']).size().eq(20).all()


def test_infeasible_refits_preserve_saved_predictions_and_original_folds(monkeypatch):
    linked, q = data()
    saved = run_models(linked, json.loads(Path('config/ema_baseline.json').read_text()))[0]
    one_person = set(linked.loc[linked.n_participant.eq(linked.n_participant.iloc[0]), 'n_ema_id'])
    q.loc[~q.n_ema_id.isin(one_person), 'occupied_minute_bins'] = 0
    def quick(train, valid, minutes):
        return {f'q_{name}': np.full(len(valid), train.pss_combo.median()) for name in ['context','hr','context_hr']}, []
    monkeypatch.setattr('garmin_hr_support_loss.experiment.fit_quantiles', quick)
    out = run_comparisons(linked, q, saved, json.loads(Path('config/hr_support_loss.json').read_text()))
    restricted = out['predictions'].loc[out['predictions'].policy.ne('A')]
    assert set(restricted.stage) == {'saved_subset'}
    assert set(restricted.n_ema_id) == one_person
    assert restricted.n_fold.eq(0).all()
    fits = out['fold_counts']
    assert fits.loc[fits.policy.ne('A') & fits.stage.eq('support_refit') & fits.fold.eq(0), 'status'].eq('insufficient_training').all()


def test_quantile_failure_excludes_entire_loss_comparison_fold(monkeypatch):
    linked, q = data()
    saved = run_models(linked, json.loads(Path('config/ema_baseline.json').read_text()))[0]
    def fail(*args, **kwargs):
        raise AuditError('Synthetic solver failure.')
    monkeypatch.setattr('garmin_hr_support_loss.experiment.fit_quantiles', fail)
    out = run_comparisons(linked, q, saved, json.loads(Path('config/hr_support_loss.json').read_text()))
    assert not out['predictions'].stage.eq('loss_only').any()
    counts = out['fold_counts']
    assert counts.loc[counts.stage.eq('loss_only'),'status'].eq('quantile_fit_failed_matched_fold_excluded').all()
    assert out['predictions'].stage.eq('support_refit').any()
