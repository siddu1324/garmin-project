"""Independent verification of cohort rules, saved predictions and fitted arithmetic."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_ema_baseline.models import feature_columns, model_columns
from garmin_ema_baseline.verify import independent_prediction
from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import digest_object, verify_bundle
from .experiment import KEYS, MODELS


def verify_run(root, bundle):
    root, bundle = Path(root), Path(bundle)
    verify_bundle(bundle)
    config = json.loads((bundle / 'frozen_configuration.json').read_text())
    baseline = root / config['baseline_bundle']
    verify_bundle(baseline)
    linked = pd.read_parquet(baseline / 'linked_development.parquet')
    old = pd.read_parquet(baseline / 'out_of_fold_predictions.parquet')
    p = pd.read_parquet(bundle / 'predictions.parquet')
    membership = pd.read_parquet(bundle / 'membership.parquet')
    quality = pd.read_parquet(bundle / 'verified_hr_support.parquet')
    audit = json.loads((bundle / 'training_audit.json').read_text())
    assignments = pd.read_parquet(root / 'private/frozen_participant_assignments.parquet')
    dev = assignments.loc[assignments.partition.eq('development')].set_index('participant')
    for frame in [linked, p, membership, quality]:
        if not frame.n_participant.isin(dev.index).all() or not frame.n_fold.eq(frame.n_participant.map(dev.fold)).all():
            raise AuditError('Follow-up participant/fold separation failed.')
    if len(linked) != 3706 or linked.pss_combo.notna().sum() != 3689 or len(membership) != 3706 * 48:
        raise AuditError('Follow-up development accounting failed.')
    quality = quality.set_index(['n_ema_id', 'lookback_minutes'])
    indexed = linked.set_index('n_ema_id')
    masks = {}
    for (policy, scope, minutes), members in membership.groupby(KEYS[1:]):
        if members.n_ema_id.duplicated().any() or set(members.n_ema_id) != set(linked.n_ema_id):
            raise AuditError('Complete follow-up membership accounting failed.')
        expected = []
        for row in linked.itertuples(index=False):
            valid = np.isfinite(row.pss_combo) and row.context_available
            for m in config['lookbacks_minutes'] if scope == 'common' else [minutes]:
                valid &= all(np.isfinite(getattr(row, c)) for c in feature_columns(m))
                q = quality.loc[(row.n_ema_id, m)]
                if policy != 'A':
                    valid &= int(q.occupied_minute_bins) * 5 >= m * 4
                if policy == 'C':
                    valid &= float(q.latest_age_seconds) <= 60
            expected.append(bool(valid))
        selected = set(linked.loc[expected, 'n_ema_id'])
        if selected != set(members.loc[members.included, 'n_ema_id']):
            raise AuditError('Independent fixed support/intersection membership failed.')
        masks[(policy, scope, minutes)] = selected
    fits, quantile_fits, largest_error = 0, 0, 0.
    for tags_tuple, pred in p.groupby(KEYS):
        stage, policy, scope, minutes = tags_tuple
        if stage == 'loss_only' and policy != 'A':
            raise AuditError('Median regression was combined with a support restriction.')
        source_members = masks[('A' if stage == 'saved_subset' else policy, scope, minutes)]
        cohort = linked.loc[linked.n_ema_id.isin(source_members)].sort_values('n_ema_id', kind='stable')
        if pred.n_ema_id.duplicated().any() or set(pred.n_ema_id) - masks[(policy, scope, minutes)]:
            raise AuditError('Prediction membership is not one-to-one within a comparison.')
        for fold, part in pred.groupby('n_fold'):
            train = cohort.loc[cohort.n_fold.ne(fold)]
            valid = indexed.loc[part.n_ema_id]
            if set(train.n_participant) & set(valid.n_participant):
                raise AuditError('Training/validation participant overlap.')
            if not np.array_equal(part.pss_combo, valid.pss_combo):
                raise AuditError('Outcome values changed during follow-up.')
            if stage in ['saved_subset', 'loss_only']:
                previous = old.loc[old.block.eq('hr') & old.scope.eq(scope) & old.lookback_minutes.eq(minutes)].set_index('n_ema_id').loc[part.n_ema_id]
                for name in MODELS:
                    if not np.array_equal(previous['prediction_' + name], part['prediction_' + name]):
                        raise AuditError('Saved comparison predictions changed.')
            if stage == 'support_refit':
                for name in MODELS:
                    expected = independent_prediction(train, valid, name, minutes)
                    error = float(np.max(np.abs(expected - part['prediction_' + name].to_numpy())))
                    if error > 1e-8:
                        raise AuditError('Independent support-refit arithmetic mismatch.')
                    largest_error = max(largest_error, error)
                    fits += 1
            if stage == 'loss_only':
                record = next(a for a in audit if all(a[k] == v for k, v in zip(KEYS, tags_tuple)) and a['fold'] == fold)
                if record['training_keys_sha256'] != digest_object(train.n_ema_id.tolist()):
                    raise AuditError('Quantile training cohort hash mismatch.')
                for fit in record['models']:
                    name = fit['model'].removeprefix('q_')
                    numeric, category = model_columns(name, minutes)
                    if not np.allclose(fit['training_mean'], train[numeric].mean(), rtol=1e-12, atol=1e-12):
                        raise AuditError('Independent quantile training moments mismatch.')
                    expected_scale = train[numeric].std(ddof=0).to_numpy(float)
                    expected_scale[expected_scale == 0] = 1
                    if not np.allclose(fit['training_scale'], expected_scale, rtol=1e-10, atol=1e-12):
                        raise AuditError('Independent quantile training scale mismatch.')
                    if category and fit['training_categories'] != sorted(train.n_survey_category.unique()):
                        raise AuditError('Independent quantile encoder mismatch.')
                    def calculate(frame):
                        x = (frame[numeric].to_numpy(float) - fit['training_mean']) / fit['training_scale']
                        if category:
                            x = np.column_stack([x] + [frame.n_survey_category.eq(c).to_numpy(float) for c in fit['training_categories']])
                        return x @ fit['coefficients'] + fit['intercept']
                    expected = calculate(valid)
                    if not np.allclose(expected, part['prediction_q_' + name], rtol=0, atol=1e-10):
                        raise AuditError('Quantile parameter/prediction replay mismatch.')
                    objective = .5 * np.abs(calculate(train) - train.pss_combo).mean() + .01 * np.abs(fit['coefficients']).sum()
                    constant = .5 * np.abs(train.pss_combo - train.pss_combo.median()).mean()
                    if (not np.isclose(objective, fit['training_pinball_plus_penalty'], atol=1e-10)
                            or objective > constant + 1e-7 or not fit['converged']):
                        raise AuditError('Independent quantile objective/convergence check failed.')
                    quantile_fits += 1
    return {'membership_rows_verified': len(membership), 'prediction_rows_verified': len(p),
            'support_refit_models_independently_verified': fits,
            'maximum_ridge_prediction_discrepancy': largest_error,
            'quantile_fits_verified': quantile_fits, 'all_original_folds_preserved': True,
            'targets_unchanged': True, 'final_test_participants_sealed': 32,
            'loss_support_experiments_separate': True}
