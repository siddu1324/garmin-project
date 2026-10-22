"""Separate saved-subset, support-refit and unchanged-cohort median-loss experiments."""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor

from garmin_ema_baseline.models import fit_fold, make_model, metrics, model_columns, paired_bootstrap
from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import digest_object
from .support import support_masks

KEYS = ['stage', 'policy', 'scope', 'lookback_minutes']
META = ['n_ema_id', 'n_participant', 'n_fold', 'n_survey_category', 'pss_combo']
MODELS = ['mean', 'median', 'context', 'hr', 'context_hr']


def fit_quantiles(train, valid, minutes):
    if set(train.n_participant) & set(valid.n_participant):
        raise AuditError('Quantile training/validation participant overlap.')
    predictions, audit = {}, []
    for name in ['context', 'hr', 'context_hr']:
        numeric, category = model_columns(name, minutes)
        columns = numeric + category
        model = make_model(name, minutes)
        model.set_params(ridge=QuantileRegressor(quantile=.5, alpha=.01, solver='highs',
                                                fit_intercept=True, solver_options=None))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(train[columns], train.pss_combo)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise AuditError('Quantile solver convergence failure; do not score this matched fold.')
        pred = model.predict(valid[columns])
        if not np.isfinite(pred).all():
            raise AuditError('Quantile solver produced nonfinite predictions.')
        transform = model.named_steps['transform']
        scaler = transform.named_transformers_['numeric']
        levels = transform.named_transformers_['category'].categories_[0].tolist() if category else []
        if not np.allclose(scaler.mean_, train[numeric].mean(), rtol=1e-12, atol=1e-12):
            raise AuditError('Quantile training-only scaling failed.')
        if category and levels != sorted(train.n_survey_category.unique()):
            raise AuditError('Quantile training-only encoding failed.')
        estimator = model.named_steps['ridge']
        training_error = np.abs(model.predict(train[columns]) - train.pss_combo.to_numpy(float))
        objective = float(.5 * training_error.mean() + .01 * np.abs(estimator.coef_).sum())
        constant_objective = float(.5 * np.abs(train.pss_combo - train.pss_combo.median()).mean())
        if objective > constant_objective + 1e-7:
            raise AuditError('Quantile objective exceeds feasible training-median solution.')
        audit.append({'model': 'q_' + name, 'converged': True, 'solver_iterations': int(estimator.n_iter_),
                      'warning_categories': [w.category.__name__ for w in caught],
                      'input_columns': columns, 'training_mean': scaler.mean_.tolist(),
                      'training_scale': scaler.scale_.tolist(), 'training_categories': levels,
                      'coefficients': estimator.coef_.tolist(), 'intercept': float(estimator.intercept_),
                      'nonzero_coefficients_at_1e_minus_8': int((np.abs(estimator.coef_) > 1e-8).sum()),
                      'training_pinball_plus_penalty': float(.5 * training_error.mean() + .01 * np.abs(estimator.coef_).sum()),
                      'validation_prediction_sd': float(pred.std()),
                      'validation_prediction_minimum': float(pred.min()),
                      'validation_prediction_maximum': float(pred.max()),
                      'validation_unique_predictions_8dp': len(np.unique(pred.round(8))),
                      'validation_equal_training_median_fraction': float(np.isclose(pred, train.pss_combo.median(), rtol=0, atol=1e-8).mean())})
        predictions['q_' + name] = pred
    return predictions, audit


def small_flag(cohort, config):
    counts = [cohort.loc[cohort.n_fold.eq(f), 'n_participant'].nunique() for f in range(5)]
    return bool(len(cohort) < config['small_cohort_responses']
                or cohort.n_participant.nunique() < config['small_cohort_participants']
                or min(counts) < config['small_validation_participants_per_fold'])


def run_comparisons(linked, quality, saved, config):
    masks = support_masks(linked, quality)
    saved = saved.loc[saved.block.eq('hr')].copy()
    predictions, membership, counts, cohorts, audits, composition, diagnostics = [], [], [], [], [], [], []
    for (policy, scope, minutes), mask in masks.items():
        tags = dict(policy=policy, scope=scope, lookback_minutes=minutes)
        reference_mask = masks[('A', scope, minutes)]
        cohort = linked.loc[mask].sort_values('n_ema_id', kind='stable')
        reference = linked.loc[reference_mask].sort_values('n_ema_id', kind='stable')
        saved_group = saved.loc[saved.scope.eq(scope) & saved.lookback_minutes.eq(minutes)]
        if saved_group.n_ema_id.duplicated().any() or set(saved_group.n_ema_id) != set(reference.n_ema_id):
            raise AuditError('Notebook 05 reference membership mismatch.')
        member = linked[META].copy()
        member['reference_included'], member['included'] = reference_mask, mask
        member['reason'] = np.select([~reference_mask, mask], ['outside_reference_complete_cases', 'retained'], default='excluded_by_support')
        for k, v in tags.items():
            member[k] = v
        membership.append(member)
        limited = small_flag(cohort, config)
        cohorts.append({**tags, 'responses': len(cohort), 'participants': cohort.n_participant.nunique(),
                        'reference_responses': len(reference), 'reference_participants': reference.n_participant.nunique(),
                        'excluded_responses': len(reference) - len(cohort), 'small_or_empty': limited})
        for label, selected in [('retained', linked.loc[mask]), ('excluded_from_reference', linked.loc[reference_mask & ~mask]),
                                ('outside_reference', linked.loc[~reference_mask])]:
            for category, part in [('all', selected)] + list(selected.groupby('n_survey_category')):
                y = part.pss_combo.dropna()
                composition.append({**tags, 'group': label, 'survey_category': category, 'responses': len(part),
                                    'participants': part.n_participant.nunique(), 'valid_outcomes': len(y),
                                    'pss_mean': y.mean(), 'pss_sd': y.std(ddof=1), 'pss_median': y.median(),
                                    'pss_q25': y.quantile(.25), 'pss_q75': y.quantile(.75),
                                    'pss_minimum': y.min(), 'pss_maximum': y.max()})
            q = quality.loc[quality.lookback_minutes.eq(minutes) & quality.n_ema_id.isin(selected.n_ema_id)]
            for field in ['accepted_occupancy_fraction', 'valid_readings', 'observed_span_seconds',
                          'latest_age_seconds', 'maximum_internal_gap_seconds', 'maximum_gap_including_edges_seconds']:
                x = q[field].dropna()
                diagnostics.append({**tags, 'group': label, 'measure': field, 'responses': len(q), 'observed': len(x),
                                    'median': x.median(), 'q25': x.quantile(.25), 'q75': x.quantile(.75),
                                    'minimum': x.min(), 'maximum': x.max()})
        for fold in range(5):
            train, valid = cohort.loc[cohort.n_fold.ne(fold)], cohort.loc[cohort.n_fold.eq(fold)]
            old_train = reference.loc[reference.n_fold.ne(fold)]
            base = {**tags, 'fold': fold, 'validation_responses': len(valid),
                    'validation_participants': valid.n_participant.nunique(), 'small_or_empty': limited}
            old_count = {**base, 'stage': 'saved_subset', 'training_responses': len(old_train),
                         'training_participants': old_train.n_participant.nunique(),
                         'status': 'evaluated' if len(valid) else 'no_validation_responses'}
            counts.append(old_count)
            reason = ('no_validation_responses' if not len(valid) else 'insufficient_training'
                      if len(train) < config['minimum_training_responses']
                      or train.n_participant.nunique() < config['minimum_training_participants'] else 'fitted')
            new_count = {**base, 'stage': 'support_refit', 'training_responses': len(train),
                         'training_participants': train.n_participant.nunique(), 'status': reason}
            counts.append(new_count)
            if not len(valid):
                if policy == 'A':
                    counts.append({**new_count, 'stage': 'loss_only'})
                continue
            old = saved_group.set_index('n_ema_id').loc[valid.n_ema_id]
            if not (old.n_fold.to_numpy() == valid.n_fold.to_numpy()).all() or not np.array_equal(old.pss_combo, valid.pss_combo):
                raise AuditError('Saved validation fold/target mismatch.')
            def append_prediction(stage, pred):
                frame = valid[META].copy()
                for k, v in {**tags, 'stage': stage}.items():
                    frame[k] = v
                for model, value in pred.items():
                    frame['prediction_' + model] = np.asarray(value)
                predictions.append(frame)
            saved_predictions = {name: old['prediction_' + name].to_numpy() for name in MODELS}
            append_prediction('saved_subset', saved_predictions)
            if reason == 'fitted':
                pred, audit, _ = fit_fold(train, valid, minutes)
                if policy == 'A' and any(not np.array_equal(pred[name], saved_predictions[name]) for name in MODELS):
                    raise AuditError('Policy A refit differs from Notebook 05 reference.')
                append_prediction('support_refit', pred)
                audits.append({**tags, 'fold': fold, 'stage': 'support_refit', 'models': audit,
                               'training_keys_sha256': digest_object(train.n_ema_id.tolist()),
                               'validation_keys_sha256': digest_object(valid.n_ema_id.tolist())})
            if policy == 'A':
                status = reason
                if reason == 'fitted':
                    try:
                        quantiles, audit = fit_quantiles(train, valid, minutes)
                    except (AuditError, ValueError, TypeError):
                        status = 'quantile_fit_failed_matched_fold_excluded'
                    else:
                        append_prediction('loss_only', {**saved_predictions, **quantiles})
                        audits.append({**tags, 'fold': fold, 'stage': 'loss_only', 'models': audit,
                                       'training_keys_sha256': digest_object(train.n_ema_id.tolist()),
                                       'validation_keys_sha256': digest_object(valid.n_ema_id.tolist())})
                counts.append({**new_count, 'stage': 'loss_only', 'status': status})
    return {'predictions': pd.concat(predictions, ignore_index=True),
            'membership': pd.concat(membership, ignore_index=True), 'fold_counts': pd.DataFrame(counts),
            'cohorts': pd.DataFrame(cohorts), 'training_audit': audits,
            'composition': pd.DataFrame(composition), 'quality_diagnostics': pd.DataFrame(diagnostics)}


def interval(frame, model, baseline, config):
    if frame.n_participant.nunique() < 2:
        return {'interval_status': 'fewer_than_two_participants',
                'mae_difference': metrics(frame, model)['mae'] - metrics(frame, baseline)['mae']}
    return {'interval_status': 'conditional_fixed_predictions',
            **paired_bootstrap(frame, model, baseline, config['bootstrap_draws'], config['bootstrap_seed'])}


def evaluate(predictions, config):
    scores, intervals, variability, retraining = [], [], [], []
    for tags_tuple, frame in predictions.groupby(KEYS, sort=True):
        tags = dict(zip(KEYS, tags_tuple))
        names = [c.removeprefix('prediction_') for c in frame if c.startswith('prediction_') and frame[c].notna().all()]
        for name in names:
            col = 'prediction_' + name
            subsets = [('pooled', 'all', frame)] + [(str(k), 'all', f) for k, f in frame.groupby('n_fold')]
            subsets += [('pooled', category, f) for category, f in frame.groupby('n_survey_category')]
            for fold, category, f in subsets:
                scores.append({**tags, 'fold': fold, 'survey_category': category, 'model': name, **metrics(f, col)})
            variability.append({**tags, 'model': name, 'prediction_sd': float(frame[col].std(ddof=0)),
                                'prediction_minimum': float(frame[col].min()), 'prediction_maximum': float(frame[col].max()),
                                'unique_predictions_8dp': len(np.unique(frame[col].round(8))),
                                'fraction_equal_median_dummy': float(np.isclose(frame[col], frame.prediction_median, rtol=0, atol=1e-8).mean())})
            if name in ['mean', 'median']:
                continue
            baselines = ['mean', 'median', 'context']
            if name.startswith('q_'):
                baselines += [name.removeprefix('q_'), 'q_context']
            for baseline in dict.fromkeys(baselines):
                if baseline == name:
                    continue
                intervals.append({**tags, 'model': name, 'baseline': baseline,
                                  'responses': len(frame), 'participants': frame.n_participant.nunique(),
                                  **interval(frame, col, 'prediction_' + baseline, config)})
    for keys, refit in predictions.loc[predictions.stage.eq('support_refit')].groupby(KEYS[1:]):
        policy, scope, minutes = keys
        old = predictions.loc[predictions.stage.eq('saved_subset') & predictions.policy.eq(policy)
                              & predictions.scope.eq(scope) & predictions.lookback_minutes.eq(minutes)]
        paired = refit.merge(old[['n_ema_id'] + ['prediction_' + name for name in MODELS]],
                             on='n_ema_id', validate='one_to_one', suffixes=('', '_saved'))
        for name in MODELS:
            retraining.append({'policy': policy, 'scope': scope, 'lookback_minutes': minutes, 'model': name,
                               'responses': len(paired), 'participants': paired.n_participant.nunique(),
                               **interval(paired, 'prediction_' + name, 'prediction_' + name + '_saved', config)})
    return {'metrics': pd.DataFrame(scores), 'paired_intervals': pd.DataFrame(intervals),
            'prediction_variability': pd.DataFrame(variability), 'retraining_intervals': pd.DataFrame(retraining)}
