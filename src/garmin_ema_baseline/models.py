"""Fixed participant-fold models; explicit predictor allowlists, no imputation."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import digest_object

HR = ['hr_mean_bpm', 'hr_sd_bpm', 'hr_latest_bpm', 'hr_slope_bpm_per_minute']
LOOKBACKS = [5, 15, 30, 60, 120, 360, 720, 1440]
CONTEXT = ['local_time_sin', 'local_time_cos']
KEYS = ['scope', 'block', 'lookback_minutes']


def feature_columns(minutes, steps=False):
    if minutes not in LOOKBACKS:
        raise AuditError('Lookback outside the frozen specification.')
    return [f'{name}__{minutes}m' for name in HR + (['steps_added_supported'] if steps else [])]


def cohort_masks(linked):
    base = linked.pss_combo.notna() & linked.context_available
    masks = {}
    for block in ['hr', 'hr_steps']:
        native = {m: base & np.isfinite(linked[feature_columns(m, block == 'hr_steps')]).all(axis=1)
                  for m in LOOKBACKS}
        common = pd.concat(native, axis=1).all(axis=1)
        for m in LOOKBACKS:
            masks[('native', block, m)] = native[m]
            masks[('common', block, m)] = common
    return masks


def model_columns(name, minutes):
    numeric, categorical = [], []
    if name in ['context', 'context_hr', 'context_hr_steps']:
        numeric += CONTEXT
        categorical += ['n_survey_category']
    if name in ['hr', 'context_hr', 'context_hr_steps']:
        numeric += feature_columns(minutes, name == 'context_hr_steps')
    if name not in ['context', 'hr', 'context_hr', 'context_hr_steps']:
        raise AuditError('Unapproved Ridge model requested.')
    return numeric, categorical


def make_model(name, minutes):
    numeric, categorical = model_columns(name, minutes)
    transformers = [('numeric', StandardScaler(), numeric)]
    if categorical:
        transformers.append(('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical))
    return Pipeline([('transform', ColumnTransformer(transformers, remainder='drop')),
                     ('ridge', Ridge(alpha=1, solver='svd', fit_intercept=True))])


def fit_fold(train, valid, minutes, steps=False):
    if set(train.n_participant) & set(valid.n_participant):
        raise AuditError('Participant leakage across training and validation.')
    names = ['mean', 'median', 'context', 'hr', 'context_hr'] + (['context_hr_steps'] if steps else [])
    predictions, audits, coefficients = {}, [], []
    for name in names:
        if name in ['mean', 'median']:
            model = DummyRegressor(strategy=name).fit(np.zeros((len(train), 1)), train.pss_combo)
            predictions[name] = model.predict(np.zeros((len(valid), 1)))
            audits.append({'model': name, 'constant': float(model.constant_.item())})
            continue
        numeric, categorical = model_columns(name, minutes)
        cols = numeric + categorical
        model = make_model(name, minutes).fit(train[cols], train.pss_combo)
        predictions[name] = model.predict(valid[cols])
        transform = model.named_steps['transform']
        scaler = transform.named_transformers_['numeric']
        if not np.allclose(scaler.mean_, train[numeric].mean().to_numpy(), rtol=1e-12, atol=1e-12):
            raise AuditError('Training-only scaler verification failed.')
        categories = transform.named_transformers_['category'].categories_[0].tolist() if categorical else []
        if categorical and categories != sorted(train.n_survey_category.unique()):
            raise AuditError('Training-only encoder verification failed.')
        audits.append({'model': name, 'input_columns': cols, 'numeric_columns': numeric,
                       'training_mean': scaler.mean_.tolist(), 'training_scale': scaler.scale_.tolist(),
                       'training_variance': scaler.var_.tolist(), 'categories': categories,
                       'training_count': int(scaler.n_samples_seen_)})
        for label, coefficient in zip(transform.get_feature_names_out(), model.named_steps['ridge'].coef_):
            coefficients.append({'model': name, 'term': label, 'coefficient': float(coefficient)})
        coefficients.append({'model': name, 'term': 'intercept', 'coefficient': float(model.named_steps['ridge'].intercept_)})
    return predictions, audits, coefficients


def run_models(linked, config):
    predictions, counts, audits, coefficients, cohorts = [], [], [], [], []
    membership = []
    for (scope, block, minutes), mask in cohort_masks(linked).items():
        tags = dict(scope=scope, block=block, lookback_minutes=minutes)
        cohort = linked.loc[mask].sort_values('n_ema_id', kind='stable')
        fold_people = [cohort.loc[cohort.n_fold.eq(k), 'n_participant'].nunique() for k in range(5)]
        limited = (len(cohort) < config['common_descriptive_minimum_responses']
                   or cohort.n_participant.nunique() < config['common_descriptive_minimum_participants']
                   or min(fold_people) < config['common_descriptive_minimum_participants_per_fold'])
        cohorts.append({**tags, 'responses': len(cohort), 'participants': cohort.n_participant.nunique(),
                        'comparative_interpretation_limited': bool(limited) if scope == 'common' else None})
        member = linked[['n_ema_id', 'n_participant', 'n_fold']].copy()
        member['included'] = mask
        for key, value in tags.items():
            member[key] = value
        membership.append(member)
        for fold in range(5):
            train, valid = cohort.loc[cohort.n_fold.ne(fold)], cohort.loc[cohort.n_fold.eq(fold)]
            reason = ('no_validation_responses' if not len(valid) else 'insufficient_training'
                      if len(train) < config['minimum_training_responses']
                      or train.n_participant.nunique() < config['minimum_training_participants'] else 'fitted')
            counts.append({**tags, 'fold': fold, 'training_responses': len(train), 'validation_responses': len(valid),
                           'training_participants': train.n_participant.nunique(),
                           'validation_participants': valid.n_participant.nunique(), 'status': reason})
            if reason != 'fitted':
                continue
            pred, fold_audits, coefs = fit_fold(train, valid, minutes, block == 'hr_steps')
            p = valid[['n_ema_id', 'n_participant', 'n_fold', 'pss_combo']].copy()
            for key, value in tags.items():
                p[key] = value
            for name, values in pred.items():
                p['prediction_' + name] = values
            predictions.append(p)
            audits.append({**tags, 'fold': fold, 'training_key_sha256': digest_object(train.n_ema_id.tolist()),
                           'validation_key_sha256': digest_object(valid.n_ema_id.tolist()), 'models': fold_audits})
            coefficients.extend({**tags, 'fold': fold, **c} for c in coefs)
    if not predictions:
        raise AuditError('No fold can be fit under the unchanged specification.')
    return (pd.concat(predictions, ignore_index=True), pd.DataFrame(counts), audits,
            pd.DataFrame(coefficients), pd.DataFrame(cohorts), pd.concat(membership, ignore_index=True))


def metrics(frame, prediction):
    residual = frame[prediction] - frame.pss_combo
    ae = residual.abs()
    return {'responses': len(frame), 'participants': frame.n_participant.nunique(),
            'mae': float(ae.mean()), 'rmse': float(np.sqrt((residual ** 2).mean())),
            'participant_mae': float(ae.groupby(frame.n_participant).mean().mean())}


def paired_bootstrap(frame, model, baseline, draws=2000, seed=20261002):
    delta = (frame[model] - frame.pss_combo).abs() - (frame[baseline] - frame.pss_combo).abs()
    grouped = delta.groupby(frame.n_participant, sort=True).agg(['sum', 'count', 'mean'])
    n = len(grouped)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, n, size=(draws, n))
    pooled = grouped['sum'].to_numpy()[sample].sum(axis=1) / grouped['count'].to_numpy()[sample].sum(axis=1)
    person = grouped['mean'].to_numpy()[sample].mean(axis=1)
    result = {'mae_difference': float(delta.mean()), 'participant_mae_difference': float(grouped['mean'].mean())}
    for label, values in [('mae', pooled), ('participant_mae', person)]:
        lo, hi = np.quantile(values, [.025, .975])
        result[label + '_difference_low'], result[label + '_difference_high'] = float(lo), float(hi)
    return result


def evaluate(predictions, config):
    scores, intervals = [], []
    for keys, frame in predictions.groupby(KEYS, sort=True):
        tags = dict(zip(KEYS, keys))
        names = [c.removeprefix('prediction_') for c in frame if c.startswith('prediction_') and frame[c].notna().all()]
        for name in names:
            scores.append({**tags, 'model': name, 'fold': 'pooled', **metrics(frame, 'prediction_' + name)})
            for fold, part in frame.groupby('n_fold'):
                scores.append({**tags, 'model': name, 'fold': str(fold), **metrics(part, 'prediction_' + name)})
            if name in ['mean', 'median']:
                continue
            baselines = ['mean', 'median', 'context'] + (['context_hr'] if name == 'context_hr_steps' else [])
            for baseline in baselines:
                if baseline == name:
                    continue
                intervals.append({**tags, 'model': name, 'baseline': baseline,
                                  **paired_bootstrap(frame, 'prediction_' + name, 'prediction_' + baseline,
                                                     config['bootstrap_draws'], config['bootstrap_seed'])})
    return pd.DataFrame(scores), pd.DataFrame(intervals)
