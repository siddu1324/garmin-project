"""Explicit predictor sets, unchanged outcomes, and matching before model fitting."""
import numpy as np
import pandas as pd
from garmin_ema_baseline.models import CONTEXT, LOOKBACKS, feature_columns
from garmin_ema_baseline.linkage import check_development
from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import verify_bundle

HISTORY = ['reference_mean_bpm', 'hr_mean_deviation_bpm', 'hr_latest_deviation_bpm']
META = ['n_ema_id', 'n_participant', 'n_partition', 'n_fold', 'n_survey_category',
        'pss_combo', 'context_available']
TASKS = {'three_class': 3, 'at_least_5': 2, 'at_least_10': 2}


def targets(score, task):
    s = np.asarray(score, float)
    valid = np.isfinite(s)
    if np.any(valid & ((s < 3) | (s > 15) | (s != np.floor(s)))):
        raise AuditError('Composite is outside the verified integer scale.')
    if task not in TASKS:
        raise AuditError('Unapproved classification task.')
    result = np.full(len(s), np.nan)
    if task == 'three_class':
        result[valid] = np.select([s[valid] <= 4, s[valid] <= 9], [0, 1], default=2)
    else:
        result[valid] = (s[valid] >= (5 if task == 'at_least_5' else 10)).astype(int)
    return result


def feature_set(view, minutes):
    if minutes not in LOOKBACKS:
        raise AuditError('Lookback outside frozen feature set.')
    if view == 'context':
        return list(CONTEXT), []
    if view == 'context_missingness':
        return list(CONTEXT), feature_columns(minutes, True)
    if view not in ['context_hr', 'context_hr_steps', 'context_hr_steps_history']:
        raise AuditError('Predictor view outside allowlist.')
    wear = feature_columns(minutes, view != 'context_hr')
    if view.endswith('_history'):
        if minutes != 60:
            raise AuditError('Earlier reference only exists for 60 minutes.')
        wear += HISTORY
    return list(CONTEXT) + wear, wear


def load_development(root, config, assignments):
    for name in ['baseline_bundle', 'feature_bundle', 'history_bundle', 'eligibility_bundle']:
        verify_bundle(root / config[name])
    original = pd.read_parquet(root / config['baseline_bundle'] / 'linked_development.parquet')
    check_development(original, assignments, 3706)
    items = original[['PSS_NERVOUS_value', 'PSS_COPE_value', 'PSS_CONTROL_value']]
    valid = items.notna().all(axis=1)
    if not items.stack().between(1, 5).all() or not (items.stack() == np.floor(items.stack())).all():
        raise AuditError('Existing item coding changed.')
    expected = (items.iloc[:, 0] + 6 - items.iloc[:, 1] + 6 - items.iloc[:, 2]).where(valid)
    if not np.allclose(expected, original.pss_combo, equal_nan=True):
        raise AuditError('Existing composite no longer matches item coding.')
    columns = META + CONTEXT + [c for m in LOOKBACKS for c in feature_columns(m, True)]
    f = original[columns].copy()
    history = pd.read_parquet(root / config['history_bundle'] / 'reference_features.parquet')
    check_development(history, assignments, 3706)
    aligned = history.set_index('n_ema_id').loc[f.n_ema_id]
    if not (aligned.n_participant.to_numpy() == f.n_participant.to_numpy()).all() or not (
        aligned.n_fold.to_numpy() == f.n_fold.to_numpy()).all():
        raise AuditError('History feature identity/fold mismatch.')
    for c in HISTORY:
        f[c] = aligned[c].to_numpy()
    for task in TASKS:
        f['target_' + task] = targets(f.pss_combo, task)
    numeric = CONTEXT + [c for m in LOOKBACKS for c in feature_columns(m, True)] + HISTORY
    if np.isinf(f[numeric].to_numpy(float)).any():
        raise AuditError('Unexpected infinite feature values.')
    return f.sort_values('n_ema_id').reset_index(drop=True)


def experiments(config):
    rows = []
    for task in TASKS:
        for minutes in (LOOKBACKS if task == 'three_class' else [60]):
            for handling in ['complete', 'imputed']:
                views = ['context', 'context_hr', 'context_hr_steps']
                if task == 'three_class' and minutes == 60 and handling == 'imputed':
                    views.append('context_missingness')
                rows.append(dict(task=task, minutes=minutes, cohort='joint', handling=handling, views=views))
    rows.append(dict(task='three_class', minutes=60, cohort='hr_native', handling='complete',
                     views=['context', 'context_hr']))
    for handling in ['complete', 'imputed']:
        rows.append(dict(task='three_class', minutes=60, cohort='history', handling=handling,
                         views=['context', 'context_hr', 'context_hr_steps', 'context_hr_steps_history']))
    for row in rows:
        row['experiment'] = f"{row['task']}__{row['minutes']}m__{row['cohort']}__{row['handling']}"
    return rows


def cohort_mask(frame, spec):
    base = frame['target_' + spec['task']].notna() & frame.context_available
    if spec['handling'] == 'imputed':
        return base
    cols = feature_columns(spec['minutes'], spec['cohort'] != 'hr_native')
    if spec['cohort'] == 'history':
        cols += HISTORY
    return base & frame[cols].notna().all(axis=1)
