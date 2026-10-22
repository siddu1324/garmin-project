"""Independent scalar target/join and normal-equation prediction verification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_inventory.audit import MISSING, PSS
from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import verify_bundle, write_json
from .models import KEYS, LOOKBACKS, feature_columns, model_columns


def independent_prediction(train, valid, name, minutes):
    """No sklearn objects: population scaling, training categories and ridge normal equations."""
    y = train.pss_combo.to_numpy(float)
    if name == 'mean':
        return np.full(len(valid), sum(y) / len(y))
    if name == 'median':
        return np.full(len(valid), float(np.median(y)))
    numeric, category = model_columns(name, minutes)
    x, v = train[numeric].to_numpy(float), valid[numeric].to_numpy(float)
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale[scale == 0] = 1
    x, v = (x - mean) / scale, (v - mean) / scale
    if category:
        levels = sorted(set(train.n_survey_category))
        x = np.column_stack([x] + [train.n_survey_category.eq(c).to_numpy(float) for c in levels])
        v = np.column_stack([v] + [valid.n_survey_category.eq(c).to_numpy(float) for c in levels])
    center = x.mean(axis=0)
    xc = x - center
    coef = np.linalg.solve(xc.T @ xc + np.eye(x.shape[1]), xc.T @ (y - y.mean()))
    return (v - center) @ coef + y.mean()


def verify_run(root, bundle):
    root, bundle = Path(root), Path(bundle)
    manifest = verify_bundle(bundle)
    config = json.loads((bundle / 'frozen_specification.json').read_text())
    source = pd.read_parquet(root / config['normalized_bundle'] / 'ema/development_ema.parquet')
    features = pd.read_parquet(root / config['feature_bundle'] / 'features.parquet')
    linked = pd.read_parquet(bundle / 'linked_development.parquet')
    predictions = pd.read_parquet(bundle / 'out_of_fold_predictions.parquet')
    assignments = pd.read_parquet(root / 'private/frozen_participant_assignments.parquet')
    dev = assignments.loc[assignments.partition.eq('development')].set_index('participant')
    if len(linked) != 3706 or linked.n_participant.nunique() != 125:
        raise AuditError('Independent row accounting failed.')
    for frame in [source, features, linked]:
        if (not frame.n_partition.eq('development').all()
                or not frame.n_fold.eq(frame.n_participant.map(dev.fold)).all()
                or frame.n_ema_id.duplicated().any()):
            raise AuditError('Independent split check failed.')
    if set(predictions.n_participant) - set(dev.index):
        raise AuditError('Prediction participant separation failed.')
    aligned = linked.set_index('n_ema_id').loc[source.n_ema_id]
    for col in features:
        pd.testing.assert_series_equal(aligned[col].reset_index(drop=True) if col != 'n_ema_id'
                                       else pd.Series(aligned.index, name=col),
                                       features.set_index('n_ema_id').loc[source.n_ema_id, col].reset_index(drop=True)
                                       if col != 'n_ema_id' else source[col].reset_index(drop=True),
                                       check_dtype=False)
    scores_checked = 0
    for i, row in enumerate(source.itertuples(index=False)):
        values = []
        for item in PSS:
            raw = getattr(row, item)
            try:
                v = float(str(raw).strip())
                valid = str(raw).strip() not in MISSING and np.isfinite(v) and v.is_integer() and 1 <= v <= 5
            except (TypeError, ValueError):
                valid, v = False, np.nan
            values.append(v if valid else np.nan)
        expected = values[0] + (6 - values[1]) + (6 - values[2])
        actual = aligned.pss_combo.iloc[i]
        if not (np.isnan(expected) and np.isnan(actual)) and expected != actual:
            raise AuditError('Independent scalar target calculation failed.')
        scores_checked += 1
    fits, largest_error = 0, 0.
    group_count = 0
    for tags, p in predictions.groupby(KEYS, sort=True):
        scope, block, minutes = tags
        needs_steps = block == 'hr_steps'
        eligible = linked.pss_combo.notna() & linked.context_available
        for m in LOOKBACKS if scope == 'common' else [minutes]:
            for col in feature_columns(m, needs_steps):
                eligible &= linked[col].map(np.isfinite)
        cohort = linked.loc[eligible].sort_values('n_ema_id', kind='stable')
        if set(cohort.n_ema_id) != set(p.n_ema_id) or p.n_ema_id.duplicated().any():
            raise AuditError('Independent matched cohort accounting failed.')
        for fold in range(5):
            train, valid = cohort.loc[cohort.n_fold.ne(fold)], cohort.loc[cohort.n_fold.eq(fold)]
            if set(train.n_participant) & set(valid.n_participant):
                raise AuditError('Independent participant separation failed.')
            observed = p.set_index('n_ema_id').loc[valid.n_ema_id]
            for name in ['mean', 'median', 'context', 'hr', 'context_hr'] + (['context_hr_steps'] if needs_steps else []):
                expected = independent_prediction(train, valid, name, minutes)
                actual = observed['prediction_' + name].to_numpy(float)
                error = float(np.max(np.abs(expected - actual)))
                largest_error = max(largest_error, error)
                if not np.allclose(expected, actual, atol=1e-9, rtol=1e-9):
                    raise AuditError('Independent normal-equation prediction agreement failed.')
                fits += 1
        group_count += 1
    results = {'version': bundle.name, 'manifest_sha256': sha256(bundle / 'manifest.json'),
               'scalar_scores_verified': scores_checked, 'linked_rows': len(linked),
               'development_participants': 125, 'sealed_final_test_participants': 32,
               'matched_comparisons': group_count, 'independent_model_fits': fits,
               'maximum_prediction_absolute_discrepancy': largest_error,
               'prediction_rows': len(predictions), 'valid_outcomes': int(linked.pss_combo.notna().sum()),
               'all_original_features_preserved': True, 'all_folds_preserved': True,
               'verified_bundle_complete': manifest['complete']}
    destination = root / 'private/ema_baseline' / ('independent_verification_' + bundle.name + '.json')
    if destination.exists():
        if json.loads(destination.read_text()) != results:
            raise AuditError('Existing independent verification changed.')
    else:
        write_json(destination, results)
    return results
