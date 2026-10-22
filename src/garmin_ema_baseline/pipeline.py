"""Guarded immutable development-only outcome linkage and baseline entrypoint."""
import argparse
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_inventory.audit import PSS
from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (digest_object, frozen_assignments, input_guard,
                                         publish_bundle, verify_bundle, write_json)
from garmin_preprocessing.normalize import write_parquet
from .linkage import link_outcomes, questionnaire_evidence
from .models import HR, LOOKBACKS, evaluate, feature_columns, run_models


def records(frame):
    # pandas serialization converts missing values to JSON null, never non-standard NaN.
    return json.loads(frame.to_json(orient='records', double_precision=12))


def verified_inputs(root, config):
    normalized, features = root / config['normalized_bundle'], root / config['feature_bundle']
    verify_bundle(normalized)
    manifest = verify_bundle(features)
    supersession = json.loads((root / 'reports/window_supersession.json').read_text())
    corrected = root / supersession['corrected_eligibility']
    verify_bundle(corrected)
    inputs = manifest['inputs']
    if (supersession['status'] != 'complete'
            or inputs['corrected_eligibility'] != supersession['corrected_eligibility']
            or inputs['corrected_eligibility_manifest'] != sha256(corrected / 'manifest.json')
            or supersession['corrected_manifest_sha256'] != sha256(corrected / 'manifest.json')
            or inputs['normalized_manifest'] != sha256(normalized / 'manifest.json')):
        raise AuditError('Feature parent does not match the current corrected provenance.')
    return normalized, features


def outcome_summary(linked):
    rows = []
    for category, f in [('all', linked)] + list(linked.groupby('n_survey_category')):
        valid = f.pss_combo.dropna()
        rows.append({'survey_category': category, 'responses': len(f),
                     'participants': f.n_participant.nunique(), 'available': len(valid),
                     'missing': len(f) - len(valid), 'mean': valid.mean(), 'sd': valid.std(ddof=1),
                     'median': valid.median(), 'minimum': valid.min(), 'maximum': valid.max()})
    distribution = linked.groupby(['n_survey_category', 'pss_combo']).size().reset_index(name='responses')
    reasons = linked.groupby(['n_survey_category', 'outcome_reason']).size().reset_index(name='responses')
    items = []
    for category, f in linked.groupby('n_survey_category'):
        for name in PSS:
            for status, count in f[name + '_status'].value_counts().items():
                items.append({'survey_category': category, 'item': name, 'status': status, 'cells': int(count)})
            # Only literal permitted missing tokens, never arbitrary malformed raw values, leave memory.
            for raw, count in f.loc[f[name + '_status'].eq('missing'), name + '_raw'].fillna('<null>').value_counts().items():
                label = '<empty>' if str(raw) == '' else '<whitespace>' if not str(raw).strip() else str(raw)
                items.append({'survey_category': category, 'item': name, 'missing_token': label, 'cells': int(count)})
    return {'summary': records(pd.DataFrame(rows)), 'distribution': records(distribution),
            'reasons': records(reasons), 'item_audit': items}


def availability(linked, quality):
    rows, diagnostic = [], []
    valid_target = linked.pss_combo.notna() & linked.context_available
    for minutes in LOOKBACKS:
        for block in ['hr', 'hr_steps']:
            available = np.isfinite(linked[feature_columns(minutes, block == 'hr_steps')]).all(axis=1)
            for label, mask in [('available', available), ('unavailable', ~available)]:
                selected = linked.loc[mask]
                for category, f in [('all', selected)] + list(selected.groupby('n_survey_category')):
                    y = f.pss_combo.dropna()
                    rows.append({'lookback_minutes': minutes, 'block': block, 'availability': label,
                                 'survey_category': category, 'responses': len(f),
                                 'participants': f.n_participant.nunique(), 'valid_outcomes': len(y),
                                 'missing_outcomes': len(f) - len(y), 'pss_mean': y.mean(),
                                 'pss_median': y.median(), 'pss_sd': y.std(ddof=1)})
            for label, mask in [('all_emas', pd.Series(True, index=linked.index)),
                                ('matched_model_cohort', available & valid_target)]:
                keys = set(linked.loc[mask, 'n_ema_id'])
                for sensor in ['hr', 'steps']:
                    q = quality.loc[quality.n_ema_id.isin(keys) & quality.lookback_minutes.eq(minutes)
                                    & quality.sensor.eq(sensor)].copy()
                    q['occupied_minute_fraction'] = q.occupied_minute_bins / q.possible_minute_bins
                    q['readings_per_lookback_minute'] = q.valid_readings / minutes
                    fields = (['recorded_rows', 'valid_readings', 'occupied_minute_fraction',
                               'readings_per_lookback_minute', 'latest_age_seconds', 'observed_span_seconds',
                               'maximum_internal_gap_seconds', 'maximum_gap_including_edges_seconds']
                              if sensor == 'hr' else ['supported_pairs', 'supported_pair_seconds', 'supported_pair_fraction'])
                    for field in fields:
                        x = q[field].dropna()
                        diagnostic.append({'lookback_minutes': minutes, 'block': block, 'cohort': label,
                                           'sensor': sensor, 'measure': field, 'responses': len(q),
                                           'observed': len(x), 'median': x.median(),
                                           'q25': x.quantile(.25), 'q75': x.quantile(.75),
                                           'minimum': x.min(), 'maximum': x.max()})
    statuses = quality.loc[quality.sensor.isin(['hr', 'steps'])].groupby(
        ['sensor', 'lookback_minutes', 'n_survey_category', 'recording_status']).size().reset_index(name='responses')
    return pd.DataFrame(rows), pd.DataFrame(diagnostic), statuses


def run(root, reference):
    root, reference = Path(root), Path(reference)
    config = json.loads((root / 'config/ema_baseline.json').read_text())
    freeze = json.loads((root / 'private/ema_baseline/prefit_specification.json').read_text())
    if (freeze['config_sha256'] != sha256(root / 'config/ema_baseline.json')
            or freeze['specification_sha256'] != sha256(root / 'reports/ema_baseline_specification.md')
            or config['hr_features'] != HR or config['lookbacks_minutes'] != LOOKBACKS
            or config['ridge_alpha'] != 1 or config['ridge_solver'] != 'svd'):
        raise AuditError('Frozen pre-fit specification changed.')
    input_guard(root, reference, root / 'data')
    normalized, feature_bundle = verified_inputs(root, config)
    if freeze['features_sha256'] != sha256(feature_bundle / 'manifest.json'):
        raise AuditError('Pre-fit feature bundle changed.')
    assignments, _ = frozen_assignments(root, reference)
    inputs = {'normalized_manifest': sha256(normalized / 'manifest.json'),
              'features_manifest': sha256(feature_bundle / 'manifest.json'),
              'assignments_sha256': sha256(root / 'private/frozen_participant_assignments.parquet'),
              'supersession_sha256': sha256(root / 'reports/window_supersession.json'),
              'config_sha256': freeze['config_sha256'], 'specification_sha256': freeze['specification_sha256'],
              'prefit_record_sha256': sha256(root / 'private/ema_baseline/prefit_specification.json'),
              'uv_lock_sha256': sha256(root / 'uv.lock'),
              'code_sha256': {p.name: sha256(p) for p in sorted((root / 'src/garmin_ema_baseline').glob('*.py'))
                              if p.name not in ['report.py', 'verify.py']},
              'versions': {p: version(p) for p in ['scikit-learn', 'numpy', 'pandas', 'scipy', 'pyarrow']}}
    identifier = digest_object(inputs)[:16]
    parent = root / 'private/ema_baseline/runs'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / identifier
    if destination.exists():
        verify_bundle(destination)
        return destination
    evidence = questionnaire_evidence(root)
    # Only already normalized development EMAs are read. Never reopen all-cohort response CSVs.
    ema = pd.read_parquet(normalized / 'ema/development_ema.parquet')
    features = pd.read_parquet(feature_bundle / 'features.parquet')
    linked = link_outcomes(ema, features, assignments)
    if linked.n_participant.nunique() != 125:
        raise AuditError('All development participants must remain accounted for.')
    quality = pd.read_parquet(feature_bundle / 'quality.parquet')
    pred, counts, fit_audits, coefficients, cohorts, membership = run_models(linked, config)
    scores, intervals = evaluate(pred, config)
    avail, diagnostics, recording = availability(linked, quality)
    build = Path(tempfile.mkdtemp(prefix='.building_', dir=parent))
    write_parquet(linked, build / 'linked_development.parquet')
    write_parquet(pred, build / 'out_of_fold_predictions.parquet')
    write_parquet(membership, build / 'cohort_membership.parquet')
    write_json(build / 'training_audit.json', fit_audits)
    tables = {'fold_counts': counts, 'metrics': scores, 'paired_intervals': intervals,
              'coefficients': coefficients, 'cohorts': cohorts, 'availability': avail,
              'quality_diagnostics': diagnostics, 'recording_status': recording}
    for name, table in tables.items():
        write_json(build / (name + '.json'), records(table))
    write_json(build / 'outcomes.json', outcome_summary(linked))
    write_json(build / 'questionnaire_evidence.json', evidence)
    write_json(build / 'frozen_specification.json', config)
    # Re-read and verify all parent artifacts and raw/reference integrity after computation.
    verified_inputs(root, config)
    input_guard(root, reference, root / 'data')
    publish_bundle(build, destination, {'version': identifier, 'inputs': inputs,
                                      'linked_rows': len(linked), 'prediction_rows': len(pred),
                                      'membership_rows': len(membership), 'final_test_access': 'identity-only assignments'})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--reference', type=Path, default=os.environ.get('GARMIN_REFERENCE_PROJECT'))
    args = parser.parse_args()
    if args.reference is None:
        parser.error('Set GARMIN_REFERENCE_PROJECT or pass --reference.')
    result = run(args.root, args.reference)
    print('Verified development baseline bundle:', result.relative_to(args.root))


if __name__ == '__main__':
    main()
