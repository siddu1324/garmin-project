"""Read-only verification of development bundles, separation and deterministic replay."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from garmin_inventory.privacy import AuditError, scan_public
from .audit import arrays_for_ibi
from .common import artifact_hashes, frozen_assignments, input_guard, read_config, verify_bundle
from .eligibility import METRICS, evaluate_windows, parameter_array


def require(condition):
    if not condition:
        raise AuditError('Verification invariant failed; private details withheld.')


def verify(root, reference, data):
    input_guard(root, reference, data)
    assignments, dev = frozen_assignments(root, reference)
    folds = assignments.set_index('participant').fold.to_dict()
    run = json.loads((root / 'private/preprocessing/current_run.json').read_text())
    normalized, audited = root / run['normalized'], root / run['eligibility']
    nm, am = verify_bundle(normalized), verify_bundle(audited)
    config = read_config(root)
    require(json.loads((audited / 'frozen_policies.json').read_text()) == config)
    people = json.loads((normalized / 'participant_directories.json').read_text())
    require({p['participant'] for p in people} == dev)
    source = json.loads((normalized / 'source_catalog.json').read_text())['garmin']
    require(all(s['participant'] in dev and s['fold'] == folds[s['participant']] for s in source))
    source_rows = {s['source_id']: s['rows'] for s in source}
    physical_rows, logical_rows, excess, invalid_ibi_clocks = 0, 0, 0, 0
    observed_source_rows = Counter()
    metadata_columns = ['n_participant', 'n_fold', 'n_partition', 'n_source_id', 'n_source_row',
                        'q_exact_record_excess', 'q_invalid_epoch']
    for person in people:
        require(person['fold'] == folds[person['participant']])
        for path in sorted((normalized / 'physical' / person['directory']).glob('*.parquet')):
            physical_rows += pq.ParquetFile(path).metadata.num_rows
            for batch in pq.ParquetFile(path).iter_batches(columns=metadata_columns, batch_size=100000):
                f = batch.to_pandas()
                require(f.n_participant.eq(person['participant']).all() and f.n_fold.eq(person['fold']).all()
                        and f.n_partition.eq('development').all())
                require(f.n_source_id.isin(source_rows).all() and f.n_source_row.ge(1).all()
                        and f.n_source_row.le(f.n_source_id.map(source_rows)).all())
                observed_source_rows.update(f.n_source_id.value_counts().to_dict())
                excess += int(f.q_exact_record_excess.sum())
                if path.stem == 'ibi':
                    invalid_ibi_clocks += int(f.q_invalid_epoch.sum())
            logical_rows += pq.ParquetFile(normalized / 'logical' / person['directory'] / path.name).metadata.num_rows
    require(dict(observed_source_rows) == source_rows)
    require(physical_rows == nm['counts']['physical_sensor_rows'])
    require(logical_rows == nm['counts']['logical_sensor_rows'] and physical_rows == logical_rows + excess)
    # An invalid clock cannot be placed in a time window; this checkpoint has none in IBI.
    require(invalid_ibi_clocks == 0)
    ema = pd.read_parquet(normalized / 'ema/development_ema.parquet', columns=[
        'n_ema_id', 'n_participant', 'n_fold', 'n_partition', 'n_cutoff_epoch', 'n_invalid_clock'])
    require(set(ema.n_participant) == dev and ema.n_partition.eq('development').all()
            and ema.n_fold.eq(ema.n_participant.map(folds)).all())
    accounting = pd.read_parquet(audited / 'ema_window_accounting.parquet')
    require(accounting.partition.eq('development').all() and set(accounting.participant) == dev
            and accounting.fold.eq(accounting.participant.map(folds)).all())
    require(len(accounting) == len(ema) * 48 and accounting.ema_id.nunique() == len(ema))
    require(accounting.groupby(['policy', 'scenario', 'lookback_minutes']).size().eq(len(ema)).all())
    require(accounting.eligible_windows.le(accounting.recorded_windows).all()
            and accounting.recorded_windows.le(accounting.candidate_windows).all())
    payload = json.loads((audited / 'aggregate_results.json').read_text())
    for row in payload['coverage']:
        a = accounting.loc[(accounting.policy == row['policy']) & (accounting.scenario == row['scenario'])
                           & (accounting.lookback_minutes == row['lookback_minutes'])]
        if row['survey_category'] != 'All':
            a = a.loc[a.survey_category == row['survey_category']]
        require(len(a) == row['ema_denominator'] and int(a.eligible_windows.sum()) == row['eligible_windows']
                and int(a.eligible_windows.gt(0).sum()) == row['eligible_ema']
                and a.loc[a.eligible_windows.gt(0), 'participant'].nunique() == row['eligible_participants'])
        require(-1e-6 <= row['unique_supported_seconds'] <= row['unique_eligible_window_seconds'] + 1e-4
                <= row['unique_requested_seconds'] + 1e-3)
    replayed, windows = 0, 0
    args = (config['artifact_relative_change'], config['artifact_neighbor_max_ratio'], config['artifact_neighbor_slack_seconds'])
    for person in people:
        f = pd.read_parquet(audited / 'windows' / f'{person["directory"]}.parquet')
        windows += len(f)
        e = ema.loc[ema.n_participant == person['participant']].set_index('n_ema_id')
        require(f.n_ema_id.isin(e.index).all() and f.n_fold.eq(person['fold']).all()
                and f.n_partition.eq('development').all())
        from garmin_hr_activity.boundaries import verify_grid
        verify_grid(f[['n_ema_id', 'n_step_index', 'n_window_start_epoch', 'n_window_end_epoch']],
                    e.n_cutoff_epoch.to_dict(), config['lookbacks_minutes'])
        indices = np.linspace(0, len(f)-1, min(32, len(f)), dtype=int) if len(f) else np.array([], int)
        for policy in config['policies']:
            for scenario in config['boundary_scenarios']:
                label = f'{policy["name"]}__{scenario}'
                require(np.all(f[f'{label}__eligible'] == f[f'{label}__rejection_mask'].eq(0)))
                indices = np.union1d(indices, np.flatnonzero(f[f'{label}__eligible'])[:5])
        chosen = f.iloc[indices]
        _, t, rr, repeated, conflict, clock = arrays_for_ibi(normalized, person['directory'])
        for policy in config['policies']:
            for si, scenario in enumerate(config['boundary_scenarios']):
                label = f'{policy["name"]}__{scenario}'
                masks, metrics, recorded = evaluate_windows(t, rr, repeated, conflict, clock,
                    chosen.n_window_end_epoch.to_numpy(), si, parameter_array(policy), *args)
                np.testing.assert_array_equal(masks, chosen[f'{label}__rejection_mask'])
                np.testing.assert_array_equal(recorded, chosen.recorded_ibi_count)
                np.testing.assert_array_equal(metrics, chosen[[f'{label}__{m}' for m in METRICS]].to_numpy())
                replayed += len(chosen)
    require(windows == am['window_rows'])
    tokens = json.loads((root / 'private/privacy_tokens.json').read_text())
    scanned = scan_public(root, tokens['identifiers'], tokens['names'])
    public_bundles = 0
    for path in (root / 'reports/ibi_eligibility').glob('*/manifest.json'):
        m = json.loads(path.read_text())
        require(artifact_hashes(path.parent) == m['artifact_sha256'])
        public_bundles += 1
    input_guard(root, reference, data)
    return dict(normalized_version=nm['version'], audit_version=am['version'],
        physical_rows=physical_rows, logical_rows=logical_rows, exact_duplicate_excess=excess,
        ema_rows=len(ema), window_rows=windows, ema_accounting_rows=len(accounting),
        deterministic_policy_window_replays=replayed, invalid_ibi_timestamps=invalid_ibi_clocks,
        public_text_artifacts_scanned=scanned, public_bundles_hash_verified=public_bundles,
        raw_and_reference_integrity=True, immutable_bundle_hashes=True,
        frozen_participant_and_fold_separation=True, all_ema_accounted=True,
        coverage_aggregate_reconciliation=True, source_row_accounting=True,
        final_test_records_processed=0, source_notebooks_output_free=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=os.environ.get('GARMIN_REFERENCE_PROJECT'))
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    try:
        if args.reference is None:
            raise AuditError('Set GARMIN_REFERENCE_PROJECT before verification.')
        result = verify(Path.cwd(), args.reference, args.data or Path.cwd() / 'data')
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(str(exc) if isinstance(exc, AuditError) else f'Verification failed ({type(exc).__name__}); private details withheld.')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
