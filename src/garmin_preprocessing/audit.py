"""Versioned window diagnostics and overlap-corrected aggregate coverage."""
import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_inventory.audit import add_hist, hist_quantile
from garmin_inventory.privacy import AuditError, sha256
from .common import (digest_object, input_guard, publish_bundle, read_config,
                     verify_bundle, write_json)
from .eligibility import (METRICS, REASONS, evaluate_windows, parameter_array,
                          selected_support, union_seconds)
from .normalize import normalize, write_parquet


def timing_summary(frame, counters, histograms):
    """Descriptive comparisons on original logical order; never a reconstructed clock."""
    if len(frame) < 2:
        return
    t, r = frame.n_epoch_s.to_numpy(), frame.n_value.to_numpy() / 1000
    dt = np.diff(t)
    counters.update(logical_ibi_rows=len(frame), zero_timestamp_differences=int((dt == 0).sum()),
                    gaps_above_5s=int((dt > 5).sum()), gaps_above_60s=int((dt > 60).sum()),
                    gaps_above_300s=int((dt > 300).sum()),
                    logical_rows_with_export_order_reversal=int(frame.q_export_order_backward.sum()),
                    repeated_timestamp_rows=int(frame.q_logical_repeated_timestamp.sum()),
                    conflicting_timestamp_rows=int(frame.q_conflicting_timestamp_value.sum()))
    plausible = np.isfinite(r) & (r >= .25) & (r <= 2.5)
    pairs = plausible[1:] & plausible[:-1] & (dt > 0)
    burst = pairs & (dt < .25 * np.minimum(r[1:], r[:-1]))
    counters['compressed_spacing_pairs_below_quarter_of_both_intervals'] += int(burst.sum())
    for label, mask in [('positive_plausible_pairs', pairs), ('within_5s_pairs', pairs & (dt <= 5)),
                        ('compressed_pairs', burst)]:
        counters[label] += int(mask.sum())
        for expected_name, expected in [('current', r[1:]), ('preceding', r[:-1])]:
            residual = dt[mask] - expected[mask]
            for tolerance in [.1, .25, .5]:
                counters[f'{label}_{expected_name}_within_{tolerance:g}s'] += int((abs(residual) <= tolerance + 1e-6).sum())
            # Keep long-gap statistics separate; bins are rounded milliseconds for timing diagnostics.
            if label != 'positive_plausible_pairs':
                add_hist(histograms[f'{label}_{expected_name}_signed_residual_s'], residual)
                add_hist(histograms[f'{label}_{expected_name}_absolute_residual_s'], abs(residual))
    pair_artifact = pairs & (dt <= 2 * np.maximum(r[1:], r[:-1]) + .25) & (abs(np.diff(r)) > .2 * r[:-1])
    counters['potential_abrupt_change_pairs'] += int(pair_artifact.sum())
    # Streaks are descriptive compressed-arrival clusters, not acquisition batches proved by data.
    changes = np.diff(np.r_[False, burst, False].astype(np.int8))
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    counters['compressed_spacing_clusters'] += len(lengths)
    counters['longest_compressed_cluster_pairs'] = max(counters['longest_compressed_cluster_pairs'], int(max(lengths, default=0)))


def arrays_for_ibi(folder, person_directory):
    p = folder / 'logical' / person_directory / 'ibi.parquet'
    if not p.exists():
        return pd.DataFrame(), np.array([], float), np.array([], float), *(np.array([], bool) for _ in range(3))
    columns = ['n_epoch_s', 'n_value', 'q_invalid_epoch', 'q_invalid_text_clock', 'q_clock_discrepancy',
               'q_unusual_offset', 'q_logical_repeated_timestamp', 'q_conflicting_timestamp_value',
               'q_export_order_backward']
    frame = pd.read_parquet(p, columns=columns)
    valid = frame.n_epoch_s.notna() & ~frame.q_invalid_epoch
    usable_clock = frame.loc[valid].copy()
    clock = usable_clock[['q_invalid_text_clock', 'q_clock_discrepancy', 'q_unusual_offset']].any(axis=1).to_numpy()
    return (usable_clock, usable_clock.n_epoch_s.to_numpy(float), usable_clock.n_value.to_numpy(float) / 1000,
            usable_clock.q_logical_repeated_timestamp.to_numpy(bool),
            usable_clock.q_conflicting_timestamp_value.to_numpy(bool), clock)


def audit_windows(root, normalized, normalization_manifest):
    config = read_config(root)
    if config['window_seconds'] != 300 or config['step_seconds'] != 60 or config['boundary_scenarios'] != ['interval_end', 'interval_start']:
        raise AuditError('This checkpoint requires five-minute windows, one-minute steps and the two documented boundary scenarios.')
    inputs = dict(normalization_version=normalization_manifest['version'],
                  normalization_manifest_sha256=sha256(normalized / 'manifest.json'),
                  policy_sha256=sha256(root / 'config/ibi_eligibility.json'),
                  policy_document_sha256=sha256(root / 'reports/ibi_eligibility_policies.md'),
                  code_sha256={n: sha256(Path(__file__).parent / n) for n in ['audit.py', 'eligibility.py']},
                  environment_sha256=sha256(root / 'uv.lock'))
    version = digest_object(inputs)[:16]
    parent = root / 'private/preprocessing/eligibility'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / version
    if destination.exists():
        return destination, verify_bundle(destination)
    build = Path(tempfile.mkdtemp(prefix=f'.building_{version}_', dir=parent))
    write_json(build / 'frozen_policies.json', config)  # Written before any retention evaluation.
    ema = pd.read_parquet(normalized / 'ema/development_ema.parquet', columns=[
        'n_ema_id', 'n_participant', 'n_partition', 'n_fold', 'n_survey_category',
        'n_cutoff_epoch', 'n_invalid_clock', 'n_start_precision'])
    if not ema.n_partition.eq('development').all() or not ema.n_fold.isin(range(5)).all():
        raise AuditError('Audit accepts development rows and original folds only.')
    people = json.loads((normalized / 'participant_directories.json').read_text())
    aggregates = defaultdict(Counter)
    rejected = defaultdict(Counter)
    timing, histograms = Counter(), defaultdict(Counter)
    ema_accounting, total_windows = [], 0
    policies = config['policies']
    nper = config['lookbacks_minutes'][-1] - 4
    artifact_args = (config['artifact_relative_change'], config['artifact_neighbor_max_ratio'],
                     config['artifact_neighbor_slack_seconds'])
    all_response_rows = 0
    for number, person in enumerate(people):
        e = ema.loc[ema.n_participant == person['participant']].reset_index(drop=True)
        if not e.n_fold.eq(person['fold']).all():
            raise AuditError('Participant/fold mismatch in normalized EMA.')
        all_response_rows += len(e)
        frame, t, rr, repeated, conflict, clock = arrays_for_ibi(normalized, person['directory'])
        timing_summary(frame, timing, histograms)
        valid_e = np.flatnonzero(~e.n_invalid_clock & np.isfinite(e.n_cutoff_epoch))
        event = np.repeat(valid_e, nper)
        step = np.tile(np.arange(nper, dtype=np.int64), len(valid_e))
        ends = e.n_cutoff_epoch.to_numpy()[event] - 60 * step
        total_windows += len(ends)
        windows = pd.DataFrame({'n_ema_id': e.n_ema_id.to_numpy()[event], 'n_step_index': step,
                                'n_window_start_epoch': ends - 300, 'n_window_end_epoch': ends,
                                'n_fold': person['fold'], 'n_partition': 'development',
                                'n_survey_category': e.n_survey_category.to_numpy()[event],
                                'interval_semantics_unconfirmed': True, 'normal_beat_status': 'unverified'})
        cases = {}
        for policy in policies:
            p = parameter_array(policy)
            for scenario_index, scenario in enumerate(config['boundary_scenarios']):
                masks, metrics, recorded = evaluate_windows(t, rr, repeated, conflict, clock, ends, scenario_index, p, *artifact_args)
                label = f'{policy["name"]}__{scenario}'
                cases[label] = (masks, metrics, recorded, scenario_index, p, policy['name'], scenario)
                windows[f'{label}__rejection_mask'] = masks
                windows[f'{label}__eligible'] = masks == 0
                for col, name in enumerate(METRICS):
                    windows[f'{label}__{name}'] = metrics[:, col]
        windows['recorded_ibi_count'] = recorded
        write_parquet(windows, build / 'windows' / f'{person["directory"]}.parquet')
        categories = ['All', 'Morning', 'Midday', 'Endday']
        for label, (masks, metrics, recorded, scenario_index, p, policy_name, scenario) in cases.items():
            for minutes in config['lookbacks_minutes']:
                in_lookback = step <= minutes - 5
                for category in categories:
                    event_scope = np.ones(len(e), bool) if category == 'All' else e.n_survey_category.eq(category).to_numpy()
                    window_scope = in_lookback & event_scope[event]
                    eligible = window_scope & (masks == 0)
                    key = (policy_name, scenario, minutes, category)
                    counter = aggregates[key]
                    valid_cutoffs = e.loc[event_scope & ~e.n_invalid_clock, 'n_cutoff_epoch'].dropna().to_numpy()
                    selected_ends = np.unique(ends[eligible])
                    count_ema = int(event_scope.sum())
                    counter.update(ema_denominator=count_ema, participant_denominator=int(count_ema > 0),
                                   invalid_cutoff_ema=int((event_scope & e.n_invalid_clock).sum()),
                                   window_instances=int(window_scope.sum()), eligible_windows=int(eligible.sum()),
                                   recorded_windows=int((window_scope & (recorded > 0)).sum()),
                                   eligible_ema=len(np.unique(event[eligible])), eligible_participants=int(eligible.any()),
                                   ema_with_ibi=int(count_ema if len(t) else 0),
                                   unique_requested_seconds=union_seconds(valid_cutoffs - minutes * 60, valid_cutoffs),
                                   unique_eligible_window_seconds=union_seconds(selected_ends - 300, selected_ends),
                                   unique_supported_seconds=selected_support(t, rr, repeated, conflict, clock, selected_ends,
                                                                            scenario_index, p, *artifact_args))
                    observed = window_scope & (recorded >= 2)
                    gaps = metrics[observed, 11]
                    counter['gap_observed_windows'] += len(gaps)
                    counter['sum_window_max_internal_gap_seconds'] += float(np.sum(gaps))
                    counter['maximum_window_internal_gap_seconds'] = max(counter['maximum_window_internal_gap_seconds'], float(np.max(gaps, initial=0)))
                    for bit, name in REASONS.items():
                        rejected[key][name] += int(((masks[window_scope] & bit) != 0).sum())
                # Every development EMA gets a diagnostic accounting row for every policy/scenario/lookback.
                for ei, row in e.iterrows():
                    chosen = in_lookback & (event == ei)
                    passed = chosen & (masks == 0)
                    b = ends[passed]
                    ema_accounting.append(dict(ema_id=row.n_ema_id, participant=row.n_participant, fold=int(row.n_fold),
                        partition='development', survey_category=row.n_survey_category,
                        policy=policy_name, scenario=scenario, lookback_minutes=minutes,
                        invalid_cutoff=bool(row.n_invalid_clock), has_garmin=person['has_garmin'], has_ibi=bool(len(t)),
                        candidate_windows=int(chosen.sum()), recorded_windows=int((chosen & (recorded > 0)).sum()),
                        eligible_windows=int(passed.sum()), eligible_nominal_union_seconds=union_seconds(b - 300, b)))
        if (number + 1) % 10 == 0:
            print(f'Eligibility audit: {number + 1}/{len(people)} development participants', flush=True)
    coverage, attrition = [], []
    for key in sorted(aggregates):
        policy, scenario, lookback, category = key
        base = dict(policy=policy, scenario=scenario, lookback_minutes=lookback, survey_category=category)
        counter = dict(aggregates[key])
        counter['mean_window_max_internal_gap_seconds'] = (counter.pop('sum_window_max_internal_gap_seconds') / counter['gap_observed_windows']
                                                         if counter['gap_observed_windows'] else None)
        coverage.append(dict(base, **counter))
        for reason, count in sorted(rejected[key].items()):
            attrition.append(dict(base, reason=reason, windows=count))
    quantiles = {name: {f'p{int(q * 100):02}': hist_quantile(hist, q) for q in [.05, .5, .95, .99]}
                 for name, hist in sorted(histograms.items())}
    accounting = pd.DataFrame(ema_accounting)
    write_parquet(accounting, build / 'ema_window_accounting.parquet')
    if all_response_rows != len(ema) or accounting.ema_id.nunique() != len(ema):
        raise AuditError('Not all development EMA records are accounted for.')
    expected = (~ema.n_invalid_clock).sum() * nper
    if total_windows != expected or accounting.groupby(['policy', 'scenario', 'lookback_minutes']).size().ne(len(ema)).any():
        raise AuditError('Window or EMA/policy denominator accounting failed.')
    payload = dict(version=version, normalization=normalization_manifest, inputs=inputs,
                   window_rows=total_windows, ema_accounting_rows=len(accounting),
                   ema_rows=len(ema), development_participants=len(people),
                   coverage=coverage, attrition=attrition, timing_counts=dict(timing), timing_quantiles=quantiles,
                   reason_bits={str(k): v for k, v in REASONS.items()},
                   all_development_ema_accounted=True, final_test_rows_processed=0,
                   normal_beat_status='unverified', targets_or_hrv_calculated=False)
    # Public payload excludes the identifying raw source catalog and participant directories.
    write_json(build / 'aggregate_results.json', payload)
    manifest = publish_bundle(build, destination, dict(version=version, inputs=inputs, window_rows=total_windows,
        ema_accounting_rows=len(accounting), all_development_ema_accounted=True, targets_or_hrv_calculated=False))
    return destination, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=os.environ.get('GARMIN_REFERENCE_PROJECT'))
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    try:
        if args.reference is None:
            raise AuditError('Set GARMIN_REFERENCE_PROJECT to the verified reference project.')
        reference, data = args.reference.resolve(), (args.data or root / 'data').resolve()
        input_guard(root, reference, data)
        normalized, nmanifest = normalize(root, reference, data)
        audited, _ = audit_windows(root, normalized, nmanifest)
        input_guard(root, reference, data)
        write_json(root / 'private/preprocessing/current_run.json', {
            'normalized': str(normalized.relative_to(root)), 'eligibility': str(audited.relative_to(root))})
        from .report import publish_report
        publish_report(root, normalized, audited)
        print('Completed immutable development preprocessing and eligibility audit', flush=True)
    except Exception as exc:
        print(str(exc) if isinstance(exc, AuditError) else f'Audit stopped ({type(exc).__name__}); source details withheld.', flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
