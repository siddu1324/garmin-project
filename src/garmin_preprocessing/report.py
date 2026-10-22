"""Publish immutable aggregate reports and scientific coverage figures only."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_inventory.privacy import AuditError, contains_private_token, public_scan_text, sha256
from garmin_inventory.render import table
from .common import artifact_hashes, digest_object, write_json


def figure_files(folder, coverage, timing):
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/garmin-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    colors = {'conservative': '#275d8c', 'screen': '#c46c22', 'timing_sensitivity': '#2a8864'}
    for (policy, scenario), g in coverage.loc[coverage.survey_category == 'All'].groupby(['policy', 'scenario']):
        g = g.sort_values('lookback_minutes')
        style = '-' if scenario == 'interval_end' else '--'
        label = f'{policy} / {scenario.replace("interval_", "")}'
        axes[0].plot(g.lookback_minutes, 100 * g.eligible_ema / g.ema_denominator,
                     style, marker='o', color=colors[policy], label=label, markersize=4)
        axes[1].plot(g.lookback_minutes, 100 * g.unique_supported_seconds / g.unique_requested_seconds,
                     style, marker='o', color=colors[policy], label=label, markersize=4)
    for ax in axes:
        ax.set_xscale('log')
        ticks = [5, 15, 30, 60, 120, 360, 720, 1440]
        ax.set_xticks(ticks, ['5m', '15m', '30m', '1h', '2h', '6h', '12h', '24h'])
        ax.set_xlabel('Lookback ending at the EMA cutoff')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Development EMA with ≥1 eligible window (%)')
    axes[0].set_title('Presence of a provisionally eligible segment')
    axes[1].set_ylabel('Unique acceptable interval support / requested time (%)')
    axes[1].set_title('Time union per participant; overlaps counted once')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('IBI screening sensitivities — normal-beat status remains unverified')
    fig.savefig(folder / 'coverage.png', dpi=160, metadata={'Software': 'garmin-preprocessing'})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    counts = timing['counts']
    denom = counts.get('within_5s_pairs', 0)
    x = np.arange(3)
    for i, expected in enumerate(['current', 'preceding']):
        values = [100 * counts.get(f'within_5s_pairs_{expected}_within_{tol:g}s', 0) / denom if denom else 0
                  for tol in [.1, .25, .5]]
        ax.bar(x + (i - .5) * .35, values, width=.35, label=f'{expected} IBI')
    ax.set_xticks(x, ['100 ms', '250 ms', '500 ms'])
    ax.set_ylabel('Positive-time plausible adjacent pairs ≤5 s (%)')
    ax.set_xlabel('Absolute timestamp-difference / IBI agreement tolerance')
    ax.set_ylim(0, 100)
    ax.set_title('Clock agreement cannot establish acquisition semantics')
    ax.legend()
    fig.savefig(folder / 'timing_agreement.png', dpi=160, metadata={'Software': 'garmin-preprocessing'})
    plt.close(fig)


def publish_report(root, normalized, audited):
    from .common import verify_bundle
    verify_bundle(audited)
    data = json.loads((audited / 'aggregate_results.json').read_text())
    norm = data['normalization']
    # Deliberately exclude the private per-partition hashes, source catalog and ID directory map.
    summary = {k: data[k] for k in ['version', 'inputs', 'window_rows', 'ema_accounting_rows', 'ema_rows',
               'development_participants', 'reason_bits', 'all_development_ema_accounted',
               'final_test_rows_processed', 'normal_beat_status', 'targets_or_hrv_calculated']}
    summary['normalization'] = {k: norm[k] for k in ['version', 'counts', 'sensor_rows', 'mapping_rows',
        'ema_rows', 'development_participants', 'development_participants_with_garmin',
        'raw_roundtrip_verified', 'final_test_rows_written', 'all_ema_folds_preserved']}
    summary['datasets'] = dict(normalized=str(normalized.relative_to(root)), eligibility=str(audited.relative_to(root)))
    report_key = digest_object({'audit': data['version'], 'renderer_sha256': sha256(Path(__file__))})[:12]
    parent = root / 'reports/ibi_eligibility'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f'{data["version"]}_{report_key}'
    if destination.exists():
        manifest = json.loads((destination / 'manifest.json').read_text())
        if artifact_hashes(destination) != manifest['artifact_sha256']:
            raise AuditError('Existing public report changed; refusing to overwrite.')
        return destination
    build = Path(tempfile.mkdtemp(prefix='.building_', dir=parent))
    coverage = pd.DataFrame(data['coverage'])
    all_scope = coverage.loc[coverage.survey_category == 'All'].copy()
    all_scope['eligible_ema_pct'] = 100 * all_scope.eligible_ema / all_scope.ema_denominator
    all_scope['unique_supported_minutes'] = all_scope.unique_supported_seconds / 60
    all_scope['unique_requested_minutes'] = all_scope.unique_requested_seconds / 60
    all_scope['unique_eligible_window_minutes'] = all_scope.unique_eligible_window_seconds / 60
    write_json(build / 'run_summary.json', summary)
    write_json(build / 'coverage.json', data['coverage'])
    write_json(build / 'attrition.json', data['attrition'])
    timing = dict(counts=data['timing_counts'], quantiles=data['timing_quantiles'])
    write_json(build / 'timing.json', timing)
    figure_files(build, coverage, timing)
    counts = norm['counts']
    t = data['timing_counts']
    agreement = []
    denominator = t.get('within_5s_pairs', 0)
    for expected in ['current', 'preceding']:
        for tolerance in [.1, .25, .5]:
            matched = t.get(f'within_5s_pairs_{expected}_within_{tolerance:g}s', 0)
            agreement.append(dict(expected_interval=expected, tolerance_seconds=tolerance,
                                  positive_pairs_within_5s=denominator, matching_pairs=matched,
                                  matching_percent=100 * matched / denominator if denominator else 0))
    failures = pd.DataFrame(data['attrition'])
    highlights = failures.loc[(failures.lookback_minutes == 5) & (failures.survey_category == 'All')]
    report = [
        '# Development preprocessing and five-minute IBI eligibility\n',
        f'The checkpoint preserves **{norm["ema_rows"]:,} development EMA records from 125 participants**, '
        f'and **{counts["physical_sensor_rows"]:,} original sensor records from '
        f'{norm["development_participants_with_garmin"]} development participants**. All 32 final-test '
        f'participants remain sealed; their {counts["sealed_final_test_sources"]} available sensor sources '
        'were skipped before CSV parsing. Exact frozen assignments and the five development folds are unchanged. '
        'No PSS target, HRV, imputation, reconstructed beats, final prediction dataset or model was produced.\n',
        '## Data organization and preservation\n',
        f'Normalized bundle: `{normalized.relative_to(root)}`.\n\n'
        '- `ema/development_ema.parquet`: all development EMA rows, raw strings, source schema/hash/row, '
        'participant provenance, original fold, clocks and cutoff flags. Survey items remain raw; no target is calculated.\n'
        '- `physical/<partition>/<sensor>.parquet`: every original Garmin row plus provenance, parsed '
        'time/value and quality flags. Original columns are unchanged strings; absent source columns remain '
        'null and are distinguishable from blanks through the source-schema record.\n'
        '- `logical/<partition>/<sensor>.parquet`: stable source/row references and parsed diagnostics '
        'after collapsing exact duplicates only. Raw values are recovered using `n_physical_row`.\n'
        '- Private source catalog, participant-directory map and immutable content manifest.\n',
        f'The physical view retains all copies. The logical view has **{counts["logical_sensor_rows"]:,} rows**, '
        f'with **{counts["exact_duplicate_excess_rows"]:,} exact-record excess rows** collapsed for logical analysis. '
        f'**{counts["conflicting_physical_rows"]:,} physical rows** belong to conflicting timestamp/value groups '
        '(across all sensors). Neither same-valued intervals at different times nor nonidentical records '
        'with matching timestamp/value are arbitrarily removed. Conflicting IBI records are never averaged; '
        'windows with remaining timestamp ties fail all policies. Original export order is retained by '
        'source/row provenance; chronological logical ordering does not prove acquisition order.\n',
        'Sources were processed in bounded CSV chunks, then participant/sensor partitions. Raw values '
        'and provenance were compared after Parquet round-trips. Version directories are content-addressed; '
        'existing complete bundles are verified and reused, never overwritten.\n',
        'EMA UTC is local time minus the row offset (minutes local-minus-UTC). Garmin Unix `ts` receives '
        'no timezone shift. Cutoff is actual UTC minus 60 seconds for minute precision, otherwise actual '
        'UTC for second precision. Clock disagreements and unusual offsets remain flags; no time is shifted '
        'to increase coverage. The sensor dictionary’s units remain exporter assumptions pending documentation.\n',
        '## Policies and interval-boundary uncertainty\n',
        'The exact definitions were saved before retention comparison in '
        '[the policy specification](../../ibi_eligibility_policies.md) and '
        '`config/ibi_eligibility.json`. The audit manifest records both hashes. Conservative, screen and '
        'timing_sensitivity are provisional alternatives, not validated eligibility rules. No policy '
        'was tuned to PSS associations or desired sample size. The third changes only the broader screen’s '
        'time tolerance. All require plausible intervals, no remaining timestamp ties/conflicts or clock '
        'flags, bounded expected-interval gaps and edge gaps, sufficient hypothetical support, a long enough '
        'unbroken sequence, bounded abrupt-change flags and interval-sum/elapsed-time agreement.\n',
        'Both interval-end and interval-start interpretations are carried through every result. Entire '
        'hypothesized intervals must fit inside the left-inclusive/right-exclusive timestamp window. '
        'Cross-boundary intervals remain raw records and receive explicit diagnostic exclusion counts. '
        'They are not corrected or borrowed from outside the window. Artifact flags use only pairs within '
        'the scenario’s contained subset; no post-window observation can alter a window’s eligibility. '
        'Every removed/rejected interval and failed pair breaks adjacency. Normal-beat status is **unverified** '
        'for every interval, including those passing all screens.\n',
        '## What the IBI clocks show\n',
        f'The logical development IBI stream has {t.get("logical_ibi_rows", 0):,} records, '
        f'{t.get("zero_timestamp_differences", 0):,} zero timestamp differences, '
        f'{t.get("gaps_above_5s", 0):,} gaps above five seconds and '
        f'{t.get("gaps_above_300s", 0):,} gaps above five minutes. '
        f'{t.get("compressed_spacing_pairs_below_quarter_of_both_intervals", 0):,} pairs have timestamps '
        'separated by less than one quarter of both associated IBI durations. '
        f'There are {t.get("compressed_spacing_clusters", 0):,} such compressed-spacing clusters; '
        f'the largest contains {t.get("longest_compressed_cluster_pairs", 0):,} pairs. '
        f'{t.get("logical_rows_with_export_order_reversal", 0):,} retained logical rows have an export-order '
        'timestamp reversal relative to the previous same-sensor source row.\n',
        table(agreement, ['expected_interval', 'tolerance_seconds', 'positive_pairs_within_5s',
                          'matching_pairs', 'matching_percent']) + '\n',
        'Current-IBI agreement is consistent with interval-end anchoring; preceding-IBI agreement is '
        'consistent with interval-start anchoring. Similar intervals can fit both explanations. Compressed '
        'spacing, ties, reversals and imperfect agreement are compatible with buffering/receipt timing, '
        'clock jitter or exporter behavior. They do not identify which mechanism occurred. These comparisons '
        'cannot prove acquisition semantics or validate reconstructed beat times. Signed and absolute '
        'residual quantiles, compressed-spacing counts and long-gap counts are in `timing.json`.\n',
        '![Timing agreement](timing_agreement.png)\n',
        '## Coverage and attrition\n',
        f'**{data["window_rows"]:,} five-minute window instances** were enumerated at one-minute steps, '
        f'with all six policy/scenario results stored per window. The private EMA accounting table has '
        f'**{data["ema_accounting_rows"]:,} rows** (one per EMA/policy/scenario/lookback). '
        'Every development EMA is retained in denominators, including participants without Garmin or '
        'eligible windows. No filtering on valid PSS items was performed. A window belongs to a lookback '
        'only if its full five-minute support lies inside it.\n',
        table(all_scope.to_dict('records'), ['policy', 'scenario', 'lookback_minutes', 'eligible_windows',
              'eligible_ema', 'ema_denominator', 'eligible_participants', 'unique_supported_minutes',
              'unique_requested_minutes']) + '\n',
        '![Coverage by lookback](coverage.png)\n',
        'Eligible-window counts are EMA/window instances; overlapping instances are correlated. '
        '**Unique supported minutes** are the union of acceptable hypothesized interval supports from '
        'eligible windows, formed within participant before summing. **Unique requested time** similarly '
        'unions EMA lookbacks. Neither repeats overlapping time. The additional nominal eligible-window '
        'union in `coverage.json` is not observed interval support. Category rows also union within '
        'participant/category; categories may overlap and must not be added. One eligible segment in a '
        '24-hour lookback never means complete 24-hour coverage.\n',
        'Five-minute-window rejection counts below are **overlapping reasons**, not mutually exclusive '
        'attrition steps; they must not be summed. Every window has a reason bitmask in the ignored '
        'diagnostic table, decoded by the immutable reason dictionary. `attrition.json` contains every '
        'lookback and survey category; `coverage.json` also contains denominator, presence, gap and '
        'unique-time summaries for all survey categories.\n',
        table(highlights.to_dict('records'), ['policy', 'scenario', 'reason', 'windows']) + '\n',
        '## Diagnostic datasets and verification\n',
        f'Eligibility bundle: `{audited.relative_to(root)}`.\n\n'
        '- `windows/<partition>.parquet`: recorded count, contained and acceptable counts, boundary '
        'exclusions, support duration, unbroken sequence, pair agreement, gaps, artifact flags, '
        'duration ratio, all rejection masks and all eligibility statuses.\n'
        '- `ema_window_accounting.parquet`: every development EMA across every lookback and policy/scenario, '
        'with original fold and presence/eligibility accounting; no outcome columns.\n'
        '- `frozen_policies.json`, `aggregate_results.json` and an immutable artifact manifest.\n',
        'The reusable verification command checks input hashes against the original inventory baseline, '
        'all versioned artifact hashes, row/fold/source accounting, deterministic kernel replay, '
        'output-free source notebooks and privacy tokens. Executed notebooks remain ignored. The separate '
        'verification report records the completed checks and synthetic test outcomes. No earlier '
        'validated inventory report or dataset was overwritten.\n',
        '## Interpretation and next step\n',
        'These are recording-quality sensitivities. Passing is not proof of sinus beats, artifact-free '
        'physiology or clinical validity. The differences between policies and timestamp hypotheses should '
        'remain visible rather than selecting the largest cohort. Exporter semantics, possible buffering, '
        'IBI units, sensor modality, beat annotations and actual-start rounding remain unresolved.\n',
        'Next, freeze an explicitly justified HRV input contract using these development-only diagnostics '
        'and exporter information where available. Cross-check synthetic segment behavior in NeuroKit2 '
        'and HeartPy before calculating real time-domain HRV. Use only contiguous eligible input, preserve '
        'gap boundaries and both timing hypotheses until resolved; do not concatenate acceptable intervals '
        'across rejected records. VLF remains deferred. Garmin stress and quality/coverage diagnostics '
        'remain excluded from initial predictors. Planned later lookback summaries remain mean, SD, '
        'minimum, maximum, trend and latest, with median/IQR additional candidates.\n',
    ]
    (build / 'report.md').write_text('\n'.join(report))
    (build / 'coverage_table.md').write_text('# Aggregate coverage by survey category\n\n' + table(
        coverage.to_dict('records'), ['policy', 'scenario', 'lookback_minutes', 'survey_category',
        'ema_denominator', 'eligible_windows', 'eligible_ema', 'eligible_participants',
        'unique_supported_seconds', 'unique_requested_seconds', 'maximum_window_internal_gap_seconds']) + '\n')
    tokens = json.loads((root / 'private/privacy_tokens.json').read_text())
    for path in build.iterdir():
        if path.suffix in {'.md', '.json', '.svg'} and contains_private_token(public_scan_text(path), tokens['identifiers'], tokens['names']):
            raise AuditError('Public report privacy scan flagged a known token; details withheld.')
    write_json(build / 'manifest.json', dict(audit_version=data['version'], renderer_version=report_key,
                                            artifact_sha256=artifact_hashes(build)))
    build.rename(destination)
    return destination
