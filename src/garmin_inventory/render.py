"""Public reports accept aggregate dictionaries only."""
import json
from pathlib import Path


def table(rows, columns):
    def fmt(value):
        if value is None:
            return 'unavailable'
        if isinstance(value, float):
            return f'{value:,.3f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(row.get(c, '—')) for c in columns) + ' |' for row in rows])


def publish(root, report, dictionary):
    root = Path(root)
    out = root / 'reports'
    out.mkdir(exist_ok=True)
    s = report['structural']
    g = s['garmin']
    split = report['frozen_split']
    sections = [
        '# Corrected Garmin / EMA inventory\n',
        'Recomputed from the local raw archives. The inventory computation performs no final linkage, '
        'PSS score calculation, HRV, imputation, feature selection or modeling, and never uploads participant data. '
        'Only structural accounting uses the whole cohort. All value, clock, recording-pattern '
        'and pre-EMA history summaries below use the frozen development participants. '
        'Final-test outcomes and exploratory sensor summaries remain untouched.\n',
        '## Sources and cohort\n',
        f'The local inputs contain **{s["ema_rows"]:,} EMA responses from {s["ema_participants"]} participants**, '
        f'and **{s["garmin_csv_tables"]} Garmin CSV tables with {g["physical_rows"]:,} physical rows**. '
        f'The Garmin tables map to {g["distinct_mapped_participants"]} participants; '
        f'{g["ema_participants_without_garmin"]} EMA participants have no Garmin source. '
        'There are three top-level archives: three corrected EMA CSVs, six questionnaire PDFs, '
        'and 154 nested Garmin ZIPs. Members are read in memory without extraction.\n',
        table(s['ema_tables'], ['category', 'rows', 'participants', 'fields']) + '\n',
        f'Local duplicate-export files: {g["duplicate_export_files"]}; logical rows: {g["logical_rows"]:,}. '
        f'{s.get("local_garmin_content_hashes_matching_reference", "Unknown")} local Garmin CSV content hashes '
        'match the corrected-v2 source catalog. '
        f'{s.get("local_ema_content_hashes_matching_reference", "Unknown")} EMA CSV content hashes also match '
        'the corrected-v2 EMA source catalog. '
        f'The reference catalog has {s.get("reference_extra_physical_sources", "unknown")} extra physical source '
        f'and {s.get("reference_duplicate_export_rows", "unknown")} duplicate-export rows. '
        'That extra copy is not part of the current local input. Legacy All/INS exports, old features, '
        'models and prior results are not inputs to this inspection.\n',
        '## Identity and split verification\n',
        f'Frozen assignments available: **{split["available"]}**. Development: '
        f'**{split["development_participants"]}**; final test: **{split["final_test_participants"]}**. '
        'The original five development folds and assignment bytes are preserved in ignored private storage. '
        'No new random split is generated.\n',
        f'Assignment SHA-256: `{split["assignment_sha256"]}`.\n',
        f'EMA participant/instance duplicate excess: {s["repeated_participant_instance_excess"]}; '
        f'participant/instance/survey-category duplicate excess: {s["repeated_participant_instance_category_excess"]}. '
        f'Across Garmin and EMA, {g["ambiguous_respondent_keys"]} respondent key maps to multiple participant keys. '
        f'Current/reference crosswalk pairs: {g["current_crosswalk_pairs"]}/{g["reference_crosswalk_pairs"]}; '
        f'current-only/reference-only pairs: {g["pairs_only_current"]}/{g["pairs_only_reference"]}.\n',
        table([dict(method=k, rows=v) for k, v in g['mapping_methods_rows'].items()], ['method', 'rows']) + '\n',
        'Use exact `mbl_cod` where present and an unambiguous, content-verified `rsp_id` crosswalk. '
        'Respondent keys alone are not universally unique. Missing-key sources require exact known filename '
        'tokens corroborated against the reference normalized provenance by CSV content hash and participant. '
        'This checks consistency with the prior verified mapping; it is not independent identity evidence '
        'inside a keyless raw CSV. Do not generalize filename-only mapping to new exports. '
        'Preserve source hash and row number in future preprocessing.\n',
        '## Questionnaire coding\n',
        'All three corrected tables use `PSS_NERVOUS`, `PSS_COPE`, and `PSS_CONTROL`. '
        'The questionnaire PDFs map exported codes 1–5 to displayed labels 0–4. '
        'The agreed score is `nervous + (6 - cope) + (6 - control)`, range 3–15; '
        'all three items must be integer codes in 1–5. Missing is not zero. '
        'A recoded 0–4 version would differ by a constant of three; do not mix score conventions. '
        'This is an adapted three-item measure, not the standard full PSS instrument. '
        'No response scores or outcome distributions were calculated.\n',
    ]
    if split['available']:
        ec, gc = report['ema_clocks_development'], report['garmin_clocks_development']
        sections += [
            table(report['pss_item_validity_development'], ['category', 'development_rows',
                  'complete_valid_items', 'rows_missing_any_item', 'nonmissing_invalid_cells']) + '\n',
            '## Clocks — development only\n',
            f'All {ec["rows"]:,} development EMA rows remain in history denominators. '
            f'Unparsed actual/scheduled clocks: {ec["actual_unparsed"]}/{ec["scheduled_unparsed"]}. '
            f'Scheduled local time minus the row offset agrees with `instance_id` for '
            f'{ec["scheduled_epoch_agreement_rows"]:,} rows. Offsets are minutes local-minus-UTC: '
            f'{ec["offsets_minutes"]}. Unix timestamps receive no additional timezone shift.\n',
            f'Actual-start precision: {ec["precision_counts"]}. The median start delay is '
            f'{ec["delay_median_seconds"]:g} seconds; {ec["delays_above_30min"]:,} delays exceed 30 minutes '
            f'and {ec["delays_above_2h"]} exceed two hours. Scheduled time is not actual start. '
            'No dedicated submission/end clock exists in these CSV schemas. Survey-completed flags '
            'and location timestamps do not establish submission time.\n',
            'The prior corrected-v2 implementation (`window_quality.py:boundary_epoch`) subtracts '
            '**60 seconds for minute precision and zero seconds for second precision**. This policy '
            'is preserved provisionally. Intervals are left-inclusive and right-exclusive: '
            '`[cutoff - lookback, cutoff)`. Rounding versus truncation remains undocumented.\n',
            f'Garmin development rows: {gc["rows"]:,}; invalid epochs/text clocks: '
            f'{gc["invalid_epoch"]}/{gc["invalid_text_clock"]}. '
            f'{gc["residual_above_1_1_seconds"]} clock pairs differ by more than 1.1 seconds after '
            f'allowing an inferred minute offset; {gc["offsets_outside_ema"]} imply offsets absent '
            'from EMA. These are overlapping flags, not independent exclusion counts. '
            'Naive Garmin text is predominantly local time. Offset inference is a diagnostic, '
            'not a clock repair or proof of acquisition timing.\n',
            f'Development Garmin UTC date span: {gc["start_date_utc"]}–{gc["end_date_utc"]}; '
            f'development EMA span: {ec["start_date_utc"]}–{ec["end_date_utc"]}. '
            'No study-date exclusions are silently applied. `ts` has fractional-second precision, '
            '`dte_tme` has mixed second/fractional precision. The exports do not expose a separate '
            'upload/acquisition clock; physiological measurement versus buffered receipt time remains unresolved.\n',
            '## Sensor values and recording patterns — development only\n',
            table(report['sensors_development'], ['sensor', 'participants', 'rows',
                  'missing_or_nonnumeric_values', 'zero_values', 'min', 'max',
                  'outside_provisional_range', 'duplicate_timestamp_excess']) + '\n',
            'Range flags are diagnostic sensitivities only: HR 30–240 bpm, IBI 250–2000 ms, '
            'PulseOx 70–100%, stress 0–100, counters ≥0. Units are plausible interpretations of '
            'the schema and values, not verified exporter specifications. HR/PulseOx zeros are '
            'unusable as physiological measurements, but remain preserved. Stress zero may be '
            'a valid resting score or an exporter sentinel; no positive-only filtering is applied.\n',
            table(report['sensors_development'], ['sensor', 'positive_distinct_interval_p05_seconds',
                  'positive_distinct_interval_median_seconds', 'positive_distinct_interval_p95_seconds',
                  'gaps_above_60s', 'gaps_above_3600s', 'run_gap_threshold_seconds',
                  'continuity_run_median_seconds', 'continuity_run_p95_seconds',
                  'longest_continuity_run_seconds', 'runs_at_least_5min']) + '\n',
            'Cadence quantiles pool positive distinct-timestamp differences within participant/sensor, '
            'rounded to milliseconds; no intervals cross participants. Continuous runs split at gaps '
            '>2 seconds for HR/IBI and >60 seconds for other streams. These descriptive thresholds '
            'are descriptive diagnostics, not validated eligibility rules. Singleton runs have duration zero. '
            'Calorie intervals slightly above 60 seconds (observed median 60.065 seconds) can '
            'split runs under the strict >60-second rule even at an approximately minute cadence. '
            'The HR result of no runs lasting five minutes under the 2-second rule does not '
            'establish absence of usable HR data; HR usability requires a separately justified, '
            'sensor-specific policy. No observations are removed by these run diagnostics. '
            'Even a long run does not establish clean NN intervals; a 24-hour first-to-last span '
            'does not imply 24 hours of continuous data.\n',
            table([r for r in report['sensors_development'] if r['sensor'] in
                   ('steps', 'calories', 'floorsClimbed', 'intensityMinutes')],
                  ['sensor', 'within_day_pairs', 'within_day_decreases',
                   'date_transition_pairs', 'date_transition_decreases']) + '\n',
            'Counter comparisons use adjacent sorted records and exported local dates. Decreases '
            'at date changes support cumulative daily counters; within-day decreases may be resets '
            'or inconsistencies. Tied timestamps remain a caveat. Do not sum raw counter readings. '
            'Exact-row duplicates, timestamp/value duplicates, conflicting timestamps, export-order '
            'reversals, more gap thresholds and an IBI clock-consistency diagnostic are in `inventory.json`.\n',
            '## Available preceding history — development only\n',
            'All development EMA responses are the denominator, including participants without Garmin. '
            'Counts use recorded timestamps without range cleaning, interpolation or imputation. '
            '“Any” requires one observation. The stronger column requires every internal and boundary '
            'gap ≤60 seconds; it is a diagnostic of recording density, not complete or usable coverage. '
            'IBI additionally has a ≤2-second version in the JSON. Missing history is never a zero value.\n',
            table(report['history_development'], ['sensor', 'lookback_minutes',
                  'denominator_development_ema', 'responses_with_any_observation',
                  'responses_with_all_edge_and_internal_gaps_le_60s']) + '\n',
        ]
    else:
        sections.append('Frozen assignments are unavailable; only structural inspection was performed. '
                        'No exploratory clock/value/history results are produced and no replacement split is drawn.\n')
    sections += [
        '## Integrity and limits\n',
        f'Source hashes and modification times were rechecked for '
        f'{report["verification"]["data_source_files_unchanged"]} local input files and '
        f'{report["verification"]["reference_project_files_unchanged"]:,} reference-project files '
        '(excluding environments, Git internals and caches). All matched. Archive CRC checking is '
        'performed by complete ZIP reads. The private manifest retains identifying paths; public '
        'reports do not. No participant-level inspection tables are written; the original frozen '
        'assignment file is copied unchanged into ignored private storage.\n',
        'Open issues: acquisition versus upload timing, the origin of clock anomalies, actual-start '
        'rounding, exporter-specific units/sentinels, keyless-source identity provenance, counter '
        'reset semantics, and validated IBI artifact/continuity eligibility. Some can be characterized '
        'further in development data; exporter/protocol semantics require external documentation. '
        'See `feature_pipeline_plan.md` for the next concrete step and the package review.\n',
    ]
    (out / 'inventory_report.md').write_text('\n'.join(sections))
    (out / 'inventory.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'data_dictionary.json').write_text(json.dumps(dictionary, indent=2, allow_nan=False) + '\n')
    intro = '''# Data dictionary

Every CSV cell is read as text with pandas default missing-value coercion disabled.
Original source cells are never rewritten. Blank/whitespace and literal NA/N/A/NaN/nan/NULL/null
are candidate missing markers in structured fields; free-text fields retain nonblank tokens as text.
-99/-999 are explicitly screened as possible sentinels, not
asserted to be valid coding for every questionnaire. The JSON profiles count observed blanks,
candidate missing cells and numeric/integer cells by field in development participants only.
Zeros remain values; absence of a sensor record is separate from a blank cell.

Garmin sensor categories and provisional units:

| data_type | Meaning | Provisional units | Initial role |
| --- | --- | --- | --- |
| ibi | Beat interval stream, not a raw ECG/PPG waveform | milliseconds | HRV after timing/quality validation |
| hr | Heart rate | beats/minute | candidate HR summaries |
| pulseOx | Blood oxygen saturation | percent | candidate, conditional on usable coverage |
| stress | Proprietary Garmin stress | score approximately 0–100 | exploratory only, excluded |
| steps | Predominantly cumulative daily steps | steps | validated increments later |
| calories | Predominantly cumulative daily energy | likely kcal; exporter unverified | validated increments later |
| floorsClimbed | Floor counter | floors | optional after reset validation |
| intensityMinutes | Intensity counter | minutes; weighting unverified | optional after reset validation |

No respiratory waveform or raw ECG/PPG stream is present. Embedded EMA device fields have no
verified preceding-measurement anchor and are excluded from the initial wearable predictors.
Free text and location are sensitive and excluded. Unknown survey codes are not inferred from
field names. Morning/Midday/Endday categories are identified by schema, not participant filenames.

All source fields follow. Field names are public schema metadata; no real values or IDs appear.

'''
    body = []
    for category in ['Morning', 'Midday', 'Endday', 'Garmin']:
        body += [f'## {category}\n', table([d for d in dictionary if d['table'] == category],
                 ['field', 'semantic_category', 'interpretation', 'units',
                  'observed_development_type', 'development_missing_cells']) + '\n']
    (out / 'data_dictionary.md').write_text(intro + '\n'.join(body))
