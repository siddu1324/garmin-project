"""Recompute inspection from raw archives. Only aggregate findings leave memory.

All-cohort access: schemas, identity, source accounting, split verification only.
Development-only access: clocks, values, item validity, gaps and history diagnostics.
"""
import argparse
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

from .privacy import AuditError, git_guard, scan_public, sha256, tree_snapshot
from .sources import discover

SENSORS = ('hr', 'ibi', 'stress', 'pulseOx', 'steps', 'calories',
           'floorsClimbed', 'intensityMinutes')
LOOKBACKS = (5, 15, 30, 60, 120, 360, 720, 1440)
KEYS = ('mbl_cod', 'rsp_id')
PSS = ('PSS_NERVOUS', 'PSS_COPE', 'PSS_CONTROL')
MISSING = {'', 'NA', 'N/A', 'NaN', 'nan', 'NULL', 'null', '-999', '-99'}


def clean(s):
    return s.astype(str).str.strip().replace({x: None for x in MISSING})


def parse_local(s):
    return pd.to_datetime(s, format='mixed', errors='coerce')


def epoch(s):
    return s.astype('int64').to_numpy(dtype=float) / 1e9


def cutoff(actual, precision):
    if precision not in ('minute', 'second'):
        raise AuditError('Unknown EMA clock precision; no default cutoff allowed.')
    return actual - (60 if precision == 'minute' else 0)


def continuity(times, threshold):
    t = np.unique(np.asarray(times, float))
    if not len(t):
        return np.array([], dtype=float)
    breaks = np.flatnonzero(np.diff(t) > threshold)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(t) - 1]
    return t[ends] - t[starts]


def history(times, boundary, minutes, gap_limit, *, sorted_unique=False):
    """Diagnostic-only; interval [boundary-lookback,boundary), no feature export."""
    times = np.asarray(times, float) if sorted_unique else np.unique(np.asarray(times, float))
    left, right = np.searchsorted(times, [boundary - minutes * 60, boundary], side='left')
    t = times[left:right]
    if not len(t):
        return False, False
    edge = np.r_[t[0] - (boundary - minutes * 60), np.diff(t), boundary - t[-1]]
    return True, bool(edge.max() <= gap_limit)


def validate_split(assignments, people):
    required = {'participant', 'partition', 'fold'}
    if not required <= set(assignments):
        raise AuditError('Frozen split schema is incomplete.')
    if assignments.participant.duplicated().any() or set(assignments.participant) != people:
        raise AuditError('Frozen split does not cover the current cohort exactly once.')
    if assignments.groupby('partition').size().to_dict() != {'development': 125, 'final_test': 32}:
        raise AuditError('Frozen split differs from the agreed participant counts.')
    d = assignments.loc[assignments.partition == 'development']
    if d.groupby('fold').size().to_dict() != {i: 25 for i in range(5)}:
        raise AuditError('Frozen development folds differ from the verified reference.')
    if not assignments.loc[assignments.partition == 'final_test', 'fold'].eq(-1).all():
        raise AuditError('Final-test fold metadata is unexpected.')
    return set(d.participant)


def resolve_pair(mbl, rsp, lookup, ambiguous, fallback=None):
    # A participant key wins only if it does not conflict with a usable respondent key.
    candidate = lookup.get(rsp) if rsp not in ambiguous else None
    if mbl and candidate and mbl != candidate:
        return None, 'conflicting'
    if mbl:
        return mbl, 'content_mbl_cod'
    if candidate:
        return candidate, 'content_rsp_id'
    if fallback:
        return fallback, 'reference_corroborated_filename'
    return None, 'unresolved'


def describe_field(name):
    if name in KEYS:
        return 'identity', 'opaque string; never numeric', 'none; exact keys only'
    if name in ('actual_start_local', 'scheduled_start_local', 'dte_tme'):
        return 'clock', 'naive local datetime text', 'local clock; offset required'
    if name == 'ts':
        return ('clock', 'numeric epoch stored as text',
                'Unix seconds; acquisition-versus-receipt semantics unconfirmed')
    if name == 'instance_id':
        return 'clock', 'numeric epoch stored as text', 'Unix seconds; instance is scheduled'
    if name == 'timezone_offset':
        return 'clock', 'integer stored as text', 'minutes local minus UTC'
    if name in PSS:
        return 'outcome item', 'integer code', '1–5 codes; displayed 0–4 labels'
    if name == 'data_type':
        return 'sensor category', 'categorical text', 'see sensor dictionary'
    if name == 'val':
        return 'sensor value', 'numeric stored as text', 'depends on data_type'
    if any(x in name.upper() for x in ['TEXT', 'THREAT_MESSAGE']):
        return 'sensitive survey text', 'text/coded content; values withheld', 'not a predictor'
    if name in ('LATITUDE', 'LONGITUDE', 'Location Timestamp', 'Location near window', 'Proximity to window'):
        return 'sensitive location', 'source text', 'not a predictor; exporter semantics unverified'
    if name in ('STRESS', 'PULSE_OX', 'HEART_RATE', 'RESPIRATION', 'BODY_BATTERY', 'STEPS',
                'CALORIES', 'FLOORS', 'INTENSITY_MINUTES', 'AVG_AMP', 'VOX_ACTV'):
        return 'embedded EMA device/context field', 'source text', 'timing/units unverified; exclude initial predictors'
    return 'survey response', 'source text / numeric codes', 'questionnaire-specific; not an initial predictor'


def field_profile(frame, category):
    result = []
    for name in frame:
        s = frame[name].astype(str).str.strip()
        kind, dtype, units = describe_field(name)
        missing = s.eq('') if kind == 'sensitive survey text' else s.isin(MISSING)
        num = pd.to_numeric(s.where(~missing), errors='coerce')
        numeric = num.dropna()
        observed = ('empty' if missing.all() else 'numeric' if num.notna().sum() == (~missing).sum()
                    else 'text/mixed')
        result.append(dict(table=category, field=name, semantic_category=kind,
                           storage_type='CSV text', interpretation=dtype, units=units,
                           observed_development_type=observed, development_cells=len(s),
                           development_missing_cells=int(missing.sum()),
                           development_blank_cells=int(s.eq('').sum()),
                           development_numeric_cells=int(num.notna().sum()),
                           development_integer_cells=int((numeric % 1 == 0).sum())))
    return result


def load_ema(sources):
    tables, inventory, raw_tables = [], [], []
    for source in sources:
        with source.open() as f:
            raw = pd.read_csv(f, dtype=str, keep_default_na=False)
        category = 'Morning' if 'SLEEP_SLEEPTIME' in raw else 'Endday' if 'ENDDAY_TEXT' in raw else 'Midday'
        raw_tables.append((category, raw))
        inventory.append(dict(category=category, rows=len(raw), participants=clean(raw.mbl_cod).nunique(),
                              fields=len(raw.columns)))
        t = raw[list(KEYS) + ['instance_id', 'scheduled_start_local', 'timezone_offset',
                             'actual_start_local']].copy()
        t['category'] = category
        for key in KEYS:
            t[key] = clean(t[key])
        tables.append(t)
    return pd.concat(tables, ignore_index=True), raw_tables, inventory


def ema_audit(ema, raw_tables, dev):
    t = ema.loc[ema.mbl_cod.isin(dev)].copy()
    t['precision'] = np.where(t.actual_start_local.str.contains(r'\d+:\d+:\d+'), 'second', 'minute')
    actual, scheduled = parse_local(t.actual_start_local), parse_local(t.scheduled_start_local)
    offset = pd.to_numeric(t.timezone_offset, errors='coerce')
    actual_utc, scheduled_utc = actual - pd.to_timedelta(offset, unit='min'), scheduled - pd.to_timedelta(offset, unit='min')
    t['actual_epoch'] = epoch(actual_utc)
    invalid = actual_utc.isna() | offset.isna()
    t.loc[invalid, 'actual_epoch'] = np.nan
    t['cutoff'] = [cutoff(a, p) for a, p in zip(t.actual_epoch, t.precision)]
    residual = epoch(scheduled_utc) - pd.to_numeric(t.instance_id, errors='coerce')
    delay = (actual - scheduled).dt.total_seconds()
    clocks = dict(scope='development only', rows=len(t), actual_unparsed=int(actual.isna().sum()),
                  scheduled_unparsed=int(scheduled.isna().sum()), offset_missing=int(offset.isna().sum()),
                  offsets_minutes=sorted(offset.dropna().unique().tolist()),
                  scheduled_epoch_agreement_rows=int((abs(residual) < .001).sum()),
                  actual_before_scheduled=int((delay < 0).sum()), delays_above_30min=int((delay > 1800).sum()),
                  delays_above_2h=int((delay > 7200).sum()), delay_median_seconds=float(delay.median()),
                  precision_counts=t.precision.value_counts().to_dict(),
                  start_date_utc=str(actual_utc.min().date()), end_date_utc=str(actual_utc.max().date()),
                  submission_or_end_columns=[c for c in ema if re.search(r'submit|completion_time|end_time', c, re.I)])
    dictionary, item_validity = [], []
    for category, raw in raw_tables:
        d = raw.loc[clean(raw.mbl_cod).isin(dev)]
        dictionary += field_profile(d, category)
        items = d[list(PSS)].apply(pd.to_numeric, errors='coerce')
        valid = items.notna() & items.ge(1) & items.le(5) & items.mod(1).eq(0)
        item_validity.append(dict(category=category, development_rows=len(d),
                                  complete_valid_items=int(valid.all(axis=1).sum()),
                                  rows_missing_any_item=int(items.isna().any(axis=1).sum()),
                                  nonmissing_invalid_cells=int((items.notna() & ~valid).sum().sum()),
                                  blank_item_cells=int(d[list(PSS)].eq('').sum().sum())))
    return t, clocks, dictionary, item_validity


def questionnaire_audit(sources):
    matched = 0
    for source in sources:
        with source.open() as f:
            text = '\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(f.read())).pages)
        if all(field in text for field in PSS):
            if '0 = Not at all' in text and '4 = Extremely' in text:
                matched += 1
    return dict(pdf_documents=len(sources), documents_with_three_pss_items_and_0_to_4_labels=matched,
                csv_codes='1–5', display_labels='0–4',
                proposed_formula='PSS_NERVOUS + (6 - PSS_COPE) + (6 - PSS_CONTROL)',
                range='3–15', score_calculated=False,
                note='Three adapted items; do not equate with a standard full PSS instrument. Complete integer-coded items required.')


def corroborate_fallback(reference, participant, content_hash):
    folder = reference / 'data_interim/v2/garmin_normalized' / ('person_' + hashlib.sha256(participant.encode()).hexdigest()[:24])
    for p in folder.rglob('*.parquet'):
        f = pd.read_parquet(p, columns=['participant', 'source_content_sha256'])
        matched = f.loc[f.source_content_sha256 == content_hash, 'participant']
        if len(matched) and set(matched) == {participant}:
            return True
    return False


def source_identity_scan(sources, ema, reference):
    pairs = set(map(tuple, ema[['rsp_id', 'mbl_cod']].dropna().to_numpy()))
    sources_meta, all_schemas, private_keys = [], Counter(), set(ema.mbl_cod.dropna()) | set(ema.rsp_id.dropna())
    for index, source in enumerate(sources):
        key_counts = Counter()
        rows = 0
        with source.open() as f:
            reader = pd.read_csv(f, dtype=str, keep_default_na=False, chunksize=250_000,
                                 usecols=lambda c: c in KEYS)
            for chunk in reader:
                # pandas cannot retain row count for a zero-column usecols result.
                if not len(chunk.columns):
                    break
                rows += len(chunk)
                for key in KEYS:
                    if key not in chunk:
                        chunk[key] = ''
                    chunk[key] = clean(chunk[key]).fillna('')
                counts = chunk.groupby(list(KEYS), dropna=False).size()
                key_counts.update({tuple(k): int(v) for k, v in counts.items()})
        with source.open() as f:
            columns = tuple(pd.read_csv(f, nrows=0).columns)
        if not {'ts', 'dte_tme', 'data_type', 'val'} <= set(columns):
            raise AuditError('Unsupported Garmin schema.')
        if not set(KEYS) & set(columns):
            with source.open() as f:
                rows = sum(len(c) for c in pd.read_csv(f, usecols=['ts'], chunksize=250_000))
            key_counts[('', '')] = rows
        all_schemas[columns] += 1
        for (m, r), n in key_counts.items():
            private_keys.update(x for x in [m, r] if x)
            if m and r:
                pairs.add((r, m))
        sources_meta.append(dict(source=source, rows=rows, key_counts=key_counts,
                                 content_hash=source.content_hash(), columns=columns))
        if (index + 1) % 20 == 0:
            print(f'Identity/content audit: {index + 1}/{len(sources)} sources', flush=True)
    candidates = defaultdict(set)
    for r, p in pairs:
        candidates[r].add(p)
    ambiguous = {r for r, ps in candidates.items() if len(ps) > 1}
    lookup = {r: next(iter(ps)) for r, ps in candidates.items() if len(ps) == 1}
    verified = reference / 'data_interim/v2/participant_crosswalk.parquet'
    reference_pairs = set()
    if verified.exists():
        c = pd.read_parquet(verified)
        reference_pairs = set(map(tuple, c[['respondent', 'participant']].to_numpy()))
    seen, methods, people, unknown, missing = set(), Counter(), set(), set(), Counter()
    for meta in sources_meta:
        source = meta['source']
        tokens = source.exact_tokens() & set(ema.mbl_cod.dropna())
        fallback = next(iter(tokens)) if len(tokens) == 1 else None
        has_unmapped = any(not m and (r not in lookup) for m, r in meta['key_counts'])
        if has_unmapped and fallback:
            if not corroborate_fallback(reference, fallback, meta['content_hash']):
                fallback = None
        source_people, unresolved = set(), 0
        for (m, r), n in meta['key_counts'].items():
            participant, method = resolve_pair(m, r, lookup, ambiguous, fallback)
            methods[method] += n
            missing['rows_missing_mbl_cod'] += n if not m else 0
            missing['rows_missing_rsp_id'] += n if not r else 0
            if participant:
                source_people.add(participant)
                if participant not in set(ema.mbl_cod):
                    unknown.add(participant)
            else:
                unresolved += n
        meta.update(participants=source_people, unresolved_rows=unresolved,
                    duplicate_export=meta['content_hash'] in seen, fallback=fallback)
        seen.add(meta['content_hash'])
        people.update(source_people)
    return sources_meta, dict(schemas=[dict(fields=list(k), files=v) for k, v in all_schemas.items()],
                             physical_rows=sum(m['rows'] for m in sources_meta),
                             logical_rows=sum(m['rows'] for m in sources_meta if not m['duplicate_export']),
                             duplicate_export_files=sum(m['duplicate_export'] for m in sources_meta),
                             distinct_mapped_participants=len(people), participants_absent_from_ema=len(unknown),
                             ema_participants_without_garmin=len(set(ema.mbl_cod) - people),
                             ambiguous_respondent_keys=len(ambiguous), mapping_methods_rows=dict(methods),
                             missing_key_rows=dict(missing),
                             current_crosswalk_pairs=len(pairs), reference_crosswalk_pairs=len(reference_pairs),
                             pairs_only_current=len(pairs - reference_pairs), pairs_only_reference=len(reference_pairs - pairs),
                             files_with_multiple_participants=sum(len(m['participants']) > 1 for m in sources_meta)), private_keys


def hist_quantile(hist, q):
    if not hist:
        return None
    ordered = sorted(hist.items())
    target = q * (sum(hist.values()) - 1)
    cumulative = 0
    for value, count in ordered:
        cumulative += count
        if cumulative > target:
            return float(value)
    return float(ordered[-1][0])


def add_hist(hist, values, decimals=3):
    values, counts = np.unique(np.round(values, decimals), return_counts=True)
    hist.update(dict(zip(values.tolist(), counts.tolist())))


def sensor_audit(sources_meta, development, ema_dev):
    counts = {s: Counter() for s in SENSORS}
    intervals = {s: Counter() for s in SENSORS}
    runs = {s: Counter() for s in SENSORS}
    people = {s: set() for s in SENSORS}
    limits = {'hr': (30, 240), 'ibi': (250, 2000), 'stress': (0, 100), 'pulseOx': (70, 100)}
    offsets, clock_totals, date_bounds = Counter(), Counter(), []
    coverage = {(s, w): Counter() for s in SENSORS for w in LOOKBACKS}
    ibi_clock_agreement = Counter()
    audited_people = set()
    files_by_person = defaultdict(list)
    for meta in sources_meta:
        if meta['unresolved_rows'] or len(meta['participants']) != 1:
            raise AuditError('Unresolved or mixed source identities require a separate structural audit.')
        person = next(iter(meta['participants']))
        if person in development and not meta['duplicate_export']:
            files_by_person[person].append(meta)
    for index, (person, metas) in enumerate(files_by_person.items()):
        frames = []
        for meta in metas:
            with meta['source'].open() as f:
                frames.append(pd.read_csv(f, dtype=str, keep_default_na=False))
        raw = pd.concat(frames, ignore_index=True)
        if not set(raw.data_type) <= set(SENSORS):
            raise AuditError('Unknown sensor type; values withheld.')
        audited_people.add(person)
        ts = pd.to_numeric(raw.ts, errors='coerce')
        good_epoch = ts.between(946684800, 2082758400)  # Explicit 2000–2035 plausibility, not a study exclusion.
        local = pd.to_datetime(raw.dte_tme, format='mixed', errors='coerce')
        wall = epoch(local)
        offset = np.rint((wall - ts) / 60)
        residual = wall - ts - offset * 60
        valid_clock = good_epoch & local.notna()
        offsets.update({str(int(k)): int(v) for k, v in pd.Series(offset[valid_clock]).value_counts().items()})
        clock_totals.update(rows=len(raw), invalid_epoch=int((~good_epoch).sum()),
                            invalid_text_clock=int(local.isna().sum()),
                            residual_above_1_1_seconds=int((valid_clock & (abs(residual) > 1.1)).sum()),
                            offsets_outside_ema=int((valid_clock & ~pd.Series(offset).isin([-480, -420, -360, -300])).sum()),
                            fractional_epoch_rows=int((good_epoch & ts.mod(1).ne(0)).sum()),
                            fractional_text_rows=int(raw.dte_tme.str.contains(r'\d+:\d+:\d+\.\d+').sum()))
        if good_epoch.any():
            date_bounds.extend([ts[good_epoch].min(), ts[good_epoch].max()])
        raw['_ts'], raw['_val'], raw['_good_epoch'] = ts, pd.to_numeric(raw.val, errors='coerce'), good_epoch
        anchors = ema_dev.loc[ema_dev.mbl_cod == person, 'cutoff'].dropna().to_numpy()
        for sensor, f in raw.groupby('data_type', sort=False):
            c = counts[sensor]
            people[sensor].add(person)
            v = f._val.to_numpy()
            valid_values = np.isfinite(v)
            low, high = limits.get(sensor, (0, np.inf))
            c.update(rows=len(f), missing_or_nonnumeric_values=int((~valid_values).sum()),
                     blank_values=int(f.val.str.strip().eq('').sum()), zero_values=int((v == 0).sum()),
                     negative_values=int((v < 0).sum()), noninteger_values=int((valid_values & (v % 1 != 0)).sum()),
                     outside_provisional_range=int((valid_values & ((v < low) | (v > high))).sum()),
                     exact_record_duplicate_excess=int(f.drop(columns=['_ts', '_val', '_good_epoch']).duplicated().sum()))
            if valid_values.any():
                c['min'] = min(c.get('min', np.inf), float(v[valid_values].min()))
                c['max'] = max(c.get('max', -np.inf), float(v[valid_values].max()))
            ordered = f.loc[f._good_epoch].sort_values('_ts', kind='stable')
            t, vals = ordered._ts.to_numpy(), ordered._val.to_numpy()
            unique = np.unique(t)
            diffs = np.diff(unique)
            c['duplicate_timestamp_excess'] += len(t) - len(unique)
            c['identical_timestamp_value_excess'] += int(ordered.duplicated(['_ts', '_val']).sum())
            distinct_pairs = ordered.drop_duplicates(['_ts', '_val'])
            c['conflicting_timestamps'] += int((distinct_pairs.groupby('_ts').size() > 1).sum())
            original_order = f.loc[f._good_epoch, '_ts'].to_numpy()
            c['backward_steps_in_export_order'] += int((np.diff(original_order) < 0).sum())
            for gap in (2, 5, 60, 300, 1800, 3600, 7200):
                c[f'gaps_above_{gap}s'] += int((diffs > gap).sum())
            add_hist(intervals[sensor], diffs)
            threshold = 2 if sensor in ('hr', 'ibi') else 60
            run_lengths = continuity(unique, threshold)
            add_hist(runs[sensor], run_lengths)
            c['runs_at_least_5min'] += int((run_lengths >= 300).sum())
            c['run_gap_threshold_seconds'] = threshold
            if sensor == 'ibi' and len(t) > 1:
                dt, rr = np.diff(t), vals[1:] / 1000
                finite = np.isfinite(rr) & (dt > 0)
                ibi_clock_agreement.update(adjacent_positive_time_pairs=int(finite.sum()),
                    time_difference_within_100ms_of_current_ibi=int((finite & (abs(dt - rr) <= .1)).sum()),
                    adjacent_value_change_above_20pct=int((abs(np.diff(vals)) > .2 * vals[:-1]).sum()))
            if sensor in ('steps', 'calories', 'floorsClimbed', 'intensityMinutes') and len(t) > 1:
                day = ordered.dte_tme.str[:10].to_numpy()
                same_day, decrease = day[1:] == day[:-1], np.diff(vals) < 0
                c.update(within_day_pairs=int(same_day.sum()), within_day_decreases=int((same_day & decrease).sum()),
                         date_transition_pairs=int((~same_day).sum()), date_transition_decreases=int((~same_day & decrease).sum()))
            for w in LOOKBACKS:
                a = coverage[sensor, w]
                for boundary in anchors:
                    present, bounded = history(unique, boundary, w, 60, sorted_unique=True)
                    a['responses_with_any_observation'] += int(present)
                    a['responses_with_all_edge_and_internal_gaps_le_60s'] += int(bounded)
                    if sensor == 'ibi':
                        _, tight = history(unique, boundary, w, 2, sorted_unique=True)
                        a['responses_with_all_edge_and_internal_gaps_le_2s'] += int(tight)
        if (index + 1) % 10 == 0:
            print(f'Development sensor audit: {index + 1}/{len(files_by_person)} participants', flush=True)
    sensor_summary = []
    for s in SENSORS:
        c = dict(counts[s])
        c.update(sensor=s, participants=len(people[s]),
                 positive_distinct_interval_p05_seconds=hist_quantile(intervals[s], .05),
                 positive_distinct_interval_median_seconds=hist_quantile(intervals[s], .5),
                 positive_distinct_interval_p95_seconds=hist_quantile(intervals[s], .95),
                 continuity_run_median_seconds=hist_quantile(runs[s], .5),
                 continuity_run_p95_seconds=hist_quantile(runs[s], .95),
                 longest_continuity_run_seconds=max(runs[s], default=None),
                 continuity_runs=sum(runs[s].values()))
        sensor_summary.append(c)
    coverage_summary = []
    for (s, w), c in coverage.items():
        coverage_summary.append(dict(sensor=s, lookback_minutes=w, denominator_development_ema=len(ema_dev),
                                     **dict(c)))
    clock_totals['inferred_offsets_minutes_counts'] = dict(offsets)
    if date_bounds:
        clock_totals['start_date_utc'] = str(pd.to_datetime(min(date_bounds), unit='s').date())
        clock_totals['end_date_utc'] = str(pd.to_datetime(max(date_bounds), unit='s').date())
    return sensor_summary, coverage_summary, dict(clock_totals), dict(ibi_clock_agreement), len(audited_people)


def json_ready(x):
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_ready(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def run(root, reference=None, data=None):
    root = Path(root).resolve()
    data = Path(data or os.environ.get('GARMIN_DATA_DIR', root / 'data')).resolve()
    reference = Path(reference or os.environ.get('GARMIN_REFERENCE_PROJECT', '/nonexistent')).resolve()
    git_guard(root)
    (root / 'private').mkdir(exist_ok=True)
    print('Recording read-only source/reference integrity baseline', flush=True)
    before_data = tree_snapshot(data)
    before_reference = tree_snapshot(reference)
    (root / 'private/integrity_baseline.json').write_text(json.dumps({'data': before_data, 'reference': before_reference}))
    ema_sources, garmin_sources, pdf_sources, archives, names = discover(data)
    ema, raw_tables, ema_inventory = load_ema(ema_sources)
    people = set(ema.mbl_cod.dropna())
    assignments_file = reference / 'data_interim/v2/v2_step13_participant_assignments.parquet'
    assignment_hash = None
    dev = set()
    if assignments_file.exists():
        assignments = pd.read_parquet(assignments_file)
        dev = validate_split(assignments, people)
        assignment_hash = sha256(assignments_file)
        # Preserve exact original bytes locally; never reconstruct or resplit.
        (root / 'private/frozen_participant_assignments.parquet').write_bytes(assignments_file.read_bytes())
    else:
        print('Frozen split unavailable: structural inspection only; no exploratory summaries', flush=True)
    structural = dict(ema_rows=len(ema), ema_participants=len(people), ema_tables=ema_inventory,
                      missing_participant_rows=int(ema.mbl_cod.isna().sum()),
                      repeated_participant_instance_excess=int(ema.duplicated(['mbl_cod', 'instance_id']).sum()),
                      repeated_participant_instance_category_excess=int(ema.duplicated(['mbl_cod', 'instance_id', 'category']).sum()),
                      ema_respondent_keys_mapping_to_multiple_people=int((ema.groupby('rsp_id').mbl_cod.nunique() > 1).sum()))
    meta, identity, private_keys = source_identity_scan(garmin_sources, ema, reference)
    structural['garmin'] = identity
    structural['garmin_csv_tables'] = len(garmin_sources)
    source_catalog = reference / 'data_interim/v2/preprocessing_current/source_catalog.parquet'
    if source_catalog.exists():
        catalog = pd.read_parquet(source_catalog)
        current_hashes = {m['content_hash'] for m in meta}
        ref_hashes = set(catalog.content_sha256)
        structural['local_garmin_content_hashes_matching_reference'] = len(current_hashes & ref_hashes)
        structural['reference_extra_physical_sources'] = len(catalog) - len(meta)
        structural['reference_duplicate_export_rows'] = int(catalog.loc[catalog.duplicate_export, 'rows'].sum())
    ema_catalog = reference / 'data_interim/v2/preprocessing_current/ema_source_catalog.json'
    if ema_catalog.exists():
        reference_ema_hashes = {x['sha256'] for x in json.loads(ema_catalog.read_text())}
        structural['local_ema_content_hashes_matching_reference'] = len(
            {source.content_hash() for source in ema_sources} & reference_ema_hashes)
    report = dict(checkpoint='inventory only; no scores/features/models',
                  scope='All-cohort structural accounting; development-only exploration',
                  archives=archives, structural=structural,
                  frozen_split=dict(available=bool(dev), development_participants=len(dev),
                                    final_test_participants=len(people) - len(dev) if dev else None,
                                    assignment_sha256=assignment_hash, original_folds_preserved=bool(dev)),
                  questionnaire=questionnaire_audit(pdf_sources))
    dictionary = []
    if dev:
        ema_dev, clocks, dictionary, items = ema_audit(ema, raw_tables, dev)
        report.update(ema_clocks_development=clocks, pss_item_validity_development=items)
        sensors, coverage, garmin_clocks, ibi_diagnostic, n = sensor_audit(meta, dev, ema_dev)
        report.update(sensors_development=sensors, history_development=coverage,
                      garmin_clocks_development=garmin_clocks, ibi_clock_diagnostic_development=ibi_diagnostic,
                      development_participants_with_garmin=n)
    else:
        # Schemas and semantic roles only, without response profiles.
        for category, raw in raw_tables:
            for field in raw:
                kind, dtype, units = describe_field(field)
                dictionary.append(dict(table=category, field=field, semantic_category=kind,
                                       storage_type='CSV text', interpretation=dtype, units=units))
    for field in ['ts', 'dte_tme', 'rsp_id', 'mbl_cod', 'data_type', 'val']:
        kind, dtype, units = describe_field(field)
        dictionary.append(dict(table='Garmin', field=field, semantic_category=kind,
                               storage_type='CSV text', interpretation=dtype, units=units))
    print('Verifying original archive and reference-project hashes', flush=True)
    if tree_snapshot(data) != before_data or tree_snapshot(reference) != before_reference:
        raise AuditError('Source/reference integrity changed during inspection; publication stopped.')
    report['verification'] = dict(data_source_files_unchanged=len(before_data),
                                  reference_project_files_unchanged=len(before_reference),
                                  final_test_exploratory_rows=0, notebook_contains_real_rows=False)
    report = json_ready(report)
    dictionary = json_ready(dictionary)
    # Save aggregate-only cache in ignored location; public renderer never receives rows or keys.
    (root / 'private/aggregate_inventory.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (root / 'private/aggregate_dictionary.json').write_text(json.dumps(dictionary, indent=2, allow_nan=False) + '\n')
    (root / 'private/privacy_tokens.json').write_text(json.dumps(dict(identifiers=sorted(private_keys), names=sorted(names))))
    from .render import publish
    publish(root, report, dictionary)
    report['verification']['public_artifacts_scanned'] = scan_public(root, private_keys, names)
    (root / 'reports/inventory.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Completed aggregate inventory; source/reference hashes unchanged', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    try:
        run(args.root, args.reference, args.data)
    except Exception as exc:
        # CSV/parser exceptions can echo identifying content. Only known safe errors surface.
        if isinstance(exc, AuditError):
            print(f'Inspection stopped: {exc}', flush=True)
        else:
            print(f'Inspection stopped ({type(exc).__name__}); source details withheld.', flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
