"""Chunked raw-preserving development normalization with separate logical views."""
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from garmin_inventory.audit import clean, resolve_pair
from garmin_inventory.privacy import AuditError, sha256
from garmin_inventory.sources import discover
from .common import (digest_object, frozen_assignments, publish_bundle, read_config,
                     verify_bundle, write_json)

RAW = ['ts', 'dte_tme', 'rsp_id', 'mbl_cod', 'data_type', 'val']


def write_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine='pyarrow', compression='zstd')


def parse_ema_clock(raw):
    """Original strings retained by caller; parse only clocks, never outcome values."""
    out = pd.DataFrame(index=raw.index)
    offsets = pd.to_numeric(raw.timezone_offset, errors='coerce')
    for name, column in [('actual', 'actual_start_local'), ('scheduled', 'scheduled_start_local')]:
        local = pd.to_datetime(raw[column], format='mixed', errors='coerce')
        out[f'n_{name}_local'] = local
        utc = local - pd.to_timedelta(offsets, unit='min')
        out[f'n_{name}_utc'] = utc.dt.tz_localize('UTC')
    precision = np.select([raw.actual_start_local.str.contains(r'\d+:\d+:\d+'),
                           raw.actual_start_local.str.contains(r'\d+:\d+')],
                          ['second', 'minute'], default='unknown')
    out['n_start_precision'] = precision
    out['n_offset_minutes'] = offsets
    out['n_invalid_clock'] = out.n_actual_utc.isna() | offsets.isna() | (precision == 'unknown')
    embargo = np.where(precision == 'minute', 60, 0)
    out['n_cutoff_utc'] = (out.n_actual_utc - pd.to_timedelta(embargo, unit='s')).mask(out.n_invalid_clock)
    out['n_cutoff_epoch'] = out.n_cutoff_utc.astype('int64') / 1e9
    out.loc[out.n_invalid_clock, 'n_cutoff_epoch'] = np.nan
    scheduled_epoch = out.n_scheduled_utc.astype('int64') / 1e9
    out['n_scheduled_epoch_residual_s'] = scheduled_epoch - pd.to_numeric(raw.instance_id, errors='coerce')
    out['n_scheduled_clock_discrepancy'] = out.n_scheduled_epoch_residual_s.abs().gt(.001) | out.n_scheduled_utc.isna()
    out['n_actual_before_scheduled'] = out.n_actual_utc < out.n_scheduled_utc
    return out


def annotate_duplicates(frame):
    """No averaging; deduplication compares all raw strings plus original source schema."""
    f = frame.sort_values(['n_source_id', 'n_source_row'], kind='stable').reset_index(drop=True).copy()
    keys = RAW + ['n_source_columns']
    f['q_exact_record_duplicate'] = f.duplicated(keys, keep=False)
    f['q_exact_record_excess'] = f.duplicated(keys, keep='first')
    f['n_logical_keep'] = ~f.q_exact_record_excess
    f['q_physical_repeated_timestamp'] = f.duplicated(['n_epoch_s'], keep=False) & f.n_epoch_s.notna()
    f['q_physical_repeated_timestamp_value'] = f.duplicated(['n_epoch_s', 'n_value'], keep=False) & f.n_epoch_s.notna()
    logical = f.loc[f.n_logical_keep]
    repeated = logical.loc[logical.duplicated('n_epoch_s', keep=False), 'n_epoch_s'].dropna().unique()
    by_time = logical.groupby('n_epoch_s').n_value.nunique(dropna=False)
    conflict = by_time.index[by_time > 1]
    f['q_logical_repeated_timestamp'] = f.n_epoch_s.isin(repeated)
    f['q_conflicting_timestamp_value'] = f.n_epoch_s.isin(conflict)
    f['q_export_order_backward'] = f.groupby('n_source_id', sort=False).n_epoch_s.diff().lt(0)
    return f


def reference_identity_registry(reference):
    """Read identity-only Parquet footer stats (fallback identity columns), including no outcomes."""
    registry = defaultdict(set)
    base = reference / 'data_interim/v2/garmin_normalized'
    for p in sorted(base.rglob('*.parquet')):
        pf = pq.ParquetFile(p)
        schema = pf.schema_arrow.names
        pi, hi = schema.index('participant'), schema.index('source_content_sha256')
        for i in range(pf.num_row_groups):
            group = pf.metadata.row_group(i)
            ps, hs = group.column(pi).statistics, group.column(hi).statistics
            if ps and hs and ps.has_min_max and hs.has_min_max and ps.min == ps.max and hs.min == hs.max:
                registry[hs.min].add(ps.min)
            else:
                ids = pf.read_row_group(i, columns=['participant', 'source_content_sha256']).to_pandas().drop_duplicates()
                for person, content in ids.itertuples(index=False, name=None):
                    registry[content].add(person)
    if not registry or any(len(v) != 1 for v in registry.values()):
        raise AuditError('Reference content-to-participant identity is not unique.')
    return {k: next(iter(v)) for k, v in registry.items()}


def normalize_ema(sources, dev, fold, build, archive_hashes):
    frames, catalog = [], []
    for source_number, source in enumerate(sources):
        with source.open() as stream:
            raw = pd.read_csv(stream, dtype=str, keep_default_na=False)
        # Gate on identity immediately; final-test survey fields never enter any transformation/output.
        ids = clean(raw.mbl_cod)
        d = raw.loc[ids.isin(dev)].copy()
        row_number = d.index.to_numpy() + 1
        d.reset_index(drop=True, inplace=True)
        columns = list(d.columns)
        category = 'Morning' if 'SLEEP_SLEEPTIME' in columns else 'Endday' if 'ENDDAY_TEXT' in columns else 'Midday'
        content_hash = source.content_hash()
        d['n_source_id'] = f'ema_{source_number:02}'
        d['n_source_row'] = row_number
        d['n_source_columns'] = json.dumps(columns, separators=(',', ':'))
        d['n_source_sha256'] = content_hash
        d['n_archive_sha256'] = archive_hashes[str(source.archive)]
        d['n_participant'] = clean(d.mbl_cod)
        d['n_mapping_method'] = 'exact_ema_mbl_cod'
        d['n_partition'] = 'development'
        d['n_fold'] = d.n_participant.map(fold).astype('int16')
        d['n_survey_category'] = category
        d['n_ema_id'] = [f'e{source_number:02}_{r:06}' for r in row_number]
        d = pd.concat([d, parse_ema_clock(d)], axis=1)
        d['q_exact_record_duplicate'] = d.duplicated(columns, keep=False)
        catalog.append(dict(source_id=f'ema_{source_number:02}', columns=columns, content_sha256=content_hash,
                            archive_sha256=archive_hashes[str(source.archive)], development_rows=len(d)))
        frames.append(d)
    ema = pd.concat(frames, ignore_index=True)
    ema['q_event_key_collision'] = ema.duplicated(['n_participant', 'instance_id', 'n_survey_category'], keep=False)
    if set(ema.n_participant) != dev or ema.n_ema_id.duplicated().any():
        raise AuditError('Development EMA accounting or event identity failed.')
    write_parquet(ema, build / 'ema/development_ema.parquet')
    restored = pd.read_parquet(build / 'ema/development_ema.parquet')
    if not restored.equals(ema):
        raise AuditError('EMA Parquet round-trip did not preserve values.')
    return ema, catalog


def normalize_chunk(raw, source, person, fold, lookup, ambiguous, start_row, config):
    if fold not in range(5):
        raise AuditError('Normalization accepts development folds 0–4 only.')
    columns = list(raw.columns)
    f = raw.copy()
    for name in RAW:
        if name not in f:
            f[name] = None
    mbl, rsp = clean(f.mbl_cod).fillna(''), clean(f.rsp_id).fillna('')
    # Missing columns are null, not the literal spelling produced by astype(str).
    if 'mbl_cod' not in columns:
        mbl[:] = ''
    if 'rsp_id' not in columns:
        rsp[:] = ''
    methods = {}
    for m, r in set(zip(mbl, rsp)):
        resolved, method = resolve_pair(m, r, lookup, ambiguous, fallback=person)
        if resolved != person:
            raise AuditError('Raw row keys conflict with verified source identity; details withheld.')
        methods[m, r] = 'reference_content_hash' if method == 'reference_corroborated_filename' else method
    f['n_mapping_method'] = [methods[m, r] for m, r in zip(mbl, rsp)]
    f['n_participant'], f['n_partition'], f['n_fold'] = person, 'development', np.int16(fold)
    f['n_source_id'], f['n_source_sha256'] = source['source_id'], source['content_sha256']
    f['n_archive_sha256'] = source['archive_sha256']
    f['n_source_columns'] = json.dumps(columns, separators=(',', ':'))
    f['n_source_row'] = np.arange(start_row, start_row + len(f), dtype=np.int64)
    f['n_epoch_s'] = pd.to_numeric(f.ts, errors='coerce')
    f['q_invalid_epoch'] = ~f.n_epoch_s.between(946684800, 2082758400)
    f['n_utc'] = pd.to_datetime(f.n_epoch_s.where(~f.q_invalid_epoch), unit='s', utc=True, errors='coerce')
    f['n_text_local'] = pd.to_datetime(f.dte_tme, format='mixed', errors='coerce')
    f['q_invalid_text_clock'] = f.n_text_local.isna()
    f['n_inferred_offset_minutes'] = np.rint((f.n_text_local.astype('int64') / 1e9 - f.n_epoch_s) / 60)
    f['n_clock_residual_s'] = f.n_text_local.astype('int64') / 1e9 - f.n_epoch_s - 60 * f.n_inferred_offset_minutes
    f['q_clock_discrepancy'] = f.n_clock_residual_s.abs() > config['clock_residual_tolerance_seconds']
    f['q_unusual_offset'] = ~f.n_inferred_offset_minutes.isin(config['allowed_offsets_minutes'])
    f['n_value'] = pd.to_numeric(f.val, errors='coerce')
    f['q_invalid_value'] = ~np.isfinite(f.n_value)
    f['q_zero_value'] = f.n_value.eq(0)
    f['q_negative_value'] = f.n_value.lt(0)
    f['q_normal_beat_unverified'] = f.data_type.eq('ibi')
    return f


def normalize(root, reference, data):
    config = read_config(root)
    assignments, dev = frozen_assignments(root, reference)
    fold = assignments.set_index('participant').fold.to_dict()
    crosswalk_path = reference / 'data_interim/v2/participant_crosswalk.parquet'
    crosswalk = pd.read_parquet(crosswalk_path)
    pairs = crosswalk.groupby('respondent').participant.agg(set)
    ambiguous = {r for r, ps in pairs.items() if len(ps) > 1}
    lookup = {r: next(iter(ps)) for r, ps in pairs.items() if len(ps) == 1}
    sources, sensors, _, _, _ = discover(data)
    archive_hashes = {str(p): sha256(p) for p in sorted(data.rglob('*.zip'))}
    inputs = dict(archives=sorted(archive_hashes.values()), crosswalk_sha256=sha256(crosswalk_path),
                  assignment_sha256=sha256(root / 'private/frozen_participant_assignments.parquet'),
                  code_sha256={name: sha256(Path(__file__).parent / name) for name in ['normalize.py', 'common.py']},
                  reused_code_sha256={name: sha256(root / 'src/garmin_inventory' / name) for name in ['audit.py', 'sources.py']},
                  environment_sha256=sha256(root / 'uv.lock'),
                  parsing_config={k: config[k] for k in ['chunk_rows', 'max_partition_rows', 'clock_residual_tolerance_seconds', 'allowed_offsets_minutes']})
    version = digest_object(inputs)[:16]
    parent = root / 'private/preprocessing/normalized'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / version
    if destination.exists():
        return destination, verify_bundle(destination)
    build = Path(tempfile.mkdtemp(prefix=f'.building_{version}_', dir=parent))
    registry = reference_identity_registry(reference)
    ema, ema_catalog = normalize_ema(sources, dev, fold, build, archive_hashes)
    source_catalog, chunks_by_person, counts = [], defaultdict(list), Counter()
    people_order = {p: f'person_{i:03}' for i, p in enumerate(sorted(dev))}
    content_seen = set()
    for i, source in enumerate(sensors):
        content = source.content_hash()
        if content not in registry:
            raise AuditError('Raw CSV content hash is absent from verified identity provenance.')
        person = registry[content]
        if person not in set(assignments.participant):
            raise AuditError('Source participant absent from frozen assignments.')
        if person not in dev:
            counts['sealed_final_test_sources'] += 1
            continue  # No final-test sensor CSV parsing or exploration.
        source_meta = dict(source_id=f'garmin_{i:03}', content_sha256=content,
                           archive_sha256=archive_hashes[str(source.archive)],
                           participant=person, fold=fold[person], duplicate_export=content in content_seen)
        content_seen.add(content)
        rows = 0
        with source.open() as stream:
            for j, raw in enumerate(pd.read_csv(stream, dtype=str, keep_default_na=False, chunksize=config['chunk_rows'])):
                f = normalize_chunk(raw, source_meta, person, fold[person], lookup, ambiguous, rows + 1, config)
                f['q_duplicate_export'] = source_meta['duplicate_export']
                chunk_path = build / '_chunks' / people_order[person] / f's{i:03}_{j:04}.parquet'
                write_parquet(f, chunk_path)
                restored = pd.read_parquet(chunk_path, columns=list(raw.columns))
                if not raw.reset_index(drop=True).equals(restored):
                    raise AuditError('Raw sensor values changed in Parquet round-trip.')
                chunks_by_person[person].append(chunk_path)
                rows += len(raw)
        source_meta['rows'] = rows
        source_catalog.append(source_meta)
        counts['physical_sensor_rows'] += rows
        if (i + 1) % 20 == 0:
            print(f'Raw-preserving source scan: {i + 1}/{len(sensors)}', flush=True)
    sensor_counts, mapping_counts, person_catalog = Counter(), Counter(), []
    for i, person in enumerate(sorted(dev)):
        person_label = people_order[person]
        paths = chunks_by_person[person]
        person_catalog.append(dict(participant=person, directory=person_label, fold=fold[person], has_garmin=bool(paths)))
        if not paths:
            continue
        # Partition by participant/sensor; no whole-cohort frame is materialized.
        for sensor in ['hr', 'ibi', 'stress', 'pulseOx', 'steps', 'calories', 'floorsClimbed', 'intensityMinutes']:
            import pyarrow.dataset as ds
            dataset = ds.dataset([str(p) for p in paths], format='parquet')
            n = dataset.count_rows(filter=ds.field('data_type') == sensor)
            if not n:
                continue
            if n > config['max_partition_rows']:
                raise AuditError('Participant/sensor exceeds configured partition memory budget.')
            frame = dataset.to_table(filter=ds.field('data_type') == sensor).to_pandas()
            physical = annotate_duplicates(frame)
            physical['n_physical_row'] = np.arange(len(physical), dtype=np.int64)
            physical_path = build / 'physical' / person_label / f'{sensor}.parquet'
            write_parquet(physical, physical_path)
            restored = pd.read_parquet(physical_path, columns=RAW + ['n_source_id', 'n_source_row'])
            if not restored.equals(physical[RAW + ['n_source_id', 'n_source_row']]):
                raise AuditError('Final physical partition failed raw-value/provenance round-trip.')
            keep_columns = [c for c in physical if c.startswith(('n_', 'q_'))] + ['data_type']
            logical = physical.loc[physical.n_logical_keep, keep_columns].sort_values(
                ['n_epoch_s', 'n_source_id', 'n_source_row'], kind='stable').reset_index(drop=True)
            logical['n_previous_timestamp_gap_s'] = logical.n_epoch_s.diff()
            logical['n_logical_row'] = np.arange(len(logical), dtype=np.int64)
            write_parquet(logical, build / 'logical' / person_label / f'{sensor}.parquet')
            counts['logical_sensor_rows'] += len(logical)
            counts['exact_duplicate_excess_rows'] += int(physical.q_exact_record_excess.sum())
            counts['conflicting_physical_rows'] += int(physical.q_conflicting_timestamp_value.sum())
            sensor_counts[sensor] += len(physical)
            mapping_counts.update(physical.n_mapping_method.value_counts().to_dict())
        if (i + 1) % 10 == 0:
            print(f'Normalized development partitions: {i + 1}/{len(dev)}', flush=True)
    # All persisted physical rows account for source rows exactly once.
    if sum(sensor_counts.values()) != counts['physical_sensor_rows']:
        raise AuditError('Sensor schema or row accounting mismatch; no rows silently dropped.')
    if counts['physical_sensor_rows'] != counts['logical_sensor_rows'] + counts['exact_duplicate_excess_rows']:
        raise AuditError('Physical/logical accounting mismatch.')
    write_json(build / 'source_catalog.json', dict(ema=ema_catalog, garmin=source_catalog))
    write_json(build / 'participant_directories.json', person_catalog)
    # Own temporary chunks are redundant after exact raw/provenance round-trip verification.
    import shutil
    shutil.rmtree(build / '_chunks')
    metadata = dict(version=version, inputs=inputs, counts=dict(counts), sensor_rows=dict(sensor_counts),
                    mapping_rows=dict(mapping_counts), ema_rows=len(ema), development_participants=len(dev),
                    development_participants_with_garmin=len([p for p in person_catalog if p['has_garmin']]),
                    raw_roundtrip_verified=True, final_test_rows_written=0,
                    all_ema_folds_preserved=bool(ema.n_fold.eq(ema.n_participant.map(fold)).all()))
    manifest = publish_bundle(build, destination, metadata)
    print('Immutable development normalization complete', flush=True)
    return destination, manifest
