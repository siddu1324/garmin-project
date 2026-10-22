"""Raw-preserving outcome parsing, questionnaire evidence and strict identity linkage."""
import io
import json
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader

from garmin_inventory.audit import MISSING, PSS
from garmin_inventory.privacy import AuditError
from garmin_inventory.sources import discover

META = ['n_ema_id', 'n_participant', 'n_partition', 'n_fold', 'n_survey_category',
        'n_cutoff_epoch', 'n_start_precision', 'n_invalid_clock']


def questionnaire_evidence(root):
    """Read blank survey PDFs, not participant response files."""
    evidence = []
    for source in discover(root / 'data')[2]:
        with source.open() as stream:
            pages = [p.extract_text() or '' for p in PdfReader(io.BytesIO(stream.read())).pages]
        text = '\n'.join(pages)
        if not all(field in text for field in PSS):
            continue
        suffix = '_1' if 'PSS_NERVOUS_1' in text else '_4' if 'PSS_NERVOUS_4' in text else ''
        category = {'_1': 'Morning', '_4': 'Endday', '': 'Midday'}[suffix]
        fields = []
        for name in PSS:
            start = text.index(name + suffix)
            # The five option lines can straddle a page break. Match export code/label columns.
            segment = text[start:start + 1600]
            patterns = [r'1\s+0 = Not at all', r'2\s+1\s+1', r'3\s+2\s+2',
                        r'4\s+3\s+3', r'5\s+4 = Extremely']
            if not all(re.search(pattern, segment) for pattern in patterns):
                raise AuditError('Questionnaire option-code verification failed.')
            fields.append({'export_field': name, 'questionnaire_field': name + suffix,
                           'page': next(i + 1 for i, p in enumerate(pages) if name + suffix in p),
                           'stored_codes': [1, 2, 3, 4, 5], 'display_labels': [0, 1, 2, 3, 4]})
        evidence.append({'category': category, 'document_sha256': source.content_hash(),
                         'items': fields, 'missing_sentinels_documented': False})
    if sorted(x['category'] for x in evidence) != ['Endday', 'Midday', 'Morning']:
        raise AuditError('Three category-specific questionnaire definitions are required.')
    return sorted(evidence, key=lambda x: x['category'])


def score_items(raw):
    out = pd.DataFrame(index=raw.index)
    for name in PSS:
        s = raw[name].astype('string').str.strip()
        missing = s.isna() | s.isin(MISSING)
        v = pd.to_numeric(s, errors='coerce').astype(float)
        valid = ~missing & np.isfinite(v) & v.between(1, 5) & v.mod(1).eq(0)
        out[name + '_raw'] = raw[name]
        out[name + '_value'] = v.where(valid)
        out[name + '_status'] = np.select([missing, valid], ['missing', 'valid'], default='invalid')
    complete = out[[p + '_status' for p in PSS]].eq('valid').all(axis=1)
    out['pss_combo'] = (out.PSS_NERVOUS_value + 6 - out.PSS_COPE_value + 6 - out.PSS_CONTROL_value).where(complete)
    reasons = []
    for row in out[[p + '_status' for p in PSS]].itertuples(index=False, name=None):
        reasons.append(';'.join(p + ':' + status for p, status in zip(PSS, row) if status != 'valid') or 'available')
    out['outcome_reason'] = reasons
    return out


def local_context(actual_local):
    local = pd.to_datetime(actual_local, format='mixed', errors='coerce')
    seconds = local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second + local.dt.microsecond / 1e6
    phase = 2 * np.pi * seconds / 86400
    return pd.DataFrame({'local_time_sin': np.sin(phase), 'local_time_cos': np.cos(phase)}, index=actual_local.index)


def check_development(frame, assignments, expected_rows=None):
    dev = assignments.loc[assignments.partition.eq('development')].set_index('participant')
    if (not frame.n_partition.eq('development').all() or frame.n_ema_id.duplicated().any()
            or not frame.n_participant.isin(dev.index).all()
            or not frame.n_fold.eq(frame.n_participant.map(dev.fold)).all()
            or not frame.n_fold.isin(range(5)).all()
            or (expected_rows is not None and len(frame) != expected_rows)):
        raise AuditError('Development identity, unique EMA key or frozen fold validation failed.')


def link_outcomes(ema, features, assignments, expected_rows=3706):
    check_development(ema, assignments, expected_rows)
    check_development(features, assignments, expected_rows)
    if set(ema.n_ema_id) != set(features.n_ema_id):
        raise AuditError('EMA/feature key sets do not match.')
    f = features.set_index('n_ema_id').loc[ema.n_ema_id].reset_index()
    e = ema.reset_index(drop=True)
    for col in META:
        if not (e[col].eq(f[col]) | (e[col].isna() & f[col].isna())).all():
            raise AuditError('EMA/feature metadata consistency failed.')
    for schema in e.n_source_columns.unique():
        if not set(PSS) <= set(json.loads(schema)):
            raise AuditError('Required original raw item names are absent from an EMA source schema.')
    metadata = e[['n_actual_local', 'n_source_sha256', 'n_source_row']]
    linked = pd.concat([f, metadata, score_items(e), local_context(e.n_actual_local)], axis=1)
    linked['context_available'] = (linked[['local_time_sin', 'local_time_cos']].notna().all(axis=1)
                                   & linked.n_survey_category.isin(['Morning', 'Midday', 'Endday'])
                                   & ~linked.n_invalid_clock)
    return linked
