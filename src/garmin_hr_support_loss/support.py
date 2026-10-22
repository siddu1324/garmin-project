"""Verify accepted-reading occupancy from unchanged provenance and form fixed policies."""
import numpy as np
import pandas as pd

from garmin_ema_baseline.models import LOOKBACKS, cohort_masks
from garmin_inventory.privacy import AuditError


def accepted_occupancy(times, start, minutes):
    """Independent bin-boundary searches, with Python integer offset arithmetic."""
    edges = np.array([start + 60 * int(k) for k in range(int(minutes) + 1)])
    positions = np.searchsorted(times, edges, side='left')
    return int(np.count_nonzero(np.diff(positions)))


def verify_quality(linked, feature_bundle):
    q = pd.read_parquet(feature_bundle / 'quality.parquet')
    q = q.loc[q.sensor.eq('hr')].copy()
    lineage = pd.read_parquet(feature_bundle / 'window_lineage.parquet')
    lineage = lineage.loc[lineage.sensor.eq('hr')].copy()
    keys = ['n_ema_id', 'lookback_minutes']
    if len(q) != len(linked) * 8 or q.duplicated(keys).any() or lineage.duplicated(keys).any():
        raise AuditError('HR quality/lineage key accounting failed.')
    joined = q.merge(lineage[keys + ['directory', 'start_epoch', 'end_epoch']], on=keys, validate='one_to_one')
    e = linked.set_index('n_ema_id')
    if (not joined.n_participant.eq(joined.n_ema_id.map(e.n_participant)).all()
            or not joined.n_fold.eq(joined.n_ema_id.map(e.n_fold)).all()
            or not joined.n_partition.eq('development').all()):
        raise AuditError('HR quality identity/fold mismatch.')
    checked = 0
    for directory, windows in joined.groupby('directory', sort=True):
        ledger = pd.read_parquet(feature_bundle / 'reading_dispositions' / directory / 'hr.parquet')
        if len(ledger) and (set(ledger.n_participant) != set(windows.n_participant)
                            or not ledger.n_partition.eq('development').all()
                            or set(ledger.n_fold) != set(windows.n_fold)):
            raise AuditError('HR ledger development identity mismatch.')
        t = ledger.loc[ledger.a_analysis_keep, 'n_epoch_s'].to_numpy(float)
        if len(t) and (not np.isfinite(t).all() or not (np.diff(t) > 0).all()):
            raise AuditError('Accepted HR timestamp order/uniqueness failed.')
        for w in windows.itertuples(index=False):
            cutoff = float(e.loc[w.n_ema_id, 'n_cutoff_epoch'])
            if not np.isfinite(cutoff):
                raise AuditError('Unexpected invalid cutoff; do not infer occupancy.')
            start = cutoff - 60 * int(w.lookback_minutes)
            if start != w.start_epoch or cutoff != w.end_epoch:
                raise AuditError('HR corrected boundary provenance mismatch.')
            occupancy = accepted_occupancy(t, start, w.lookback_minutes)
            left, right = np.searchsorted(t, [start, cutoff], side='left')
            selected = t[left:right]
            if occupancy != w.occupied_minute_bins or len(selected) != w.valid_readings or w.possible_minute_bins != w.lookback_minutes:
                raise AuditError('Accepted HR occupancy/count diagnostic mismatch.')
            if len(selected):
                expected = [cutoff - selected[-1], selected[-1] - selected[0],
                            np.diff(np.r_[start, selected, cutoff]).max()]
                actual = [w.latest_age_seconds, w.observed_span_seconds, w.maximum_gap_including_edges_seconds]
                if not np.allclose(expected, actual, rtol=0, atol=1e-6):
                    raise AuditError('Accepted HR freshness/span/gap mismatch.')
            checked += 1
    q['accepted_occupancy_fraction'] = q.occupied_minute_bins / q.possible_minute_bins
    return q, {'verified_windows': checked, 'occupancy_counts_accepted_readings': True,
               'feature_values_altered': False, 'bin_origin': 'lookback left boundary',
               'verification': 'Independent searches in original accepted-reading ledger; integer offsets.'}


def support_masks(linked, quality):
    existing = cohort_masks(linked)
    out = {}
    for policy in ['A', 'B', 'C']:
        native = {}
        for minutes in LOOKBACKS:
            q = quality.loc[quality.lookback_minutes.eq(minutes)].set_index('n_ema_id').reindex(linked.n_ema_id)
            if len(q) != len(linked) or q.occupied_minute_bins.isna().any():
                raise AuditError('Quality records do not cover every EMA.')
            mask = existing[('native', 'hr', minutes)].copy()
            if policy in ['B', 'C']:
                mask &= pd.Series((5 * q.occupied_minute_bins >= 4 * minutes).to_numpy(), index=linked.index)
            if policy == 'C':
                mask &= pd.Series(q.latest_age_seconds.le(60).to_numpy(), index=linked.index)
            native[minutes] = mask
        common = pd.concat(native, axis=1).all(axis=1)
        for minutes in LOOKBACKS:
            out[(policy, 'native', minutes)] = native[minutes]
            out[(policy, 'common', minutes)] = common
    return out
