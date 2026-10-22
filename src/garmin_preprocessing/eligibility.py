"""Interval-support diagnostics only. No HRV, reconstruction or imputation.

Numba accelerates exact window-local loops; all scalar policy definitions are in JSON.
"""
import numpy as np
from numba import njit

REASONS = {
    1: 'no_recorded_ibi', 2: 'fewer_than_two_contained_intervals',
    4: 'implausible_or_missing_ibi', 8: 'repeated_timestamp',
    16: 'conflicting_timestamp_values', 32: 'clock_flag',
    64: 'insufficient_interval_time_agreement', 128: 'excessive_internal_gap_ratio',
    256: 'insufficient_edge_support', 512: 'insufficient_supported_duration',
    1024: 'insufficient_unbroken_sequence', 2048: 'excessive_artifact_flags',
    4096: 'duration_elapsed_mismatch',
}
METRICS = ['contained_count', 'boundary_excluded_count', 'acceptable_count', 'supported_seconds',
           'longest_sequence_seconds', 'consistent_pair_fraction', 'max_gap_expected_ratio',
           'left_gap_seconds', 'right_gap_seconds', 'artifact_interval_fraction',
           'duration_elapsed_ratio', 'max_internal_gap_seconds']
PARAMETERS = ['ibi_min_ms', 'ibi_max_ms', 'consistency_absolute_seconds', 'consistency_relative',
              'minimum_consistent_pair_fraction', 'maximum_gap_expected_ratio',
              'maximum_edge_gap_seconds', 'minimum_supported_seconds', 'minimum_contiguous_seconds',
              'maximum_artifact_interval_fraction', 'duration_elapsed_ratio_min', 'duration_elapsed_ratio_max']


def parameter_array(policy):
    return np.array([policy[k] for k in PARAMETERS], dtype=np.float64)


def window_ends(cutoff, lookback_minutes=1440):
    if not np.isfinite(cutoff):
        return np.array([], dtype=float)
    if lookback_minutes < 5:
        return np.array([], dtype=float)
    return cutoff - 60 * np.arange(lookback_minutes - 4, dtype=float)


@njit(cache=True)
def _masks(t, rr, left, right, a, b, scenario, p, artifact_change, neighbor_ratio, neighbor_slack):
    n = right - left
    contained = np.zeros(n, dtype=np.bool_)
    artifact = np.zeros(n, dtype=np.bool_)
    plausible = np.zeros(n, dtype=np.bool_)
    for j in range(n):
        i = left + j
        plausible[j] = np.isfinite(rr[i]) and p[0] <= rr[i] * 1000 <= p[1]
        if plausible[j]:
            start = t[i] - rr[i] if scenario == 0 else t[i]
            end = t[i] if scenario == 0 else t[i] + rr[i]
            contained[j] = start >= a - 1e-6 and end <= b + 1e-6
        if j > 0 and contained[j] and contained[j-1]:
            dt = t[i] - t[i-1]
            if 0 < dt <= neighbor_ratio * max(rr[i], rr[i-1]) + neighbor_slack:
                if abs(rr[i] - rr[i-1]) > artifact_change * rr[i-1]:
                    artifact[j] = True
                    artifact[j-1] = True
    return contained, artifact, plausible


@njit(cache=True)
def support_seconds(t, rr, chosen, scenario):
    """Union measure; monotonic starts (start scenario) or reverse ends (end scenario)."""
    total = 0.0
    if scenario == 1:
        cursor = -np.inf
        for i in range(len(t)):
            if chosen[i]:
                total += max(0.0, t[i] + rr[i] - max(t[i], cursor))
                cursor = max(cursor, t[i] + rr[i])
    else:
        cursor = np.inf
        for i in range(len(t) - 1, -1, -1):
            if chosen[i]:
                total += max(0.0, min(t[i], cursor) - (t[i] - rr[i]))
                cursor = min(cursor, t[i] - rr[i])
    return total


@njit(cache=True)
def evaluate_windows(t, rr, repeated, conflict, clock, ends, scenario, p,
                     artifact_change=.20, neighbor_ratio=2.0, neighbor_slack=.25):
    metrics = np.full((len(ends), 12), np.nan)
    masks = np.zeros(len(ends), dtype=np.uint16)
    recorded = np.zeros(len(ends), dtype=np.int32)
    for w in range(len(ends)):
        b, a = ends[w], ends[w] - 300
        left, right = np.searchsorted(t, a), np.searchsorted(t, b)
        n = right - left
        recorded[w] = n
        if n == 0:
            masks[w] = 1
            metrics[w, 0:5] = 0
            continue
        contained, artifact, plausible = _masks(t, rr, left, right, a, b, scenario, p,
                                               artifact_change, neighbor_ratio, neighbor_slack)
        good = np.zeros(n, dtype=np.bool_)
        reason, nc, na, boundary = 0, 0, 0, 0
        sum_rr, first, last = 0.0, np.inf, -np.inf
        consistent, pairs, maximum_ratio, maximum_gap = 0, 0, 0.0, 0.0
        run_start, longest = np.nan, 0.0
        for j in range(n):
            i = left + j
            if not plausible[j]:
                reason |= 4
            if repeated[i]:
                reason |= 8
            if conflict[i]:
                reason |= 16
            if clock[i]:
                reason |= 32
            boundary += int(plausible[j] and not contained[j])
            if contained[j]:
                nc += 1
                sum_rr += rr[i]
                first, last = min(first, t[i]), max(last, t[i])
                na += int(artifact[j])
            good[j] = contained[j] and not artifact[j] and not repeated[i] and not conflict[i] and not clock[i]
            link = False
            if j > 0:
                dt = t[i] - t[i-1]
                maximum_gap = max(maximum_gap, dt)
                if contained[j] and contained[j-1] and dt > 0:
                    expected = rr[i] if scenario == 0 else rr[i-1]
                    pairs += 1
                    agrees = abs(dt - expected) <= max(p[2], p[3] * expected) + 1e-6
                    consistent += int(agrees)
                    ratio = dt / expected
                    maximum_ratio = max(maximum_ratio, ratio)
                    link = agrees and ratio <= p[5] + 1e-9
            if good[j]:
                if j == 0 or not good[j-1] or not link:
                    run_start = t[i]
                longest = max(longest, t[i] - run_start)
            else:
                run_start = np.nan
        supported = support_seconds(t[left:right], rr[left:right], good, scenario)
        fraction = consistent / pairs if pairs else 0.0
        artifact_fraction = na / nc if nc else 1.0
        ratio_duration = sum_rr / (last - first) if nc >= 2 and last > first else np.nan
        edge_left, edge_right = first - a, b - last
        if nc < 2:
            reason |= 2
        if fraction < p[4]:
            reason |= 64
        if maximum_ratio > p[5]:
            reason |= 128
        if max(edge_left, edge_right) > p[6]:
            reason |= 256
        if supported + 1e-6 < p[7]:
            reason |= 512
        if longest + 1e-6 < p[8]:
            reason |= 1024
        if artifact_fraction > p[9] + 1e-12:
            reason |= 2048
        if not np.isfinite(ratio_duration) or not p[10] <= ratio_duration <= p[11]:
            reason |= 4096
        masks[w] = reason
        metrics[w] = np.array([nc, boundary, np.sum(good), supported, longest, fraction,
                               maximum_ratio, edge_left, edge_right, artifact_fraction,
                               ratio_duration, maximum_gap])
    return masks, metrics, recorded


@njit(cache=True)
def selected_support(t, rr, repeated, conflict, clock, selected_ends, scenario, p,
                     artifact_change=.20, neighbor_ratio=2.0, neighbor_slack=.25):
    """Union of actual acceptable hypothetical intervals from eligible windows, not nominal windows."""
    if len(selected_ends) == 0:
        return 0.0
    selected = np.zeros(len(t), dtype=np.bool_)
    for b in selected_ends:
        a = b - 300
        left, right = np.searchsorted(t, a), np.searchsorted(t, b)
        contained, artifact, plausible = _masks(t, rr, left, right, a, b, scenario, p,
                                               artifact_change, neighbor_ratio, neighbor_slack)
        for j in range(right - left):
            i = left + j
            if contained[j] and not artifact[j] and not repeated[i] and not conflict[i] and not clock[i]:
                selected[i] = True
    return support_seconds(t, rr, selected, scenario)


@njit(cache=True)
def union_seconds(starts, ends):
    if len(starts) == 0:
        return 0.0
    order = np.argsort(starts)
    cursor, total = -np.inf, 0.0
    for j in order:
        if np.isfinite(starts[j]) and ends[j] > starts[j]:
            total += max(0., ends[j] - max(starts[j], cursor))
            cursor = max(cursor, ends[j])
    return total


def reasons(mask):
    return [label for bit, label in REASONS.items() if int(mask) & bit]
