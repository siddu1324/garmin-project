"""Lossless window membership and explicit run boundaries; never computes HRV."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from garmin_preprocessing.eligibility import _masks, parameter_array, support_seconds


@dataclass
class WindowInput:
    """Original records plus per-scenario flags. Rejects remain in records with run_id=-1."""

    start: float
    end: float
    policy: str
    scenario: str
    records: pd.DataFrame
    runs: pd.DataFrame
    supported_seconds: float


def window_input(frame, end, policy, scenario, *, excluded=None):
    """Hydrate any approved window from its immutable logical stream.

    The original logical-row sequence, including rejected rows, is retained. An
    externally rejected or absent original row severs adjacency. No package call.
    """
    if scenario not in {"interval_end", "interval_start"}:
        raise ValueError("Unknown interval-boundary hypothesis.")
    if not np.isfinite(end):
        raise ValueError("Cannot assign a window to an invalid cutoff.")
    t_all = frame.n_epoch_s.to_numpy(float)
    if not np.isfinite(t_all).all() or np.any(np.diff(t_all) < 0):
        raise ValueError("Contract requires finite, chronologically ordered timestamps.")
    left, right = np.searchsorted(t_all, [end - 300, end])
    records = frame.iloc[left:right].copy().reset_index(drop=True)
    t = records.n_epoch_s.to_numpy(float)
    rr = records.n_value.to_numpy(float) / 1000
    n = len(t)
    si = int(scenario == "interval_start")
    p = parameter_array(policy)
    contained, jump, plausible = _masks(t, rr, 0, n, end - 300, end, si, p, 0.2, 2.0, 0.25)
    repeats = records.q_logical_repeated_timestamp.to_numpy(bool)
    conflicts = records.q_conflicting_timestamp_value.to_numpy(bool)
    clock = (
        records[["q_invalid_text_clock", "q_clock_discrepancy", "q_unusual_offset"]]
        .any(axis=1)
        .to_numpy()
    )
    # Defend against caller-supplied stale duplicate flags.
    repeats |= records.n_epoch_s.duplicated(keep=False).to_numpy()
    external = np.zeros(n, bool) if excluded is None else np.asarray(excluded, bool)[left:right]
    good = contained & ~jump & ~repeats & ~conflicts & ~clock & ~external
    ids = records.n_logical_row.to_numpy(np.int64)
    link = np.zeros(n, bool)
    nk_link = np.zeros(n, bool)
    if n > 1:
        dt = np.diff(t)
        expected = rr[1:] if si == 0 else rr[:-1]
        time_link = (
            (dt > 0)
            & (np.abs(dt - expected) <= np.maximum(p[2], p[3] * expected) + 1e-6)
            & (dt <= p[5] * expected + 1e-9)
        )
        link[1:] = good[1:] & good[:-1] & (np.diff(ids) == 1) & time_link
        # NeuroKit native succession is end-anchored regardless of this scenario.
        nk_link[1:] = link[1:] & (np.abs(dt - rr[1:]) <= 0.010)
    run_id = np.full(n, -1, np.int32)
    current = -1
    for i in range(n):
        if good[i]:
            if not link[i]:
                current += 1
            run_id[i] = current
    records["c_plausible"] = plausible
    records["c_boundary_excluded"] = plausible & ~contained
    records["c_contained"] = contained
    records["c_provisional_jump_flag"] = jump
    records["c_external_rejection"] = external
    records["c_acceptable"] = good
    records["c_link_to_previous_original_record"] = link
    records["c_neurokit_native_pair_compatible"] = nk_link
    records["c_run_id"] = run_id
    records["c_hypothesized_start"] = t - rr if si == 0 else t
    records["c_hypothesized_end"] = t if si == 0 else t + rr
    rows = []
    for rid in range(current + 1):
        ix = np.flatnonzero(run_id == rid)
        chosen = np.zeros(n, bool)
        chosen[ix] = True
        count = len(ix)
        rows.append(
            dict(
                run_id=rid,
                first_member=int(ix[0]),
                last_member=int(ix[-1]),
                first_logical_row=int(ids[ix[0]]),
                last_logical_row=int(ids[ix[-1]]),
                interval_count=count,
                allowed_pair_count=count - 1,
                timestamp_span_seconds=float(t[ix[-1]] - t[ix[0]]),
                supported_seconds=float(support_seconds(t, rr, chosen, si)),
                native_neurokit_pair_count=int(nk_link[ix].sum()),
                mean_median_mathematically_defined=count >= 1,
                sdnn_mathematically_defined=count >= 2,
                rmssd_pnn50_mathematically_defined=count >= 2,
                normal_beat_status="unverified",
                real_hrv_authorized=False,
            )
        )
    return WindowInput(
        end - 300,
        end,
        policy["name"],
        scenario,
        records,
        pd.DataFrame(rows),
        float(support_seconds(t, rr, good, si)),
    )
