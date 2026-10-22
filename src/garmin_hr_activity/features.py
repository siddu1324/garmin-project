"""Observed-reading summaries and counter pairs; no interpolation or physiological inference."""

import numpy as np
import pandas as pd

HR_SUMMARIES = [
    "mean_bpm",
    "median_bpm",
    "sd_bpm",
    "minimum_bpm",
    "maximum_bpm",
    "q25_bpm",
    "q75_bpm",
]
HR_FEATURES = HR_SUMMARIES + ["latest_bpm", "slope_bpm_per_minute"]
COUNTERS = ["steps", "floorsClimbed", "calories", "intensityMinutes"]
CLOCK_FLAGS = ["q_invalid_epoch", "q_invalid_text_clock", "q_clock_discrepancy", "q_unusual_offset"]
PAIR_REASONS = {
    1: "no_previous",
    2: "rejected_endpoint",
    4: "date_transition",
    8: "offset_transition",
    16: "source_transition",
    32: "nonpositive_time",
    64: "long_gap",
    128: "counter_decrease",
}


def prepare(frame, sensor, config):
    """Preserve every row in ledger; explicit grouped view retains rejected points as barriers."""
    f = frame.copy()
    t = f.n_epoch_s.to_numpy(float)
    v = f.n_value.to_numpy(float)
    bad_clock = f[CLOCK_FLAGS].any(axis=1).to_numpy() | ~np.isfinite(t)
    bad_value = ~np.isfinite(v)
    if sensor == "hr":
        bad_value |= (v < config["hr_min_bpm"]) | (v > config["hr_max_bpm"])
    else:
        bad_value |= (v < 0) | (v != np.floor(v))
    f["a_bad_clock"] = bad_clock
    f["a_bad_value"] = bad_value
    groups = f.groupby("n_epoch_s", sort=False, dropna=False)
    f["a_group_size"] = groups.n_value.transform("size")
    f["a_conflict"] = groups.n_value.transform("nunique", dropna=False) > 1
    f["a_conflict"] |= f.q_conflicting_timestamp_value
    f["a_group_bad_clock"] = groups.a_bad_clock.transform("any")
    f["a_group_bad_value"] = groups.a_bad_value.transform("any")
    f["a_first"] = ~f.n_epoch_s.duplicated()
    f["a_same_value_tie_excess"] = ~f.a_first & ~f.a_conflict
    f["a_acceptable"] = ~(f.a_conflict | f.a_group_bad_clock | f.a_group_bad_value)
    f["a_analysis_keep"] = f.a_first & f.a_acceptable
    # All finite groups remain, even rejected ones, so pairs cannot jump over bad readings.
    g = f.loc[f.a_first & np.isfinite(t)].copy().reset_index(drop=True)
    if len(g) and not g.n_epoch_s.is_monotonic_increasing:
        raise ValueError("Normalized sensor order is not chronological.")
    if sensor != "hr":
        n = len(g)
        reason = np.zeros(n, np.uint16)
        if n:
            reason[0] = 1
            dt = g.n_epoch_s.diff().to_numpy(float)
            delta = g.n_value.diff().to_numpy(float)
            prev_good = g.a_acceptable.shift(fill_value=False).to_numpy(bool)
            tests = {
                2: ~(g.a_acceptable.to_numpy(bool) & prev_good),
                4: g.n_text_local.dt.date.ne(g.n_text_local.dt.date.shift()).to_numpy(),
                8: g.n_inferred_offset_minutes.ne(g.n_inferred_offset_minutes.shift()).to_numpy(),
                16: g.n_source_sha256.ne(g.n_source_sha256.shift()).to_numpy(),
                32: ~(dt > 0),
                64: dt > config["counter_maximum_pair_gap_seconds"],
                128: delta < 0,
            }
            for bit, mask in tests.items():
                reason[mask] |= bit
            g["a_pair_start"] = g.n_epoch_s.shift()
            g["a_pair_delta"] = delta
            g["a_pair_duration"] = dt
        else:
            for name in ["a_pair_start", "a_pair_delta", "a_pair_duration"]:
                g[name] = pd.Series(dtype=float)
        g["a_pair_rejection_mask"] = reason
        g["a_pair_accepted"] = reason == 0
    ledger_columns = [
        "n_logical_row",
        "n_physical_row",
        "n_source_sha256",
        "n_source_row",
        "n_epoch_s",
        "n_value",
        "n_participant",
        "n_partition",
        "n_fold",
    ] + [c for c in f if c.startswith("a_")]
    ledger = f[[c for c in ledger_columns if c in f]].copy()
    return g, ledger


def empty_frame():
    names = [
        "n_epoch_s",
        "n_value",
        "n_logical_row",
        "n_text_local",
        "n_inferred_offset_minutes",
        "n_source_sha256",
        "q_conflicting_timestamp_value",
        *CLOCK_FLAGS,
    ]
    f = pd.DataFrame({n: pd.Series(dtype=float) for n in names})
    f["n_text_local"] = pd.Series(dtype="datetime64[ns]")
    for n in ["q_conflicting_timestamp_value", *CLOCK_FLAGS]:
        f[n] = pd.Series(dtype=bool)
    return f


def window(g, start, end, sensor, config):
    """One direct lookback. Pair start and end must both be inside it; no borrowed baseline."""
    left, right = np.searchsorted(g.n_epoch_s.to_numpy(float), [start, end])
    w = g.iloc[left:right]
    valid = w[w.a_acceptable]
    t = valid.n_epoch_s.to_numpy(float)
    v = valid.n_value.to_numpy(float)
    n = len(valid)
    span = float(t[-1] - t[0]) if n else None
    status = (
        "no_recording" if not len(w) else "all_readings_rejected" if not n else "observed_valid"
    )
    quality = dict(
        recorded_rows=int(w.a_group_size.sum()),
        distinct_recorded_timestamps=len(w),
        valid_readings=n,
        rejected_rows=int(w.loc[~w.a_acceptable, "a_group_size"].sum()),
        same_value_tie_excess=int((w.loc[~w.a_conflict, "a_group_size"] - 1).sum()),
        conflicting_rows=int(w.loc[w.a_conflict, "a_group_size"].sum()),
        clock_rejected_rows=int(w.loc[w.a_group_bad_clock, "a_group_size"].sum()),
        value_rejected_rows=int(w.loc[w.a_group_bad_value, "a_group_size"].sum()),
        recording_status=status,
        observed_span_seconds=span,
        latest_age_seconds=float(end - t[-1]) if n else None,
        left_gap_seconds=float(t[0] - start) if n else None,
        right_gap_seconds=float(end - t[-1]) if n else None,
        maximum_internal_gap_seconds=float(np.diff(t).max()) if n > 1 else None,
        maximum_gap_including_edges_seconds=float(np.diff(np.r_[start, t, end]).max()),
        occupied_minute_bins=int(len(np.unique(np.floor((t - start) / 60)))) if n else 0,
        possible_minute_bins=int((end - start) / 60),
    )
    for threshold in config["quality_gap_diagnostics_seconds"]:
        quality[f"internal_gaps_gt_{threshold}s"] = int(np.sum(np.diff(t) > threshold))
    lineage = dict(
        group_slice_start=int(left),
        group_slice_stop=int(right),
        first_logical_row=int(w.n_logical_row.iloc[0]) if len(w) else None,
        last_logical_row=int(w.n_logical_row.iloc[-1]) if len(w) else None,
    )
    if sensor == "hr":
        values = {k: None for k in HR_FEATURES}
        reason = (
            status
            if status != "observed_valid"
            else "insufficient_observations"
            if n < config["hr_summary_minimum_observations"]
            else "insufficient_timestamp_span"
            if span < config["hr_summary_minimum_span_seconds"]
            else "available"
        )
        slope_reason = (
            status
            if status != "observed_valid"
            else "insufficient_observations"
            if n < config["hr_slope_minimum_observations"]
            else "insufficient_timestamp_span"
            if span < config["hr_slope_minimum_span_seconds"]
            else "available"
        )
        if reason == "available":
            values.update(
                mean_bpm=float(v.mean()),
                median_bpm=float(np.median(v)),
                sd_bpm=float(v.std(ddof=config["sd_ddof"])),
                minimum_bpm=float(v.min()),
                maximum_bpm=float(v.max()),
                q25_bpm=float(np.quantile(v, 0.25, method=config["quantile_method"])),
                q75_bpm=float(np.quantile(v, 0.75, method=config["quantile_method"])),
            )
        if n:
            values["latest_bpm"] = float(v[-1])
        if slope_reason == "available":
            x = (t - t[0]) / 60
            centered = x - x.mean()
            values["slope_bpm_per_minute"] = float(
                np.dot(centered, v - v.mean()) / np.dot(centered, centered)
            )
        for f in HR_SUMMARIES:
            quality["status_" + f] = reason
        quality["status_latest_bpm"] = "available" if n else status
        quality["status_slope_bpm_per_minute"] = slope_reason
    else:
        fully_inside = w.a_pair_start.ge(start)
        accepted = w.a_pair_accepted & fully_inside
        for bit, name in PAIR_REASONS.items():
            quality["pairs_" + name] = int(
                ((w.a_pair_rejection_mask.to_numpy(np.uint16) & bit) != 0).sum()
            )
        quality["pairs_crossing_left_boundary"] = int((w.a_pair_start < start).sum())
        # Right-crossing pairs need their future endpoint and are intentionally never inspected.
        quality["supported_pairs"] = int(accepted.sum())
        quality["supported_pair_seconds"] = float(w.loc[accepted, "a_pair_duration"].sum())
        quality["supported_pair_fraction"] = quality["supported_pair_seconds"] / (end - start)
        if sensor in config["unresolved_counters"]:
            values = {"added_supported": None}
            reason = config["unresolved_counters"][sensor]
            quality["supported_pairs"] = 0
            quality["supported_pair_seconds"] = 0.0
            quality["supported_pair_fraction"] = 0.0
        else:
            reason = (
                "available"
                if accepted.any()
                else status
                if status != "observed_valid"
                else "no_supported_counter_pair"
            )
            values = {
                "added_supported": float(w.loc[accepted, "a_pair_delta"].sum())
                if accepted.any()
                else None
            }
        quality["status_added_supported"] = reason
    return values, quality, lineage
