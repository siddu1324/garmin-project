"""Correct the historical int16 offset overflow without modifying old bundles."""

import numpy as np

from garmin_preprocessing.eligibility import evaluate_windows, parameter_array
from .selection import require


def corrected_windows(windows, ema_lookup, frame, policy, policy_config):
    """Re-evaluate Screen only where stored ends differ from the approved cutoff grid."""
    w = windows.copy()
    cutoff = w.n_ema_id.map(ema_lookup.n_cutoff_epoch).to_numpy(float)
    steps = w.n_step_index.to_numpy(np.int64)
    ends = cutoff - 60 * steps
    require(np.isfinite(ends).all() and np.all(ends <= cutoff))
    require(np.all(ends - 300 >= cutoff - 86400))
    changed = w.n_window_end_epoch.to_numpy() != ends
    w["historical_window_clock_corrected"] = changed
    w["n_window_end_epoch"] = ends
    w["n_window_start_epoch"] = ends - 300
    if frame is None:
        t, rr = np.array([], float), np.array([], float)
        repeated, conflict, clock = (np.array([], bool) for _ in range(3))
    else:
        t, rr = frame.n_epoch_s.to_numpy(float), frame.n_value.to_numpy(float) / 1000
        repeated = frame.q_logical_repeated_timestamp.to_numpy(bool)
        conflict = frame.q_conflicting_timestamp_value.to_numpy(bool)
        clock = (
            frame[["q_invalid_text_clock", "q_clock_discrepancy", "q_unusual_offset"]]
            .any(axis=1)
            .to_numpy()
        )
    for si, scenario in enumerate(["interval_end", "interval_start"]):
        if changed.any():
            masks, metrics, _ = evaluate_windows(
                t,
                rr,
                repeated,
                conflict,
                clock,
                ends[changed],
                si,
                parameter_array(policy),
                policy_config["artifact_relative_change"],
                policy_config["artifact_neighbor_max_ratio"],
                policy_config["artifact_neighbor_slack_seconds"],
            )
            w.loc[changed, f"screen__{scenario}__eligible"] = masks == 0
            w.loc[changed, f"screen__{scenario}__longest_sequence_seconds"] = metrics[:, 4]
    return w
