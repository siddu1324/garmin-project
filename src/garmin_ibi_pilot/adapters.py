"""Separate real-pilot adapters under explicit selection and run guards.

The historical synthetic-only adapter is imported for arithmetic/validation only,
and is never bypassed by fabricating a SyntheticCase for real data.
"""

import math
import warnings

import numpy as np

from garmin_hrv_input.synthetic import independent_reference, validate_run
from garmin_preprocessing.eligibility import support_seconds

FEATURES = ["mean_ibi_ms", "median_ibi_ms", "sdnn_sample_ms", "rmssd_ms", "pnn50_pairs_percent"]
LABEL = "IBI variability—normal-beat status unverified"


def guard_run(
    records, scenario, *, screen_eligible, development, span_min=240.0, support_min=240.0
):
    if not screen_eligible or not development or span_min < 240 or support_min < 240:
        raise ValueError(
            "Pilot requires development Screen eligibility and unchanged duration minima."
        )
    if scenario not in {"interval_end", "interval_start"}:
        raise ValueError("Unknown timestamp hypothesis.")
    validate_run(records)
    t = records.n_epoch_s.to_numpy(float)
    rr = records.n_value.to_numpy(float) / 1000
    support = float(
        support_seconds(t, rr, np.ones(len(t), bool), int(scenario == "interval_start"))
    )
    if t[-1] - t[0] < span_min or support < support_min:
        raise ValueError("Selected run fails the span/support pilot restriction.")
    return float(t[-1] - t[0]), support


def reference_run(records):
    validate_run(records)
    pairs = records.c_link_to_previous_original_record.to_numpy(bool)[1:]
    values = independent_reference(records.n_value.to_numpy(float), pairs)
    return {name: values[name] for name in FEATURES}


def package_comparison(records, scenario, backend):
    """Called only after the selection/Screen/duration guard. No cleaning or clock repair."""
    validate_run(records)
    rr = records.n_value.to_numpy(float)
    t = records.n_epoch_s.to_numpy(float)
    if backend == "neurokit2":
        if scenario != "interval_end":
            return dict(
                status="blocked", reason="interval_start_not_native_end_semantics", values={}
            )
        if not np.all(np.abs(np.diff(t) - rr[1:] / 1000) <= 0.010):
            return dict(status="blocked", reason="native_10ms_pair_set_incompatible", values={})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if backend == "heartpy":
            from heartpy.analysis import calc_ts_measures

            d = np.abs(np.diff(rr))
            _, m = calc_ts_measures(rr, d, d * d, measures={}, working_data={})
            values = dict(
                mean_ibi_ms=float(m["ibi"]),
                median_ibi_ms=None,
                sdnn_sample_ms=float(m["sdnn"] * math.sqrt(len(rr) / (len(rr) - 1))),
                rmssd_ms=float(m["rmssd"]),
                pnn50_pairs_percent=float(100 * m["pnn50"]),
            )
        elif backend == "neurokit2":
            import neurokit2 as nk
            from neurokit2.hrv.intervals_utils import _intervals_sanitize

            sanitized, clock, missing = _intervals_sanitize(rr, t)
            if missing or not np.array_equal(sanitized, rr) or not np.array_equal(clock, t):
                return dict(
                    status="blocked", reason="input_sanitization_changed_contract", values={}
                )
            m = nk.hrv_time({"RRI": rr, "RRI_Time": t}, show=False).iloc[0]
            values = dict(
                mean_ibi_ms=float(m.HRV_MeanNN),
                median_ibi_ms=float(m.HRV_MedianNN),
                sdnn_sample_ms=float(m.HRV_SDNN),
                rmssd_ms=float(m.HRV_RMSSD),
                pnn50_pairs_percent=float(m.HRV_pNN50 * len(rr) / (len(rr) - 1)),
            )
        else:
            raise ValueError("Unknown package backend.")
    return dict(
        status="compared",
        reason=None,
        values=values,
        warning_categories=sorted({type(w.message).__name__ for w in caught}),
    )
