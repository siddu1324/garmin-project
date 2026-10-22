"""Synthetic-only package tests. This module has no participant-data reader."""

import importlib
import importlib.metadata
import inspect
import math
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_inventory.privacy import sha256
from .contract import window_input


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    timestamps: tuple
    intervals_ms: tuple
    logical_rows: tuple
    scenario: str = "interval_end"
    excluded_rows: tuple = ()
    window_end: float = 300.0

    def frame(self):
        n = len(self.timestamps)
        frame = pd.DataFrame(
            dict(
                n_epoch_s=np.array(self.timestamps, float),
                n_value=np.array(self.intervals_ms, float),
                n_logical_row=np.array(self.logical_rows, int),
                q_logical_repeated_timestamp=np.zeros(n, bool),
                q_conflicting_timestamp_value=np.zeros(n, bool),
                q_invalid_text_clock=np.zeros(n, bool),
                q_clock_discrepancy=np.zeros(n, bool),
                q_unusual_offset=np.zeros(n, bool),
            )
        )
        frame["q_logical_repeated_timestamp"] = frame.n_epoch_s.duplicated(keep=False)
        distinct = frame.groupby("n_epoch_s").n_value.nunique()
        frame["q_conflicting_timestamp_value"] = frame.n_epoch_s.isin(distinct.index[distinct > 1])
        return frame


def fixtures():
    r = (800.0, 900.0, 1000.0, 950.0, 850.0)
    t = (10.8, 11.7, 12.7, 13.65, 14.5)
    ids = tuple(range(5))
    return [
        SyntheticCase("known_end", t, r, ids),
        SyntheticCase("known_start", (10.0, 10.8, 11.7, 12.7, 13.65), r, ids, "interval_start"),
        SyntheticCase(
            "constant_end", (10.8, 11.6, 12.4, 13.2, 14.0, 14.8), (800.0,) * 6, tuple(range(6))
        ),
        SyntheticCase("jitter_some_10ms_matches", (10.8, 11.7, 12.73, 13.68, 14.5), r, ids),
        SyntheticCase("jitter_no_10ms_matches", (10.8, 11.73, 12.68, 13.675, 14.5), r, ids),
        SyntheticCase("recording_gap", (10.8, 11.7, 32.7, 33.65, 34.5), r, ids),
        SyntheticCase(
            "removed_interval",
            (10.8, 11.7, 13.65, 14.5),
            (800.0, 900.0, 950.0, 850.0),
            (0, 1, 3, 4),
        ),
        SyntheticCase("rejected_interval", t, r, ids, excluded_rows=(2,)),
        SyntheticCase(
            "remaining_duplicate_timestamp",
            (10.8, 11.7, 11.7, 12.7, 13.65, 14.5),
            (800.0, 900.0, 900.0, 1000.0, 950.0, 850.0),
            tuple(range(6)),
        ),
        SyntheticCase(
            "conflicting_timestamp",
            (10.8, 11.7, 11.7, 12.7, 13.65, 14.5),
            (800.0, 900.0, 850.0, 1000.0, 950.0, 850.0),
            tuple(range(6)),
        ),
        SyntheticCase("boundary_end", (0.0, 0.5, 1.3, 299.5, 300.0), (800.0,) * 5, ids),
        SyntheticCase(
            "boundary_start", (0.0, 0.5, 1.3, 299.5, 300.0), (800.0,) * 5, ids, "interval_start"
        ),
        SyntheticCase(
            "strict_50ms_threshold",
            (10.8, 11.65, 12.45, 13.301),
            (800.0, 850.0, 800.0, 851.0),
            tuple(range(4)),
        ),
    ]


def independent_reference(values, links=None):
    """Python scalar arithmetic, independent of both packages and NumPy reductions."""
    r = [float(x) for x in values]
    if not r or not all(math.isfinite(x) and x > 0 for x in r):
        raise ValueError("Reference requires finite positive intervals.")
    if links is None:
        links = [True] * (len(r) - 1)
    if len(links) != len(r) - 1:
        raise ValueError("Explicit pair-mask length mismatch.")
    diffs = [r[i + 1] - r[i] for i, allowed in enumerate(links) if allowed]
    n, p = len(r), len(diffs)
    mean = sum(r) / n
    ss = sum((x - mean) ** 2 for x in r)
    return dict(
        mean_ibi_ms=mean,
        median_ibi_ms=statistics.median(r),
        sdnn_sample_ms=math.sqrt(ss / (n - 1)) if n > 1 else None,
        sd_population_ms=math.sqrt(ss / n),
        rmssd_ms=math.sqrt(sum(d * d for d in diffs) / p) if p else None,
        pnn50_pairs_percent=100 * sum(abs(d) > 50 for d in diffs) / p if p else None,
        nk_native_pnn50_percent=100 * sum(abs(d) > 50 for d in diffs) / (p + 1),
        intervals=n,
        pairs=p,
    )


def validate_run(records):
    if len(records) < 2 or not records.c_acceptable.all() or records.c_run_id.nunique() != 1:
        raise ValueError("Package adapter requires one acceptable run with at least two intervals.")
    if (
        not np.all(np.diff(records.n_logical_row) == 1)
        or not records.c_link_to_previous_original_record.iloc[1:].all()
    ):
        raise ValueError("Package adapter refuses broken original adjacency.")
    if (
        not np.isfinite(records.n_value).all()
        or not np.isfinite(records.n_epoch_s).all()
        or np.any(np.diff(records.n_epoch_s) <= 0)
    ):
        raise ValueError("Package adapter refuses missing, tied or reversed values/times.")


def package_run(case, records, backend):
    """Only synthetic fixtures are accepted in this checkpoint; no real-data entrypoint."""
    if not isinstance(case, SyntheticCase):
        raise TypeError("Only synthetic validation is authorized.")
    expected = case.frame().set_index("n_logical_row")
    if not records.n_logical_row.isin(expected.index).all():
        raise ValueError("Records do not belong to the synthetic fixture.")
    original = expected.loc[records.n_logical_row]
    if not np.array_equal(
        records.n_epoch_s.to_numpy(), original.n_epoch_s.to_numpy()
    ) or not np.array_equal(records.n_value.to_numpy(), original.n_value.to_numpy()):
        raise ValueError("Records do not belong to the synthetic fixture.")
    validate_run(records)
    rr = records.n_value.to_numpy(float)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if backend == "heartpy":
            from heartpy.analysis import calc_ts_measures

            d = np.abs(np.diff(rr))
            _, m = calc_ts_measures(rr, d, d * d, measures={}, working_data={})
            # Explicit conversion: population -> sample SD; proportion -> percent.
            result = dict(
                mean_ibi_ms=float(m["ibi"]),
                median_ibi_ms=None,
                sdnn_sample_ms=float(m["sdnn"] * math.sqrt(len(rr) / (len(rr) - 1))),
                rmssd_ms=float(m["rmssd"]),
                pnn50_pairs_percent=float(100 * m["pnn50"]),
            )
        elif backend == "neurokit2":
            if (
                case.scenario != "interval_end"
                or not records.c_neurokit_native_pair_compatible.iloc[1:].all()
            ):
                raise ValueError(
                    "NeuroKit native adjacency differs from the contract; timestamps must not be repaired."
                )
            import neurokit2 as nk
            from neurokit2.hrv.intervals_utils import _intervals_sanitize

            t = records.n_epoch_s.to_numpy(float)  # Always provide actual clock coordinates.
            sanitized, clock, missing = _intervals_sanitize(rr, t)
            if missing or not np.array_equal(sanitized, rr) or not np.array_equal(clock, t):
                raise ValueError("NeuroKit input sanitization changed the contract.")
            m = nk.hrv_time({"RRI": rr, "RRI_Time": t}, show=False).iloc[0]
            result = dict(
                mean_ibi_ms=float(m.HRV_MeanNN),
                median_ibi_ms=float(m.HRV_MedianNN),
                sdnn_sample_ms=float(m.HRV_SDNN),
                rmssd_ms=float(m.HRV_RMSSD),
                pnn50_pairs_percent=float(m.HRV_pNN50 * len(rr) / (len(rr) - 1)),
            )
        else:
            raise ValueError("Unknown backend.")
    result["warning_categories"] = sorted({type(w.message).__name__ for w in captured})
    return result


def package_evidence():
    specs = [
        ("neurokit2.hrv.hrv_time", "hrv_time"),
        ("neurokit2.hrv.intervals_utils", "_intervals_sanitize"),
        ("neurokit2.hrv.intervals_utils", "_intervals_successive"),
        ("neurokit2.hrv.intervals_utils", "_intervals_missing"),
        ("heartpy.heartpy", "process_rr"),
        ("heartpy.analysis", "calc_ts_measures"),
    ]
    sources = []
    with warnings.catch_warnings():
        # Import-time deprecations are documented separately; numerical call warnings
        # below must not depend on whether this is the first import in a process.
        warnings.simplefilter("ignore")
        for module, name in specs:
            f = getattr(importlib.import_module(module), name)
            sources.append(
                dict(
                    module=module,
                    function=name,
                    signature=str(inspect.signature(f)),
                    source_sha256=sha256(Path(inspect.getsourcefile(f))),
                    installed_docstring_present=bool(inspect.getdoc(f)),
                )
            )
    return dict(
        versions={
            n: importlib.metadata.version(n)
            for n in ["neurokit2", "heartpy", "numpy", "pandas", "scipy", "setuptools"]
        },
        inspected_interfaces=sources,
    )


def native_probe(case):
    """Negative-control demonstration only; never an approved adapter for broken runs."""
    import neurokit2 as nk
    from neurokit2.hrv.intervals_utils import _intervals_sanitize, _intervals_successive

    rr = np.array(case.intervals_ms)
    t = np.array(case.timestamps)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r, clock, missing = _intervals_sanitize(rr, t)
        native_pairs = _intervals_successive(r, clock)
        effective = native_pairs if missing else np.ones(len(r) - 1, bool)
        expected = independent_reference(r, effective)
        out = nk.hrv_time({"RRI": rr, "RRI_Time": t}, show=False).iloc[0]
    for key, value in [
        ("HRV_MeanNN", expected["mean_ibi_ms"]),
        ("HRV_MedianNN", expected["median_ibi_ms"]),
        ("HRV_SDNN", expected["sdnn_sample_ms"]),
        ("HRV_RMSSD", expected["rmssd_ms"]),
        ("HRV_pNN50", expected["nk_native_pnn50_percent"]),
    ]:
        np.testing.assert_allclose(out[key], value, rtol=1e-12, atol=1e-10)
    return dict(
        case=case.name,
        demonstration_only=True,
        native_missing_detected=bool(missing),
        native_10ms_matching_pairs=int(native_pairs.sum()),
        effective_package_pairs=int(effective.sum()),
        native_rmssd_ms=float(out.HRV_RMSSD),
        native_pnn50_percent=float(out.HRV_pNN50),
        mean_ibi_ms=float(out.HRV_MeanNN),
        median_ibi_ms=float(out.HRV_MedianNN),
        sample_sdnn_ms=float(out.HRV_SDNN),
        warnings=sorted({type(w.message).__name__ for w in caught}),
        reference_conventions_match=True,
    )


def validate_all(policy):
    evidence = package_evidence()
    results = []
    for case in fixtures():
        frame = case.frame()
        excluded = frame.n_logical_row.isin(case.excluded_rows).to_numpy()
        window = window_input(frame, case.window_end, policy, case.scenario, excluded=excluded)
        for _, run in window.runs.iterrows():
            rec = window.records.loc[window.records.c_run_id == run.run_id]
            if len(rec) < 2:
                continue
            ref = independent_reference(rec.n_value)
            for backend in ["heartpy", "neurokit2"]:
                try:
                    actual = package_run(case, rec, backend)
                except ValueError:
                    if backend != "neurokit2":
                        raise
                    results.append(
                        dict(
                            case=case.name,
                            run=int(run.run_id),
                            backend=backend,
                            status="blocked_native_adjacency_or_start_semantics",
                        )
                    )
                    continue
                for name, value in actual.items():
                    if name != "warning_categories" and value is not None:
                        np.testing.assert_allclose(value, ref[name], rtol=1e-12, atol=1e-10)
                results.append(
                    dict(
                        case=case.name,
                        run=int(run.run_id),
                        backend=backend,
                        status="matched",
                        reference=ref,
                        actual=actual,
                    )
                )
    probes = [
        native_probe(c)
        for c in fixtures()
        if c.name
        in [
            "known_end",
            "known_start",
            "jitter_some_10ms_matches",
            "jitter_no_10ms_matches",
            "recording_gap",
            "removed_interval",
        ]
    ]
    return dict(
        package_evidence=evidence,
        cases=len(fixtures()),
        comparisons=results,
        native_probes=probes,
        real_data_hrv_calculated=False,
        automatic_correction=False,
        interpolation=False,
        spectral_or_nonlinear_functions_called=False,
    )
