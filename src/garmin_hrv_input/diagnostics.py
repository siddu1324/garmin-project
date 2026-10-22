"""Read-only development rejection/continuity analysis. No HRV package imports."""

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (
    digest_object,
    frozen_assignments,
    input_guard,
    publish_bundle,
    read_config,
    verify_bundle,
    write_json,
)
from garmin_preprocessing.eligibility import METRICS, REASONS, evaluate_windows, parameter_array
from garmin_preprocessing.normalize import write_parquet
from .contract import window_input


def require(condition):
    if not condition:
        raise AuditError("HRV input diagnostic invariant failed; details withheld.")


def configuration_path(root):
    corrected = root / "config/hrv_input_diagnostics_corrected.json"
    return corrected if corrected.exists() else root / "config/hrv_input_diagnostics.json"


def run(root, reference):
    cfg_path = configuration_path(root)
    cfg = json.loads(cfg_path.read_text())
    policy_cfg = read_config(root)
    normalized, audited = root / cfg["normalized_bundle"], root / cfg["eligibility_bundle"]
    input_guard(root, reference, root / "data")
    _, dev = frozen_assignments(root, reference)
    verify_bundle(normalized)
    verify_bundle(audited)
    require(json.loads((audited / "frozen_policies.json").read_text()) == policy_cfg)
    inputs = dict(
        normalized_manifest=sha256(normalized / "manifest.json"),
        eligibility_manifest=sha256(audited / "manifest.json"),
        diagnostic_config=sha256(cfg_path),
        frozen_policy=sha256(root / "config/ibi_eligibility.json"),
        code={p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("contract.py")]},
    )
    version = digest_object(inputs)[:16]
    parent = root / "private/hrv_input_validation/diagnostics"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / version
    if destination.exists():
        verify_bundle(destination)
        return destination
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    write_json(build / "diagnostic_config.json", cfg)
    people = json.loads((normalized / "participant_directories.json").read_text())
    require({p["participant"] for p in people} == dev)
    policies = policy_cfg["policies"]
    screen = next(p for p in policies if p["name"] == "screen")
    overview = defaultdict(Counter)
    reasons = defaultdict(Counter)
    combinations = defaultdict(Counter)
    variations = defaultdict(Counter)
    continuity = defaultdict(Counter)
    run_stats = defaultdict(Counter)
    precision = Counter()
    pairs = Counter()
    window_rows = immediate_rows = record_rows = run_rows = 0
    cutoffs = pd.read_parquet(
        normalized / "ema/development_ema.parquet",
        columns=["n_ema_id", "n_participant", "n_fold", "n_cutoff_epoch", "n_invalid_clock"],
    )
    for number, person in enumerate(people):
        old = pd.read_parquet(audited / "windows" / f"{person['directory']}.parquet")
        require(old.n_partition.eq("development").all() and old.n_fold.eq(person["fold"]).all())
        immediate = old.loc[old.n_step_index.eq(0)].reset_index(drop=True)
        require(
            len(immediate)
            == int(
                ((cutoffs.n_participant == person["participant"]) & ~cutoffs.n_invalid_clock).sum()
            )
        )
        window_rows += len(old)
        immediate_rows += len(immediate)
        logical = normalized / "logical" / person["directory"] / "ibi.parquet"
        if logical.exists():
            frame = pd.read_parquet(logical)
            require(
                frame.n_participant.eq(person["participant"]).all()
                and frame.n_partition.eq("development").all()
                and frame.n_fold.eq(person["fold"]).all()
                and not frame.q_invalid_epoch.any()
            )
        else:
            frame = pd.DataFrame(
                {
                    c: pd.Series(dtype=d)
                    for c, d in {
                        "n_epoch_s": float,
                        "n_value": float,
                        "n_logical_row": int,
                        "q_logical_repeated_timestamp": bool,
                        "q_conflicting_timestamp_value": bool,
                        "q_invalid_text_clock": bool,
                        "q_clock_discrepancy": bool,
                        "q_unusual_offset": bool,
                    }.items()
                }
            )
        t, rr = frame.n_epoch_s.to_numpy(float), frame.n_value.to_numpy(float) / 1000
        repeated = frame.q_logical_repeated_timestamp.to_numpy(bool)
        conflict = frame.q_conflicting_timestamp_value.to_numpy(bool)
        clock = (
            frame[["q_invalid_text_clock", "q_clock_discrepancy", "q_unusual_offset"]]
            .any(axis=1)
            .to_numpy()
        )
        # Compact reference index for every approved window; original values remain in the immutable stream.
        index = old[
            ["n_ema_id", "n_step_index", "n_window_start_epoch", "n_window_end_epoch", "n_fold"]
        ].copy()
        index["logical_slice_start"] = np.searchsorted(t, index.n_window_start_epoch.to_numpy())
        index["logical_slice_stop"] = np.searchsorted(t, index.n_window_end_epoch.to_numpy())
        write_parquet(index, build / "window_index" / f"{person['directory']}.parquet")
        physical = normalized / "physical" / person["directory"] / "ibi.parquet"
        if physical.exists():
            for batch in pq.ParquetFile(physical).iter_batches(columns=["ts"], batch_size=100000):
                s = batch.to_pandas().ts
                digits = s.str.partition(".")[2].str.rstrip("0").str.len()
                precision.update(
                    {
                        f"raw_ts_fraction_digits_{int(k)}": int(v)
                        for k, v in digits.value_counts().items()
                    }
                )
                precision["scientific_notation_rows"] += int(
                    s.str.contains("[eE]", regex=True).sum()
                )
        dt = np.diff(t)
        plausible = np.isfinite(rr) & (rr >= 0.25) & (rr <= 2.5)
        goodpair = plausible[1:] & plausible[:-1] & (dt > 0)
        pairs.update(
            logical_intervals=len(t),
            positive_plausible_pairs=int(goodpair.sum()),
            zero_timestamp_pairs=int((dt == 0).sum()),
            compressed_pairs=int((goodpair & (dt < 0.25 * np.minimum(rr[1:], rr[:-1]))).sum()),
            gaps_over_5s=int((dt > 5).sum()),
            gaps_over_60s=int((dt > 60).sum()),
        )
        for quantum in [1.0, 0.1, 0.01, 0.001]:
            precision[f"logical_timestamps_on_{quantum:g}s_grid"] += int(
                (abs(t / quantum - np.rint(t / quantum)) * quantum <= 1e-6).sum()
            )
        for anchoring, expected in [("end", rr[1:]), ("start", rr[:-1])]:
            residual = abs(dt - expected)
            pairs[f"{anchoring}_pairs_within_10ms"] += int((goodpair & (residual <= 0.010)).sum())
            pairs[f"{anchoring}_pairs_within_250ms"] += int(
                (goodpair & (residual <= 0.250 + 1e-6)).sum()
            )
            # Quantization sensitivity bound only: no clocks are rounded/modified.
            pairs[f"{anchoring}_pairs_within_251ms"] += int(
                (goodpair & (residual <= 0.251 + 1e-6)).sum()
            )
        member_parts = []
        run_parts = []
        immediate_case_rows = []
        for policy in policies:
            for si, scenario in enumerate(policy_cfg["boundary_scenarios"]):
                label = f"{policy['name']}__{scenario}"
                mask = immediate[f"{label}__rejection_mask"].to_numpy(np.uint16)
                for category in ["All", "Morning", "Midday", "Endday"]:
                    scoped = (
                        np.ones(len(immediate), bool)
                        if category == "All"
                        else immediate.n_survey_category.eq(category).to_numpy()
                    )
                    m = mask[scoped]
                    recorded = immediate.recorded_ibi_count.to_numpy()[scoped]
                    key = (policy["name"], scenario, category)
                    overview[key].update(
                        ema_windows=len(m),
                        no_recorded_ibi=int((recorded == 0).sum()),
                        recorded_rejected=int(((recorded > 0) & (m != 0)).sum()),
                        eligible=int((m == 0).sum()),
                        recorded_fewer_than_two_contained=int(
                            ((recorded > 0) & ((m & 2) != 0)).sum()
                        ),
                        eligible_participants=int((m == 0).any()),
                    )
                    for bit, name in REASONS.items():
                        reasons[key + (name,)].update(
                            overlapping=int(((m & bit) != 0).sum()),
                            only_this_check=int((m == bit).sum()),
                            pass_if_only_gate_waived=int(
                                (((m & np.uint16(65535 - bit)) == 0) & (recorded > 0)).sum()
                            ),
                        )
                    combinations[key].update(
                        {int(k): int(v) for k, v in zip(*np.unique(m, return_counts=True))}
                    )
                for scope, subset in [("immediate", immediate), ("all_24h_instances", old)]:
                    for eligible_only in [False, True]:
                        g = subset.loc[subset[f"{label}__eligible"]] if eligible_only else subset
                        values = g[f"{label}__longest_sequence_seconds"].to_numpy()
                        key = (
                            policy["name"],
                            scenario,
                            scope,
                            "eligible" if eligible_only else "all",
                        )
                        continuity[key]["windows"] += len(g)
                        for lo, hi in zip(
                            cfg["continuity_span_bins_seconds"][:-1],
                            cfg["continuity_span_bins_seconds"][1:],
                        ):
                            continuity[key][f"span_{lo}_to_{hi}s"] += int(
                                ((values >= lo) & (values < hi)).sum()
                            )
                        for threshold in [120, 240, 285]:
                            continuity[key][f"span_at_least_{threshold}s"] += int(
                                (values + 1e-6 >= threshold).sum()
                            )
                for wi, row in immediate.iterrows():
                    hydrated = window_input(frame, float(row.n_window_end_epoch), policy, scenario)
                    rec = hydrated.records
                    runs = hydrated.runs
                    observed = float(runs.timestamp_span_seconds.max()) if len(runs) else 0.0
                    require(
                        np.isclose(observed, row[f"{label}__longest_sequence_seconds"], atol=1e-6)
                    )
                    require(
                        np.isclose(
                            hydrated.supported_seconds,
                            row[f"{label}__supported_seconds"],
                            atol=1e-6,
                        )
                    )
                    rec = rec.assign(
                        n_ema_id=row.n_ema_id, c_policy=policy["name"], c_scenario=scenario
                    )
                    if len(rec):
                        member_parts.append(rec)
                    if len(runs):
                        runs = runs.assign(
                            n_ema_id=row.n_ema_id,
                            c_policy=policy["name"],
                            c_scenario=scenario,
                            frozen_window_eligible=bool(mask[wi] == 0),
                        )
                        run_parts.append(runs)
                    stats = run_stats[(policy["name"], scenario)]
                    stats.update(
                        records=len(rec),
                        accepted_records=int(rec.c_acceptable.sum()),
                        provisional_jump_records=int(rec.c_provisional_jump_flag.sum()),
                        boundary_excluded_records=int(rec.c_boundary_excluded.sum()),
                        runs=len(runs),
                        allowed_adjacent_pairs=int(rec.c_link_to_previous_original_record.sum()),
                        native_neurokit_compatible_pairs=int(
                            rec.c_neurokit_native_pair_compatible.sum()
                        ),
                    )
                    candidates = 0
                    if len(runs) and mask[wi] == 0:
                        candidates = int(
                            (
                                (
                                    runs.timestamp_span_seconds
                                    >= cfg["pilot_candidate_minimum_run_span_seconds"]
                                )
                                & (
                                    runs.supported_seconds
                                    >= cfg["pilot_candidate_minimum_run_support_seconds"]
                                )
                            ).sum()
                        )
                    stats["pilot_candidate_runs"] += candidates
                    stats["windows_with_pilot_candidate"] += int(candidates > 0)
                    immediate_case_rows.append(
                        dict(
                            ema_id=row.n_ema_id,
                            policy=policy["name"],
                            scenario=scenario,
                            rejection_mask=int(mask[wi]),
                            run_count=len(runs),
                            longest_run_span=observed,
                            supported_seconds=hydrated.supported_seconds,
                            pilot_candidate_runs=candidates,
                        )
                    )
        if member_parts:
            members = pd.concat(member_parts, ignore_index=True)
            write_parquet(members, build / "immediate_members" / f"{person['directory']}.parquet")
            record_rows += len(members)
        if run_parts:
            runs = pd.concat(run_parts, ignore_index=True)
            write_parquet(runs, build / "immediate_runs" / f"{person['directory']}.parquet")
            run_rows += len(runs)
        write_parquet(
            pd.DataFrame(immediate_case_rows),
            build / "immediate_cases" / f"{person['directory']}.parquet",
        )
        ends = immediate.n_window_end_epoch.to_numpy()
        sensitivity_rows = []
        for variation in cfg["variations"]:
            policy = {**screen, **variation["changes"]}
            jump = variation.get("jump_relative_change", 0.2)
            for si, scenario in enumerate(policy_cfg["boundary_scenarios"]):
                m, metrics, recorded = evaluate_windows(
                    t, rr, repeated, conflict, clock, ends, si, parameter_array(policy), jump
                )
                if variation["name"] == "baseline":
                    require(np.array_equal(m, immediate[f"screen__{scenario}__rejection_mask"]))
                    np.testing.assert_array_equal(
                        metrics, immediate[[f"screen__{scenario}__{x}" for x in METRICS]].to_numpy()
                    )
                key = (variation["name"], scenario)
                variations[key].update(
                    ema_windows=len(m),
                    eligible=int((m == 0).sum()),
                    recorded_rejected=int(((recorded > 0) & (m != 0)).sum()),
                    eligible_participants=int((m == 0).any()),
                )
                sensitivity_rows.append(
                    pd.DataFrame(
                        dict(
                            ema_id=immediate.n_ema_id,
                            variation=variation["name"],
                            scenario=scenario,
                            rejection_mask=m,
                        )
                    )
                )
        write_parquet(
            pd.concat(sensitivity_rows, ignore_index=True),
            build / "sensitivity" / f"{person['directory']}.parquet",
        )
        if (number + 1) % 10 == 0:
            print(f"Development diagnostic partitions: {number + 1}/{len(people)}", flush=True)
    require(immediate_rows == len(cutoffs) == 3706 and window_rows == 5321816)

    def records(mapping, names):
        return [dict(zip(names, k), **dict(v)) for k, v in sorted(mapping.items())]

    combo = []
    for key, counts in sorted(combinations.items()):
        for mask, count in sorted(counts.items()):
            combo.append(
                dict(
                    zip(["policy", "scenario", "survey_category"], key),
                    mask=mask,
                    reasons=[name for bit, name in REASONS.items() if mask & bit],
                    windows=count,
                )
            )
    result = dict(
        version=version,
        inputs=inputs,
        window_index_rows=window_rows,
        immediate_windows=immediate_rows,
        immediate_member_case_rows=record_rows,
        immediate_run_rows=run_rows,
        overview=records(overview, ["policy", "scenario", "survey_category"]),
        reasons=records(reasons, ["policy", "scenario", "survey_category", "reason"]),
        combinations=combo,
        variations=records(variations, ["variation", "scenario"]),
        continuity=records(continuity, ["policy", "scenario", "scope", "subset"]),
        runs=records(run_stats, ["policy", "scenario"]),
        timestamp_precision=dict(precision),
        pair_timing=dict(pairs),
        real_hrv_calculated=False,
        final_test_processed=False,
        all_development_ema_accounted=True,
    )
    write_json(build / "aggregate_results.json", result)
    input_guard(root, reference, root / "data")
    publish_bundle(
        build, destination, dict(version=version, inputs=inputs, real_hrv_calculated=False)
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=os.environ.get("GARMIN_REFERENCE_PROJECT")
    )
    args = parser.parse_args()
    try:
        if args.reference is None:
            raise AuditError("Set GARMIN_REFERENCE_PROJECT.")
        path = run(Path.cwd(), args.reference)
        print("Completed immutable diagnostics:", path.relative_to(Path.cwd()), flush=True)
    except Exception as exc:
        print(
            str(exc)
            if isinstance(exc, AuditError)
            else f"Diagnostics failed ({type(exc).__name__}); private details withheld.",
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
