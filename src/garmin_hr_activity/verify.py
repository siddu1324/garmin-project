"""Independent real-data verification without outcomes or expanded HRV."""

import argparse
import json
from datetime import timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (
    frozen_assignments,
    input_guard,
    verify_bundle,
    verify_archive_bundle,
    write_json,
)
from garmin_preprocessing.eligibility import _masks, parameter_array
from garmin_preprocessing.access import load_eligibility_windows
from .boundaries import verify_grid, expected_window
from .features import prepare, empty_frame, window, COUNTERS
from .pipeline import META, require, aggregate


def independent_counter(g, start, end, max_gap):
    """Scalar reference walks original timestamp groups, never a filtered adjacency sequence."""
    increments = []
    seconds = 0.0
    previous = None
    for r in g.itertuples():
        if r.n_epoch_s >= end:
            break
        if previous is not None and r.n_epoch_s >= start:
            p = previous
            okay = (
                p.n_epoch_s >= start
                and p.a_acceptable
                and r.a_acceptable
                and p.n_text_local.date() == r.n_text_local.date()
                and p.n_inferred_offset_minutes == r.n_inferred_offset_minutes
                and p.n_source_sha256 == r.n_source_sha256
                and 0 < r.n_epoch_s - p.n_epoch_s <= max_gap
                and r.n_value >= p.n_value
            )
            if okay:
                increments.append(float(r.n_value - p.n_value))
                seconds += float(r.n_epoch_s - p.n_epoch_s)
        previous = r
    return (sum(increments) if increments else None), len(increments), seconds


def verify(root, reference, bundle):
    input_guard(root, reference, root / "data")
    manifest = verify_bundle(bundle)
    cfg = json.loads((root / "config/hr_activity.json").read_text())
    n = root / cfg["normalized_bundle"]
    verify_bundle(n)
    a = root / manifest["inputs"]["corrected_eligibility"]
    verify_bundle(a)
    require(manifest["inputs"]["corrected_eligibility_manifest"] == sha256(a / "manifest.json"))
    require(
        manifest["inputs"]["specification_sha256"]
        == sha256(root / "reports/hr_activity_feature_specification.md")
    )
    assignments, dev = frozen_assignments(root, reference)
    fold = assignments.set_index("participant").fold.to_dict()
    ema = pd.read_parquet(
        n / "ema/development_ema.parquet", columns=META + ["n_actual_local", "n_offset_minutes"]
    )
    for r in ema.itertuples():
        if r.n_invalid_clock:
            continue
        utc = (
            r.n_actual_local.to_pydatetime() - timedelta(minutes=int(r.n_offset_minutes))
        ).replace(tzinfo=timezone.utc)
        cutoff = utc - timedelta(seconds=60 if r.n_start_precision == "minute" else 0)
        require(cutoff.timestamp() == r.n_cutoff_epoch)
    f = pd.read_parquet(bundle / "features.parquet")
    q = pd.read_parquet(bundle / "quality.parquet")
    lineage = pd.read_parquet(bundle / "window_lineage.parquet")
    pd.testing.assert_frame_equal(f[META], ema[META], check_dtype=False)
    require(len(f) == 3706 and f.n_ema_id.is_unique)
    require(set(f.n_participant) == dev and f.n_fold.eq(f.n_participant.map(fold)).all())
    require(
        q.n_partition.eq("development").all()
        and set(q.n_participant) == dev
        and q.n_fold.eq(q.n_participant.map(fold)).all()
    )
    require(
        len(q) == len(lineage) == 3706 * 8 * 5
        and not q.duplicated(["n_ema_id", "sensor", "lookback_minutes"]).any()
    )
    require(
        lineage.start_epoch.eq(
            lineage.end_epoch - 60 * lineage.lookback_minutes.astype("int64")
        ).all()
    )
    cutoffs = ema.set_index("n_ema_id").n_cutoff_epoch.to_dict()
    require(lineage.end_epoch.eq(lineage.n_ema_id.map(cutoffs)).all())
    entries = json.loads((bundle / "feature_dictionary.json").read_text())
    require(
        len(entries["predictor_columns"]) == 88
        and not set(entries["predictor_columns"]) & set(q.columns)
    )
    require(f[[e["name"] for e in entries["features"] if not e["predictor"]]].isna().all().all())
    policy_cfg = json.loads((root / "config/ibi_eligibility.json").read_text())
    people = json.loads((n / "participant_directories.json").read_text())
    total_windows = contained_checked = ledgers = feature_replays = scalar_counter_checks = 0
    features = f.set_index("n_ema_id")
    quality = q.set_index(["n_ema_id", "sensor", "lookback_minutes"])
    for number, person in enumerate(people):
        w = load_eligibility_windows(n, a, person["directory"])
        total_windows += verify_grid(
            w[["n_ema_id", "n_step_index", "n_window_start_epoch", "n_window_end_epoch"]],
            cutoffs,
            cfg["lookbacks_minutes"],
        )
        p = n / "logical" / person["directory"] / "ibi.parquet"
        if p.exists():
            ibi = pd.read_parquet(p, columns=["n_epoch_s", "n_value"])
            t = ibi.n_epoch_s.to_numpy(float)
            rr = ibi.n_value.to_numpy(float) / 1000
            for si, scenario in enumerate(["interval_end", "interval_start"]):
                for policy in policy_cfg["policies"]:
                    params = parameter_array(policy)
                    for row in w.loc[w[f"{policy['name']}__{scenario}__eligible"]].itertuples():
                        start, end = expected_window(cutoffs[row.n_ema_id], row.n_step_index)
                        left, right = np.searchsorted(t, [start, end])
                        tw, rw = t[left:right], rr[left:right]
                        contained, _, _ = _masks(
                            tw, rw, 0, len(tw), start, end, si, params, 0.2, 2.0, 0.25
                        )
                        starts = tw - rw if si == 0 else tw
                        ends = tw if si == 0 else tw + rw
                        require(
                            np.all(starts[contained] >= start - 1e-6)
                            and np.all(ends[contained] <= end + 1e-6)
                        )
                        require(np.all(ends[contained] <= cutoffs[row.n_ema_id]))
                        contained_checked += int(contained.sum())
        local = ema[ema.n_participant == person["participant"]]
        for sensor in ["hr", *COUNTERS]:
            path = n / "logical" / person["directory"] / f"{sensor}.parquet"
            original = pd.read_parquet(path) if path.exists() else empty_frame()
            ledger = pd.read_parquet(
                bundle / "reading_dispositions" / person["directory"] / f"{sensor}.parquet"
            )
            ledgers += len(ledger)
            g, replayed = prepare(original, sensor, cfg)
            pd.testing.assert_frame_equal(ledger, replayed)
            if len(ledger):
                require(
                    ledger.n_participant.eq(person["participant"]).all()
                    and ledger.n_fold.eq(person["fold"]).all()
                )
            # Recompute all HR/activity features only; no HRV function is imported or invoked.
            for r in local.itertuples():
                if r.n_invalid_clock:
                    continue
                for minutes in cfg["lookbacks_minutes"]:
                    end = float(r.n_cutoff_epoch)
                    start = end - int(minutes) * 60
                    values, checks, _ = window(g, start, end, sensor, cfg)
                    for metric, value in values.items():
                        stored = features.loc[r.n_ema_id, f"{sensor}_{metric}__{minutes}m"]
                        require(
                            pd.isna(stored)
                            if value is None
                            else np.isclose(stored, value, rtol=1e-12, atol=1e-10)
                        )
                        require(
                            quality.loc[(r.n_ema_id, sensor, minutes), "status_" + metric]
                            == checks["status_" + metric]
                        )
                    feature_replays += 1
            # A deterministic independent scalar audit for first EMA per participant, all lookbacks.
            if len(local) and len(g) and sensor in cfg["supported_counters"]:
                r = local.iloc[0]
                for minutes in cfg["lookbacks_minutes"]:
                    end = float(r.n_cutoff_epoch)
                    start = end - int(minutes) * 60
                    # Limit to preceding group plus observations within the lookback, preserving barriers.
                    lo = max(0, int(np.searchsorted(g.n_epoch_s.to_numpy(), start)) - 1)
                    hi = int(np.searchsorted(g.n_epoch_s.to_numpy(), end))
                    value, count, seconds = independent_counter(
                        g.iloc[lo:hi], start, end, cfg["counter_maximum_pair_gap_seconds"]
                    )
                    stored = features.loc[r.n_ema_id, f"{sensor}_added_supported__{minutes}m"]
                    quality_row = quality.loc[(r.n_ema_id, sensor, minutes)]
                    require(pd.isna(stored) if value is None else stored == value)
                    require(
                        quality_row.supported_pairs == count
                        and np.isclose(
                            quality_row.supported_pair_seconds, seconds, rtol=0, atol=1e-6
                        )
                    )
                    scalar_counter_checks += 1
        if (number + 1) % 20 == 0:
            print(f"Verification: {number + 1}/{len(people)} development partitions", flush=True)
    require(total_windows == 5321816)
    expected = aggregate(f, q, entries["features"])
    reported = json.loads((bundle / "aggregate_results.json").read_text())
    for key, value in expected.items():
        require(value == reported[key])
    # Original archived artifacts must remain byte-identical even though their current-use guard rejects them.
    old = [
        "private/preprocessing/eligibility/6747223eb50269ba",
        "private/hrv_input_validation/diagnostics/bc13763b35972be4",
        "private/ibi_pilot/selection/52eb55d251e1d5e1",
        "private/ibi_pilot/calculation/59b11baba770afb0",
    ]
    for name in old:
        verify_archive_bundle(root / name)
    require(ledgers == reported["reading_disposition_rows"])
    input_guard(root, reference, root / "data")
    return dict(
        input_integrity=True,
        final_test_processed=False,
        frozen_development_participants=125,
        sealed_final_test_participants=32,
        folds=5,
        feature_rows=len(f),
        quality_rows=len(q),
        window_lineage_rows=len(lineage),
        reading_disposition_rows=ledgers,
        independently_checked_window_rows=total_windows,
        contained_interval_members_checked=contained_checked,
        feature_window_replays=feature_replays,
        independent_scalar_counter_checks=scalar_counter_checks,
        all_ema_metadata_preserved=True,
        all_reading_dispositions_reproduced=True,
        all_aggregates_reconciled=True,
        old_bundles_preserved=True,
        dataset_rows={
            str(p.relative_to(root)): pq.ParquetFile(p).metadata.num_rows
            for p in [
                bundle / "features.parquet",
                bundle / "quality.parquet",
                bundle / "window_lineage.parquet",
            ]
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    a = p.parse_args()
    try:
        result = verify(Path.cwd(), a.reference, a.bundle)
        write_json(Path("private/hr_activity/verification_checks.json"), result)
        print("Independent boundary, feature, cohort and integrity checks passed.", flush=True)
    except Exception as exc:
        print(
            str(exc)
            if isinstance(exc, AuditError)
            else f"Verification failed ({type(exc).__name__}); private details withheld.",
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
