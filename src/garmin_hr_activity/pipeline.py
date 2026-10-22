"""Versioned development HR/activity features; only metadata and authorized streams are read."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (
    digest_object,
    frozen_assignments,
    input_guard,
    publish_bundle,
    verify_bundle,
    write_json,
)
from garmin_preprocessing.normalize import write_parquet
from .features import HR_FEATURES, COUNTERS, prepare, empty_frame, window

META = [
    "n_ema_id",
    "n_participant",
    "n_partition",
    "n_fold",
    "n_survey_category",
    "n_cutoff_epoch",
    "n_start_precision",
    "n_invalid_clock",
]


def require(condition):
    if not condition:
        raise AuditError("HR/activity invariant failed; private details withheld.")


def dictionary(config):
    entries = []
    for minutes in config["lookbacks_minutes"]:
        for sensor in ["hr", *COUNTERS]:
            for metric in HR_FEATURES if sensor == "hr" else ["added_supported"]:
                unresolved = sensor in config["unresolved_counters"]
                entries.append(
                    dict(
                        name=f"{sensor}_{metric}__{minutes}m",
                        sensor=sensor,
                        metric=metric,
                        lookback_minutes=minutes,
                        units="bpm/minute"
                        if metric.startswith("slope")
                        else "bpm"
                        if sensor == "hr"
                        else "steps"
                        if sensor == "steps"
                        else "floors"
                        if sensor == "floorsClimbed"
                        else "unconfirmed",
                        predictor=not unresolved,
                        minimum_observations=1
                        if metric == "latest_bpm"
                        else 3
                        if metric.startswith("slope")
                        else 2,
                        minimum_timestamp_span_seconds=0
                        if metric == "latest_bpm"
                        else 60
                        if metric.startswith("slope")
                        else 1
                        if sensor == "hr"
                        else "positive; each pair ≤300",
                        missingness_column="status_" + metric,
                        semantics="observed readings"
                        if sensor == "hr"
                        else "supported within-day counter additions"
                        if not unresolved
                        else config["unresolved_counters"][sensor],
                    )
                )
    return dict(
        label=config["label"],
        features=entries,
        predictor_columns=[r["name"] for r in entries if r["predictor"]],
        metadata_columns=META,
        quality_predictors=[],
        excluded_families=[
            "IBI",
            "Garmin stress",
            "PulseOx",
            "calories",
            "intensityMinutes",
            "PSS",
            "quality/missingness",
        ],
        specification="reports/hr_activity_feature_specification.md",
    )


def aggregate(features, quality, entries):
    availability = []
    gaps = []
    overlaps = []
    for minutes in sorted(quality.lookback_minutes.unique()):
        for category in ["All", "Morning", "Midday", "Endday"]:
            f = features if category == "All" else features[features.n_survey_category == category]
            q = quality[(quality.lookback_minutes == minutes) & quality.n_ema_id.isin(f.n_ema_id)]
            for entry in [e for e in entries if e["lookback_minutes"] == minutes]:
                count = f[entry["name"]].notna()
                z = q[q.sensor == entry["sensor"]]
                status = z["status_" + entry["metric"]].value_counts().to_dict()
                availability.append(
                    dict(
                        feature=entry["name"],
                        sensor=entry["sensor"],
                        metric=entry["metric"],
                        lookback_minutes=int(minutes),
                        survey_category=category,
                        ema_denominator=len(f),
                        participant_denominator=f.n_participant.nunique(),
                        available_ema=int(count.sum()),
                        available_participants=f.loc[count, "n_participant"].nunique(),
                        predictor=entry["predictor"],
                        no_recording_ema=int(z.recording_status.eq("no_recording").sum()),
                        all_rejected_ema=int(z.recording_status.eq("all_readings_rejected").sum()),
                        reasons={str(k): int(v) for k, v in status.items()},
                    )
                )
            for sensor, z in q.groupby("sensor", sort=True):

                def stat(col, fn):
                    vals = z[col].dropna()
                    return float(getattr(vals, fn)()) if len(vals) else None

                gaps.append(
                    dict(
                        sensor=sensor,
                        lookback_minutes=int(minutes),
                        survey_category=category,
                        ema_denominator=len(z),
                        no_recording_ema=int(z.recording_status.eq("no_recording").sum()),
                        all_rejected_ema=int(z.recording_status.eq("all_readings_rejected").sum()),
                        ema_with_valid_readings=int(z.valid_readings.gt(0).sum()),
                        median_reading_count=stat("valid_readings", "median"),
                        median_latest_age_seconds=stat("latest_age_seconds", "median"),
                        median_maximum_internal_gap_seconds=stat(
                            "maximum_internal_gap_seconds", "median"
                        ),
                        maximum_internal_gap_seconds=stat("maximum_internal_gap_seconds", "max"),
                        median_maximum_gap_including_edges_seconds=stat(
                            "maximum_gap_including_edges_seconds", "median"
                        ),
                        mean_occupied_minute_fraction=float(
                            (z.occupied_minute_bins / z.possible_minute_bins).mean()
                        ),
                        mean_supported_pair_fraction=stat("supported_pair_fraction", "mean")
                        if sensor != "hr"
                        else None,
                        ema_with_gap_gt_300s=int(
                            z.maximum_gap_including_edges_seconds.gt(300).sum()
                        ),
                    )
                )
            hr = f[f"hr_mean_bpm__{minutes}m"].notna()
            activity = (
                f[[f"{s}_added_supported__{minutes}m" for s in ["steps", "floorsClimbed"]]]
                .notna()
                .any(axis=1)
            )
            overlaps.append(
                dict(
                    lookback_minutes=int(minutes),
                    survey_category=category,
                    ema_denominator=len(f),
                    hr_and_activity_ema=int((hr & activity).sum()),
                    hr_only_ema=int((hr & ~activity).sum()),
                    activity_only_ema=int((~hr & activity).sum()),
                    neither_ema=int((~hr & ~activity).sum()),
                    participants_with_both=f.loc[hr & activity, "n_participant"].nunique(),
                )
            )
    return dict(availability=availability, gaps=gaps, overlap=overlaps)


def run(root, reference):
    config = json.loads((root / "config/hr_activity.json").read_text())
    require(config["lookbacks_minutes"] == [5, 15, 30, 60, 120, 360, 720, 1440])
    input_guard(root, reference, root / "data")
    assignments, dev = frozen_assignments(root, reference)
    folds = assignments.set_index("participant").fold.to_dict()
    normalized = root / config["normalized_bundle"]
    verify_bundle(normalized)
    reissue = json.loads((root / "private/hr_activity/reissue.json").read_text())
    audited = root / reissue["eligibility"]
    verify_bundle(audited)
    supersession = json.loads((root / "reports/window_supersession.json").read_text())
    require(
        supersession["corrected_manifest_sha256"] == sha256(audited / "manifest.json")
        and supersession["status"] == "complete"
    )
    inputs = dict(
        normalized_manifest=sha256(normalized / "manifest.json"),
        corrected_eligibility_manifest=sha256(audited / "manifest.json"),
        corrected_eligibility=str(audited.relative_to(root)),
        config_sha256=sha256(root / "config/hr_activity.json"),
        specification_sha256=sha256(root / "reports/hr_activity_feature_specification.md"),
        assignments_sha256=sha256(root / "private/frozen_participant_assignments.parquet"),
        environment_sha256=sha256(root / "uv.lock"),
        code={p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("features.py")]},
    )
    version = digest_object(inputs)[:16]
    parent = root / "private/hr_activity/features"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / version
    if destination.exists():
        verify_bundle(destination)
        return destination
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    write_json(build / "frozen_feature_rules.json", config)
    d = dictionary(config)
    write_json(build / "feature_dictionary.json", d)
    ema = pd.read_parquet(normalized / "ema/development_ema.parquet", columns=META)
    require(
        len(ema) == 3706 and set(ema.n_participant) == dev and not ema.n_ema_id.duplicated().any()
    )
    require(
        ema.n_partition.eq("development").all()
        and ema.n_fold.eq(ema.n_participant.map(folds)).all()
    )
    people = json.loads((normalized / "participant_directories.json").read_text())
    require({p["participant"] for p in people} == dev)
    rows = {row.n_ema_id: row._asdict() for row in ema.itertuples(index=False)}
    quality = []
    lineage = []
    sensor_counts = {}
    supported_union = {}
    ledger_rows = 0
    for number, person in enumerate(people):
        e = ema[ema.n_participant == person["participant"]]
        for sensor in ["hr", *COUNTERS]:
            path = normalized / "logical" / person["directory"] / f"{sensor}.parquet"
            f = pd.read_parquet(path) if path.exists() else empty_frame()
            if len(f):
                require(
                    f.n_participant.eq(person["participant"]).all()
                    and f.n_partition.eq("development").all()
                    and f.n_fold.eq(person["fold"]).all()
                )
            source_hash = sha256(path) if path.exists() else None
            g, ledger = prepare(f, sensor, config)
            ledger_rows += len(ledger)
            write_parquet(
                ledger, build / "reading_dispositions" / person["directory"] / f"{sensor}.parquet"
            )
            # Counter view retains every finite timestamp group, including rejected barriers.
            if sensor != "hr":
                fields = [
                    "n_logical_row",
                    "n_epoch_s",
                    "n_value",
                    "a_acceptable",
                    "a_group_size",
                    "a_pair_start",
                    "a_pair_delta",
                    "a_pair_duration",
                    "a_pair_rejection_mask",
                    "a_pair_accepted",
                ]
                write_parquet(
                    g[fields], build / "counter_pairs" / person["directory"] / f"{sensor}.parquet"
                )
            sensor_counts[sensor] = sensor_counts.get(sensor, 0) + len(f)
            for minutes in config["lookbacks_minutes"]:
                chosen = np.zeros(len(g), bool)
                for row in e.itertuples(index=False):
                    end = float(row.n_cutoff_epoch)
                    start = end - 60 * int(minutes)
                    invalid = bool(row.n_invalid_clock) or not np.isfinite(end)
                    if invalid:
                        values, q, ref = window(g.iloc[:0], 0.0, 60.0 * minutes, sensor, config)
                        q["recording_status"] = "invalid_cutoff"
                        for name in q:
                            if name.startswith("status_"):
                                q[name] = "invalid_cutoff"
                    else:
                        values, q, ref = window(g, start, end, sensor, config)
                    for metric, value in values.items():
                        rows[row.n_ema_id][f"{sensor}_{metric}__{minutes}m"] = value
                    meta = {k: getattr(row, k) for k in META}
                    quality.append(dict(meta, sensor=sensor, lookback_minutes=minutes, **q))
                    lineage.append(
                        dict(
                            n_ema_id=row.n_ema_id,
                            directory=person["directory"],
                            sensor=sensor,
                            lookback_minutes=minutes,
                            start_epoch=start,
                            end_epoch=end,
                            source_partition_sha256=source_hash,
                            **ref,
                        )
                    )
                    if sensor in config["supported_counters"] and not invalid:
                        a, b = ref["group_slice_start"], ref["group_slice_stop"]
                        chosen[a:b] |= (
                            g.a_pair_accepted.iloc[a:b] & g.a_pair_start.iloc[a:b].ge(start)
                        ).to_numpy()
                if sensor in config["supported_counters"]:
                    # Original pairs cannot overlap except shared endpoints; deduplicate via Boolean union.
                    key = f"{sensor}__{minutes}m"
                    supported_union[key] = supported_union.get(key, 0.0) + float(
                        g.loc[chosen, "a_pair_duration"].sum()
                    )
        if (number + 1) % 20 == 0:
            print(f"HR/activity: {number + 1}/{len(people)} development participants", flush=True)
    features = pd.DataFrame(list(rows.values()))
    features = features[META + [e["name"] for e in d["features"]]]
    q = pd.DataFrame(quality)
    lineage_frame = pd.DataFrame(lineage)
    require(
        len(features) == 3706
        and len(q) == 3706 * 8 * 5
        and not q.duplicated(["n_ema_id", "sensor", "lookback_minutes"]).any()
    )
    require(features[[e["name"] for e in d["features"] if not e["predictor"]]].isna().all().all())
    write_parquet(features, build / "features.parquet")
    write_parquet(q, build / "quality.parquet")
    write_parquet(lineage_frame, build / "window_lineage.parquet")
    result = aggregate(features, q, d["features"])
    result.update(
        version=version,
        inputs=inputs,
        label=config["label"],
        feature_rows=len(features),
        quality_rows=len(q),
        lineage_rows=len(lineage_frame),
        reading_disposition_rows=ledger_rows,
        sensor_rows=sensor_counts,
        predictor_count=len(d["predictor_columns"]),
        supported_counter_union_seconds=supported_union,
        all_development_ema_retained=True,
        final_test_processed=False,
        expanded_hrv=False,
        pss_targets=False,
        imputation=False,
        model=False,
    )
    write_json(build / "aggregate_results.json", result)
    input_guard(root, reference, root / "data")
    publish_bundle(
        build,
        destination,
        dict(
            version=version,
            inputs=inputs,
            feature_rows=len(features),
            quality_rows=len(q),
            development_only=True,
        ),
    )
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=os.environ.get("GARMIN_REFERENCE_PROJECT"))
    a = p.parse_args()
    if a.reference is None:
        raise AuditError("Set GARMIN_REFERENCE_PROJECT.")
    out = run(Path.cwd(), a.reference)
    write_json(
        Path("private/hr_activity/feature_run.json"), {"features": str(out.relative_to(Path.cwd()))}
    )
    print("Completed HR/activity bundle:", out.relative_to(Path.cwd()), flush=True)


if __name__ == "__main__":
    main()
