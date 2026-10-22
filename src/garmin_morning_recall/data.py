"""All-Morning accounting and observed-reading features; quality stays separate."""

import math

import numpy as np
import pandas as pd

from garmin_ema_baseline.models import CONTEXT, HR, feature_columns
from garmin_hr_activity.features import window
from garmin_inventory.privacy import AuditError
from garmin_recall_diagnostics.recall import META

WINDOWS = ["wake", "fixed_60", "fixed_5", "fixed_120"]
VALUES = ["mean_bpm", "sd_bpm", "latest_bpm", "slope_bpm_per_minute"]
STATUS = ["status_" + c for c in VALUES]
QUALITY = [
    "recorded_rows",
    "valid_readings",
    "rejected_rows",
    "conflicting_rows",
    "clock_rejected_rows",
    "value_rejected_rows",
    "recording_status",
    "observed_span_seconds",
    "latest_age_seconds",
    "left_gap_seconds",
    "right_gap_seconds",
    "maximum_internal_gap_seconds",
    "maximum_gap_including_edges_seconds",
    "occupied_minute_bins",
    "possible_minute_bins",
    *STATUS,
]


def columns_for(interval):
    if interval not in WINDOWS:
        raise AuditError("Unspecified Morning window requested.")
    return (
        [name + "__wake" for name in HR]
        if interval == "wake"
        else feature_columns(int(interval.split("_")[1]))
    )


def wake_features(groups, start, end, config):
    if not np.isfinite([start, end]).all() or end <= start:
        raise AuditError("Invalid wake interval cannot produce features.")
    values, quality, lineage = window(groups, start, end, "hr", config)
    quality = {k: quality[k] for k in QUALITY}
    quality["possible_minute_bins"] = math.ceil((end - start) / 60)
    quality["occupied_bin_fraction"] = (
        quality["occupied_minute_bins"] / quality["possible_minute_bins"]
    )
    return {col: values[name] for col, name in zip(columns_for("wake"), VALUES)}, quality, lineage


def unavailable_reason(quality):
    return (
        ";".join(sorted({quality[k] for k in STATUS if quality[k] != "available"})) or "available"
    )


def build_features(baseline, recall, features, config, hr_config):
    needed = (
        META
        + [
            "n_cutoff_epoch",
            "n_start_precision",
            "n_invalid_clock",
            "pss_combo",
            "outcome_reason",
            "context_available",
            *CONTEXT,
        ]
        + [c for interval in WINDOWS[1:] for c in columns_for(interval)]
    )
    original = pd.read_parquet(
        baseline / "linked_development.parquet",
        columns=needed,
        filters=[("n_survey_category", "=", "Morning")],
    )
    if len(original) != 934 or original.n_ema_id.duplicated().any():
        raise AuditError("Original Morning accounting changed.")
    boundaries = pd.read_parquet(
        recall / "boundaries.parquet", filters=[("n_survey_category", "=", "Morning")]
    )
    lineages = pd.read_parquet(
        features / "window_lineage.parquet",
        filters=[("sensor", "=", "hr"), ("lookback_minutes", "=", 5)],
    )
    directory_map = lineages[["n_ema_id", "directory"]]
    fixed_quality = pd.read_parquet(
        features / "quality.parquet",
        columns=META + ["lookback_minutes", *QUALITY],
        filters=[
            ("n_survey_category", "=", "Morning"),
            ("sensor", "=", "hr"),
            ("lookback_minutes", "in", [5, 60, 120]),
        ],
    )
    rows, qualities, source_links = [], [], []
    for scenario in config["scenarios"]:
        b = boundaries.loc[boundaries.scenario.eq(scenario)]
        f = original.merge(b, on=META, how="left", validate="one_to_one").merge(
            directory_map, on="n_ema_id", how="left", validate="one_to_one"
        )
        if (
            len(f) != 934
            or f.boundary_reason.isna().any()
            or f.directory.isna().any()
            or not f.end_epoch.eq(f.n_cutoff_epoch).all()
        ):
            raise AuditError("Wake linkage or cutoff consistency failed.")
        for directory, part in f.groupby("directory", sort=True):
            ledger = pd.read_parquet(features / "reading_dispositions" / directory / "hr.parquet")
            if len(ledger) and (
                not ledger.n_partition.eq("development").all()
                or not ledger.n_participant.isin(part.n_participant).all()
                or not ledger.n_fold.isin(part.n_fold).all()
            ):
                raise AuditError("Unexpected HR participant/partition/fold.")
            groups = ledger.loc[ledger.a_first & np.isfinite(ledger.n_epoch_s)].reset_index(
                drop=True
            )
            if not groups.n_epoch_s.is_monotonic_increasing:
                raise AuditError("HR groups must be chronological.")
            for record in part.to_dict("records"):
                if record["candidate_available"]:
                    values, q, lineage = wake_features(
                        groups, record["start_epoch"], record["end_epoch"], hr_config
                    )
                else:
                    values = dict.fromkeys(columns_for("wake"))
                    q = dict.fromkeys(QUALITY + ["occupied_bin_fraction"])
                    q.update({k: "boundary_unavailable" for k in STATUS})
                    q["recording_status"] = "boundary_unavailable"
                    lineage = {}
                rows.append({**record, **values})
                meta = {k: record[k] for k in META}
                qualities.append(
                    {
                        **meta,
                        "scenario": scenario,
                        "interval": "wake",
                        "start_epoch": record["start_epoch"],
                        "end_epoch": record["end_epoch"],
                        "duration_seconds": record["duration_seconds"],
                        **q,
                        "availability_reason": unavailable_reason(q),
                    }
                )
                source_links.append(
                    {
                        **meta,
                        "scenario": scenario,
                        "interval": "wake",
                        "directory": directory,
                        **lineage,
                    }
                )
        for record in fixed_quality.to_dict("records"):
            interval = "fixed_" + str(record.pop("lookback_minutes"))
            source = original.set_index("n_ema_id").loc[record["n_ema_id"]]
            for col in META[1:]:
                if source[col] != record[col]:
                    raise AuditError("Fixed quality identity mismatch.")
            end = float(source.n_cutoff_epoch)
            duration = int(interval.split("_")[1]) * 60
            qualities.append(
                {
                    **record,
                    "scenario": scenario,
                    "interval": interval,
                    "start_epoch": end - duration,
                    "end_epoch": end,
                    "duration_seconds": duration,
                    "occupied_bin_fraction": record["occupied_minute_bins"]
                    / record["possible_minute_bins"],
                    "availability_reason": unavailable_reason(record),
                }
            )
    f = pd.DataFrame(rows).sort_values(["scenario", "n_ema_id"]).reset_index(drop=True)
    q = (
        pd.DataFrame(qualities)
        .sort_values(["scenario", "interval", "n_ema_id"])
        .reset_index(drop=True)
    )
    accounting = f[
        META
        + [
            "scenario",
            "pss_combo",
            "outcome_reason",
            "context_available",
            "boundary_reason",
            "candidate_available",
            "available_at_prediction",
        ]
    ].copy()
    accounting["score_available"] = np.isfinite(f.pss_combo)
    for interval in WINDOWS:
        complete = np.isfinite(f[columns_for(interval)]).all(axis=1)
        accounting["complete_hr__" + interval] = complete
        reasons = q.loc[q.interval.eq(interval), ["scenario", "n_ema_id", "availability_reason"]]
        accounting = accounting.merge(
            reasons.rename(columns={"availability_reason": "reason__" + interval}),
            on=["scenario", "n_ema_id"],
            validate="one_to_one",
        )
        if not np.array_equal(accounting["reason__" + interval].eq("available"), complete):
            raise AuditError("Feature availability and quality reasons disagree.")
    return f, q, accounting, pd.DataFrame(source_links)


def availability(features, quality):
    rows = []
    for (scenario, interval), q in quality.groupby(["scenario", "interval"], sort=True):
        f = features.loc[features.scenario.eq(scenario)].set_index("n_ema_id").loc[q.n_ema_id]
        complete = np.isfinite(f[columns_for(interval)]).all(axis=1)
        rows.append(
            {
                "scenario": scenario,
                "interval": interval,
                "responses": len(q),
                "participants": q.n_participant.nunique(),
                "any_accepted_hr": int(q.valid_readings.gt(0).sum()),
                "complete_hr": int(complete.sum()),
                "complete_hr_participants": f.loc[complete, "n_participant"].nunique(),
                "complete_hr_and_score": int((complete & np.isfinite(f.pss_combo)).sum()),
                **{
                    name: int(q.recording_status.eq(name).sum())
                    for name in ["boundary_unavailable", "no_recording", "all_readings_rejected"]
                },
                "insufficient_feature_support": int(
                    (
                        q.recording_status.eq("observed_valid")
                        & q.availability_reason.ne("available")
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)
