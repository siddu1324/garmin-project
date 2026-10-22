"""Documented recall boundaries; no inferred prompts or wake-time repair."""

import math
import re

import numpy as np
import pandas as pd

from garmin_hr_activity.features import window
from garmin_inventory.privacy import AuditError

META = ["n_ema_id", "n_participant", "n_partition", "n_fold", "n_survey_category"]
TIMING_FIELDS = [
    "actual_start_local",
    "scheduled_start_local",
    "timezone_offset",
    "instance_id",
    "SLEEP_WAKETIME",
    "SLEEP_SLEEPTIME",
    "Location Timestamp",
    "MORNING_SURVEY_COMPLETED",
    "ENDDAY_REMINDER",
]
EMA_COLUMNS = (
    META
    + TIMING_FIELDS
    + [
        "n_source_columns",
        "n_actual_local",
        "n_actual_utc",
        "n_offset_minutes",
        "n_cutoff_epoch",
        "n_start_precision",
        "n_invalid_clock",
        "n_scheduled_clock_discrepancy",
        "n_actual_before_scheduled",
    ]
)


def wake_boundary(actual_local, offset_minutes, cutoff, wake, invalid_clock=False):
    """Nominal minute start. The current EMA offset at waking is an explicit assumption."""
    if (
        invalid_clock
        or pd.isna(actual_local)
        or not np.isfinite(offset_minutes)
        or not np.isfinite(cutoff)
    ):
        return None, "invalid_ema_clock"
    if pd.isna(wake) or not str(wake).strip():
        return None, "missing_wake_response"
    match = re.fullmatch(r"(\d{1,2}):(\d{2})\s+(AM|PM)", str(wake).strip(), re.IGNORECASE)
    if not match:
        return None, "unparseable_wake_response"
    hour, minute, ampm = int(match[1]), int(match[2]), match[3].upper()
    if not 1 <= hour <= 12 or not 0 <= minute < 60:
        return None, "unparseable_wake_response"
    hour = hour % 12 + (12 if ampm == "PM" else 0)
    local = pd.Timestamp(actual_local).normalize() + pd.Timedelta(hours=hour, minutes=minute)
    utc = local - pd.Timedelta(minutes=float(offset_minutes))
    epoch = utc.tz_localize("UTC").timestamp()
    return epoch, "candidate" if epoch < cutoff else "wake_not_before_cutoff"


def boundaries(ema, config):
    rows = []
    for row in ema.itertuples(index=False):
        tags = {k: getattr(row, k) for k in META}
        cutoff = row.n_cutoff_epoch
        if row.n_survey_category == "Morning":
            start, reason = wake_boundary(
                row.n_actual_local,
                row.n_offset_minutes,
                cutoff,
                row.SLEEP_WAKETIME,
                row.n_invalid_clock,
            )
        else:
            start, reason = None, "previous_prompt_log_unavailable"
        for scenario, seconds in config["wake_scenarios_seconds"].items():
            left = None if start is None else start + seconds
            status = reason
            if reason == "candidate" and left >= cutoff:
                status = "minute_uncertainty_reaches_cutoff"
            rows.append(
                {
                    **tags,
                    "scenario": scenario,
                    "boundary_reason": status,
                    "start_epoch": left,
                    "end_epoch": cutoff,
                    "duration_seconds": cutoff - left if status == "candidate" else None,
                    "candidate_available": status == "candidate",
                    "available_at_prediction": False,
                    "retrospective_only": row.n_survey_category == "Morning",
                    "utc_offset_at_wake_verified": False,
                    "original_wake_precision_seconds": 60 if start is not None else None,
                }
            )
    return pd.DataFrame(rows)


def interval_quality(groups, start, end, config):
    if not np.isfinite([start, end]).all() or end <= start:
        raise AuditError("Candidate interval must be finite and positive.")
    # Reuse the corrected direct boundary engine; discard newly computed summary values.
    # Variable intervals need ceil, rather than fixed-lookback integer-minute denominators.
    _, q, lineage = window(groups, start, end, "hr", config)
    q = {k: v for k, v in q.items() if not k.startswith("status_")}
    q["possible_minute_bins"] = math.ceil((end - start) / 60)
    q["occupied_bin_fraction"] = q["occupied_minute_bins"] / q["possible_minute_bins"]
    q["partial_final_bin_seconds"] = (end - start) % 60
    return q, lineage


def coverage(boundary, features, config):
    lineage = pd.read_parquet(
        features / "window_lineage.parquet",
        columns=["n_ema_id", "directory", "sensor", "lookback_minutes"],
    )
    mapping = lineage.loc[
        lineage.sensor.eq("hr") & lineage.lookback_minutes.eq(5), ["n_ema_id", "directory"]
    ]
    work = boundary.loc[boundary.candidate_available].merge(
        mapping, on="n_ema_id", validate="many_to_one"
    )
    if len(work) != boundary.candidate_available.sum():
        raise AuditError("Candidate/source mapping lost rows.")
    rows = []
    for directory, part in work.groupby("directory", sort=True):
        ledger = pd.read_parquet(features / "reading_dispositions" / directory / "hr.parquet")
        if len(ledger) and (
            not ledger.n_partition.eq("development").all()
            or not ledger.n_participant.isin(part.n_participant).all()
        ):
            raise AuditError("Unexpected sensor identity in development candidate.")
        g = ledger.loc[ledger.a_first & np.isfinite(ledger.n_epoch_s)].reset_index(drop=True)
        if not g.n_epoch_s.is_monotonic_increasing:
            raise AuditError("HR ledger is not ordered.")
        for row in part.to_dict("records"):
            q, lineage_row = interval_quality(g, row["start_epoch"], row["end_epoch"], config)
            rows.append({**row, **q, **lineage_row})
    return pd.DataFrame(rows).sort_values(["scenario", "n_ema_id"]).reset_index(drop=True)


def describe(frame, tags):
    out = dict(tags, responses=len(frame), participants=frame.n_participant.nunique())
    if "valid_readings" not in frame:
        return out
    out.update(
        with_accepted_hr=int(frame.valid_readings.gt(0).sum()),
        participants_with_accepted_hr=frame.loc[
            frame.valid_readings.gt(0), "n_participant"
        ].nunique(),
        no_recording=int(frame.recording_status.eq("no_recording").sum()),
        all_readings_rejected=int(frame.recording_status.eq("all_readings_rejected").sum()),
    )
    columns = [
        "duration_seconds",
        "valid_readings",
        "occupied_bin_fraction",
        "observed_span_seconds",
        "latest_age_seconds",
        "maximum_internal_gap_seconds",
        "maximum_gap_including_edges_seconds",
    ]
    for col in columns:
        if col in frame:
            for label, quantile in [("q25", 0.25), ("median", 0.5), ("q75", 0.75)]:
                out[col + "_" + label] = frame[col].quantile(quantile)
            out[col + "_available"] = int(frame[col].notna().sum())
    return out


def summaries(boundary, candidates, fixed):
    availability = []
    for keys, f in boundary.groupby(
        ["scenario", "n_survey_category", "boundary_reason"], sort=True
    ):
        availability.append(describe(f, dict(zip(["scenario", "survey_category", "reason"], keys))))
    candidate_summary, matched, paired = [], [], []
    for (scenario, survey), f in candidates.groupby(["scenario", "n_survey_category"], sort=True):
        tags = {"scenario": scenario, "survey_category": survey}
        candidate_summary.append(describe(f, tags))
        for minutes, q in fixed.groupby("lookback_minutes", sort=True):
            comparison = (
                f[["n_ema_id", "occupied_bin_fraction", "valid_readings"]]
                .rename(
                    columns={
                        "occupied_bin_fraction": "occupied_bin_fraction_candidate",
                        "valid_readings": "valid_readings_candidate",
                    }
                )
                .merge(q, on="n_ema_id", validate="one_to_one")
            )
            if len(comparison) != len(f):
                raise AuditError("Fixed and candidate comparison membership differs.")
            comparison["occupied_bin_fraction"] = (
                comparison.occupied_minute_bins / comparison.possible_minute_bins
            )
            comparison["duration_seconds"] = int(minutes) * 60
            matched.append(describe(comparison, {**tags, "lookback_minutes": int(minutes)}))
            paired.append(
                {
                    **tags,
                    "lookback_minutes": int(minutes),
                    "responses": len(f),
                    "both_have_hr": int(
                        (
                            comparison.valid_readings.gt(0)
                            & comparison.valid_readings_candidate.gt(0)
                        ).sum()
                    ),
                    "candidate_only_hr": int(
                        (
                            comparison.valid_readings.eq(0)
                            & comparison.valid_readings_candidate.gt(0)
                        ).sum()
                    ),
                    "fixed_only_hr": int(
                        (
                            comparison.valid_readings.gt(0)
                            & comparison.valid_readings_candidate.eq(0)
                        ).sum()
                    ),
                    "neither_hr": int(
                        (
                            comparison.valid_readings.eq(0)
                            & comparison.valid_readings_candidate.eq(0)
                        ).sum()
                    ),
                    "median_paired_occupancy_difference_candidate_minus_fixed": float(
                        (
                            comparison.occupied_bin_fraction_candidate
                            - comparison.occupied_bin_fraction
                        ).median()
                    ),
                }
            )
    representation = []
    for keys, f in boundary.groupby(["scenario", "n_survey_category", "n_fold"], sort=True):
        representation.append(
            {
                **dict(zip(["scenario", "survey_category", "fold"], keys)),
                "responses": len(f),
                "participants": f.n_participant.nunique(),
                "candidate_responses": int(f.candidate_available.sum()),
                "candidate_participants": f.loc[f.candidate_available, "n_participant"].nunique(),
            }
        )
    return {
        k: pd.DataFrame(v)
        for k, v in dict(
            boundary_availability=availability,
            candidate_coverage=candidate_summary,
            matched_fixed_coverage=matched,
            paired_coverage=paired,
            representation=representation,
        ).items()
    }
