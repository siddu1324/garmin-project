"""Past-only unlabeled features, separate quality and full-response accounting."""

import numpy as np
import pandas as pd

from garmin_ema_baseline.models import feature_columns
from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import digest_object

META = ["n_ema_id", "n_participant", "n_partition", "n_fold", "n_survey_category", "n_cutoff_epoch"]
NEW = ["reference_mean_bpm", "hr_mean_deviation_bpm", "hr_latest_deviation_bpm"]


def earlier_reference(ledger, cutoff, config):
    """Cut before UTC binning. Records at the right endpoint belong to the current window."""
    if not np.isfinite(cutoff):
        return None, dict(reason="invalid_cutoff", reference_available=False), dict()
    start = cutoff - config["history_start_hours"] * 3600
    end = cutoff - config["history_end_hours"] * 3600
    # The original logical ledger retains duplicate dispositions and rejected groups.
    mask = ledger.n_epoch_s.ge(start) & ledger.n_epoch_s.lt(end)
    recorded = ledger.loc[mask]
    accepted = recorded[recorded.a_analysis_keep].sort_values("n_epoch_s")
    t = accepted.n_epoch_s.to_numpy(float)
    v = accepted.n_value.to_numpy(float)
    bins = np.floor(t / 60).astype(np.int64)
    minute = pd.DataFrame({"minute": bins, "value": v}).groupby("minute", sort=True).value.mean()
    hours = len(np.unique(np.floor(t / 3600)))
    reasons = []
    if not len(recorded):
        reasons.append("no_recording")
    elif not len(accepted):
        reasons.append("all_recorded_readings_rejected")
    if len(minute) < config["minimum_occupied_utc_minutes"]:
        reasons.append("fewer_than_30_occupied_minutes")
    if hours < config["minimum_occupied_utc_hours"]:
        reasons.append("fewer_than_3_utc_hours")
    available = not reasons
    q = dict(
        start_epoch=float(start),
        end_epoch=float(end),
        duration_seconds=float(end - start),
        recorded_rows=len(recorded),
        accepted_readings=len(accepted),
        rejected_rows=int((~recorded.a_acceptable).sum()),
        same_value_tie_excess=int(recorded.a_same_value_tie_excess.sum()),
        conflicting_rows=int(recorded.a_conflict.sum()),
        occupied_utc_minutes=len(minute),
        occupied_utc_hours=hours,
        possible_utc_minute_bins=int(np.ceil(end / 60) - np.floor(start / 60)),
        occupied_minute_fraction=len(minute) / (np.ceil(end / 60) - np.floor(start / 60)),
        reference_available=available,
        reason=";".join(reasons) or "available",
        first_reading_age_seconds=float(cutoff - t[0]) if len(t) else None,
        latest_reading_age_seconds=float(cutoff - t[-1]) if len(t) else None,
        observed_span_seconds=float(t[-1] - t[0]) if len(t) else None,
        left_gap_seconds=float(t[0] - start) if len(t) else None,
        right_gap_seconds=float(end - t[-1]) if len(t) else None,
        maximum_internal_gap_seconds=float(np.diff(t).max()) if len(t) > 1 else None,
        maximum_gap_including_edges_seconds=float(np.diff(np.r_[start, t, end]).max()),
        mean_observed_minute_age_seconds=float(
            cutoff - pd.DataFrame({"minute": bins, "time": t}).groupby("minute").time.mean().mean()
        )
        if len(minute)
        else None,
    )
    lineage = dict(
        accepted_source_set_sha256=digest_object(
            [(str(a), int(b)) for a, b in zip(accepted.n_source_sha256, accepted.n_source_row)]
        ),
        first_logical_row=int(accepted.n_logical_row.iloc[0]) if len(t) else None,
        last_logical_row=int(accepted.n_logical_row.iloc[-1]) if len(t) else None,
    )
    return float(minute.mean()) if available else None, q, lineage


def build_reference(unlabeled, features, config):
    allowed = META + feature_columns(60)
    if set(unlabeled) != set(allowed):
        raise AuditError("Reference builder accepts only explicit unlabeled inputs.")
    lineage = pd.read_parquet(features / "window_lineage.parquet")
    mapping = (
        lineage[lineage.sensor.eq("hr") & lineage.lookback_minutes.eq(60)]
        .set_index("n_ema_id")
        .directory
    )
    if not mapping.index.is_unique or set(mapping.index) != set(unlabeled.n_ema_id):
        raise AuditError("HR lineage keys differ.")
    data = unlabeled.copy()
    data["directory"] = data.n_ema_id.map(mapping)
    outputs, quality, provenance = [], [], []
    empty = pd.DataFrame(
        {
            c: pd.Series(dtype=t)
            for c, t in {
                "n_epoch_s": float,
                "n_value": float,
                "a_analysis_keep": bool,
                "a_acceptable": bool,
                "a_same_value_tie_excess": bool,
                "a_conflict": bool,
                "n_source_sha256": str,
                "n_source_row": int,
                "n_logical_row": int,
            }.items()
        }
    )
    for directory, part in data.groupby("directory", dropna=False):
        path = features / "reading_dispositions" / str(directory) / "hr.parquet"
        ledger = pd.read_parquet(path) if path.exists() else empty
        if not len(ledger):
            ledger = empty
        if len(ledger) and (
            not ledger.n_partition.eq("development").all()
            or set(ledger.n_participant) != set(part.n_participant)
            or set(ledger.n_fold) != set(part.n_fold)
        ):
            raise AuditError("HR ledger identity/partition/fold mismatch.")
        for _, row in part.iterrows():
            reference, q, lin = earlier_reference(ledger, row.n_cutoff_epoch, config)
            meta = row[META].to_dict()
            outputs.append(
                {
                    **meta,
                    "reference_mean_bpm": reference,
                    "hr_mean_deviation_bpm": row["hr_mean_bpm__60m"] - reference
                    if reference is not None
                    else np.nan,
                    "hr_latest_deviation_bpm": row["hr_latest_bpm__60m"] - reference
                    if reference is not None
                    else np.nan,
                }
            )
            quality.append({**meta, **q})
            provenance.append({**meta, "directory": directory, **lin})
    return pd.DataFrame(outputs), pd.DataFrame(quality), pd.DataFrame(provenance)
