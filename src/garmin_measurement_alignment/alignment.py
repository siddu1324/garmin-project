"""Recall construction and observed features under an approved event-time contract."""

import math
import numpy as np
import pandas as pd

from garmin_root_cause.data import independent_hr, interval_support
from garmin_temporal_personalization.history import build_history
from .metadata import MetadataError


def reconstruct(ema, events, contract):
    """Do not replace a prompt with a response; reject uncertainty about the last event."""
    cutoff = float(ema["n_cutoff_epoch"])
    person = ema["n_participant"]
    if ema["n_partition"] != "development" or ema["n_fold"] not in range(5):
        raise MetadataError("nondevelopment_alignment")
    scope = "wake" if ema["n_survey_category"] == "Morning" else "prompt"
    eligible_type = "wake_observed" if scope == "wake" else "prompt_delivered"
    clock_name = "wake" if scope == "wake" else "delivered"
    selected = [e for e in events if e["participant_id"] == person]
    events_of_type = [e for e in selected if e["event_type"] == eligible_type]
    # A current prompt may or may not count, depending on approved wording; never guess.
    if scope == "prompt" and contract.get("recall_prompt_rule") == "exclude_current_prompt":
        current = ema.get("provider_prompt_id")
        if not current:
            return dict(status="BLOCKED", reason="current_prompt_key_missing")
        events_of_type = [e for e in events_of_type if e["prompt_id"] != current]
    if scope == "wake":
        # Provider date is independently verified, never guessed from first HR or current wake answer.
        date = ema.get("verified_local_date")
        if not date:
            return dict(status="BLOCKED", reason="verified_waking_date_missing")
        events_of_type = [e for e in events_of_type if e.get("verified_local_date") == date]
    before = [e for e in events_of_type if e["parsed_clocks"][clock_name]["lower"] < cutoff]
    if not before:
        return dict(status="BLOCKED", reason=f"no_verified_{scope}_event")
    # Delayed receipt of the latest event cannot cause fallback to an older event.
    before.sort(key=lambda e: e["parsed_clocks"][clock_name]["lower"])
    last = before[-1]
    bounds = last["parsed_clocks"][clock_name]
    if bounds["upper"] >= cutoff:
        return dict(status="BLOCKED", reason="boundary_uncertainty_reaches_cutoff")
    if last["parsed_clocks"]["received"]["upper"] >= cutoff:
        return dict(status="BLOCKED", reason="latest_boundary_not_available")
    if len(before) > 1 and before[-2]["parsed_clocks"][clock_name]["upper"] >= bounds["lower"]:
        return dict(status="BLOCKED", reason="ambiguous_last_event_order")
    coverage = [
        e
        for e in selected
        if e["event_type"] == "coverage"
        and e["scope"] == scope
        and e["complete"]
        and e["parsed_clocks"]["coverage_start"]["upper"] <= bounds["lower"]
        and e["parsed_clocks"]["coverage_end"]["lower"] >= cutoff
    ]
    if not coverage:
        return dict(status="BLOCKED", reason="complete_event_log_unverified")
    # Use latest plausible boundary; retain its entire uncertainty interval separately.
    start = bounds["upper"]
    return dict(
        status="RECONSTRUCTIBLE",
        reason="verified_conservative_boundary",
        start=start,
        end=cutoff,
        boundary_lower=bounds["lower"],
        boundary_upper=bounds["upper"],
        duration_seconds=cutoff - start,
        boundary_event_id=last["event_id"],
        scope=scope,
        boundary_uncertainty_seconds=bounds["precision"],
    )


def measurements(events, person, start, cutoff):
    rows = []
    for e in events:
        if e["participant_id"] != person or e["event_type"] != "sensor_measurement":
            continue
        t = e["parsed_clocks"]["acquired"]
        # Known before cutoff AND wholly pre-cutoff. No reliance on future delivery.
        if not (t["lower"] >= start and t["lower"] < cutoff):
            continue
        v = e["value"]
        accepted = e["quality"] == "accepted" and e["wear"] == "worn"
        accepted &= t["upper"] < cutoff and e["parsed_clocks"]["received"]["upper"] < cutoff
        accepted &= 30 <= v <= 240 if e["sensor"] == "hr" else v >= 0 and float(v).is_integer()
        rows.append(
            dict(
                sensor=e["sensor"],
                t=t["lower"],
                upper=t["upper"],
                value=v,
                accepted=bool(accepted),
                event_id=e["event_id"],
                counter_epoch=e.get("counter_epoch"),
                previous_event_id=e.get("previous_event_id"),
                source_sha256=e["source_sha256"],
                source_row=e["source_row"],
            )
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "sensor",
            "t",
            "upper",
            "value",
            "accepted",
            "event_id",
            "counter_epoch",
            "previous_event_id",
            "source_sha256",
            "source_row",
        ],
    )
    out = []
    for (_, time), g in frame.groupby(["sensor", "t"], sort=True):
        row = g.iloc[0].to_dict()
        row["accepted"] = bool(
            g.accepted.all() and g.value.nunique() == 1 and g.upper.nunique() == 1
        )
        row["conflict"] = g.value.nunique() > 1 or g.upper.nunique() > 1
        row["duplicate_count"] = len(g) - 1
        out.append(row)
    return pd.DataFrame(out, columns=list(frame) + ["conflict", "duplicate_count"])


def aligned_features(ema, events, contract):
    boundary = reconstruct(ema, events, contract)
    if boundary["status"] != "RECONSTRUCTIBLE":
        return boundary, {}, {}
    start, end = boundary["start"], boundary["end"]
    m = measurements(events, ema["n_participant"], start, end)
    h = m[(m.sensor == "hr") & m.accepted.astype(bool)].sort_values("t")
    # Overlapping time uncertainty cannot be resolved by arbitrary ordering.
    if len(h) > 1 and (h.upper.to_numpy()[:-1] >= h.t.to_numpy()[1:]).any():
        return {**boundary, "status": "BLOCKED", "reason": "ambiguous_hr_time_order"}, {}, {}
    t = h.t.to_numpy(float)
    v = h.value.to_numpy(float)
    features = {"aligned_" + k: val for k, val in independent_hr(t, v).items()}
    p = m[m.sensor == "steps"].sort_values("t")
    intervals = []
    added = 0.0
    pairs = 0
    # Rejected/uncertain original endpoints stay in sequence; never filter them before differencing.
    for a, b in zip(p.iloc[:-1].itertuples(), p.iloc[1:].itertuples()):
        dt = b.t - a.t
        if not (
            a.accepted
            and b.accepted
            and a.upper < b.t
            and 0 < dt <= 300
            and b.value >= a.value
            and a.counter_epoch == b.counter_epoch
            and a.source_sha256 == b.source_sha256
            and b.previous_event_id == a.event_id
        ):
            continue
        # Counter interval timing must be exact to assign its complete increment defensibly.
        if a.upper != a.t or b.upper != b.t:
            continue
        intervals.append((a.t, b.t))
        added += b.value - a.value
        pairs += 1
    support, gap = interval_support(intervals, start, end)
    features["aligned_steps_rate"] = 60 * added / support if support else np.nan
    possible = math.ceil((end - start) / 60)
    occupied = len(np.unique(np.floor((t - start) / 60)))
    quality = dict(
        accepted_hr=len(t),
        recorded_hr=int(m.sensor.eq("hr").sum()),
        occupied_minutes=occupied,
        occupied_fraction=occupied / possible,
        largest_gap_seconds=float(np.diff(np.r_[start, t, end]).max()),
        freshness_seconds=float(end - t[-1]) if len(t) else np.nan,
        supported_activity_seconds=support,
        supported_activity_fraction=support / (end - start),
        unsupported_activity_gap_seconds=gap,
        supported_activity_pairs=pairs,
    )
    return boundary, features, quality


def versioned_history(development, events, current):
    """Latest submitted version known at this cutoff; later edits never leak backwards."""
    cutoff = current["n_cutoff_epoch"]
    versions = {}
    known = development.set_index("n_ema_id")
    for e in events:
        if (
            e["event_type"] != "answer_submitted"
            or e["participant_id"] != current["n_participant"]
            or e["ema_id"] == current["n_ema_id"]
        ):
            continue
        if e["ema_id"] not in known.index:
            raise MetadataError("history_key_missing")
        base = known.loc[e["ema_id"]]
        if base.n_participant != e["participant_id"]:
            raise MetadataError("history_identity_mismatch")
        c = e["parsed_clocks"]
        if (
            c["received"]["upper"] >= cutoff
            or c["submitted"]["upper"] >= cutoff
            or base.n_actual_epoch >= cutoff
        ):
            continue
        old = versions.get(e["ema_id"])
        if old is None or e["answer_version"] > old["answer_version"]:
            versions[e["ema_id"]] = e
    rows = []
    for key, e in versions.items():
        r = known.loc[key].copy()
        i = e["items"]
        r["item_nervous"], r["item_cope"], r["item_control"] = [
            i.get(k) if i.get(k) is not None else np.nan
            for k in ["PSS_NERVOUS", "PSS_COPE", "PSS_CONTROL"]
        ]
        r["pss_combo"] = 12 + r.item_nervous - r.item_cope - r.item_control
        r["n_available_epoch"] = e["parsed_clocks"]["received"]["upper"]
        r["n_availability_verified"] = True
        r["n_ema_id"] = key
        rows.append(r)
    current_row = pd.Series(current).copy()
    # Current labels play no role in history construction, even if present in the caller's row.
    current_row["pss_combo"] = np.nan
    current_row["n_available_epoch"] = np.nan
    current_row["n_availability_verified"] = False
    frame = pd.DataFrame([*rows, current_row])
    history, lineage = build_history(frame)
    return history[history.n_ema_id.eq(current["n_ema_id"])], lineage[
        lineage.n_ema_id.eq(current["n_ema_id"])
    ]
