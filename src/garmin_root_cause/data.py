"""Frozen development inputs, conditional activity labels and strict recall gate."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_preprocessing.common import verify_bundle

HR = [f"hr__{s}__60m" for s in ["mean", "sd", "latest", "slope"]]
STEP = "steps__supported_rate__60m"
ITEMS = ["nervous", "cope_difficulty", "control_difficulty"]
TASKS = ITEMS + ["composite5", "composite10"]


def load(root):
    root = Path(root)
    cfg = json.loads((root / "config/root_cause_analysis.json").read_text())
    verify_bundle(root / cfg["development_bundle"])
    f = pd.read_parquet(root / cfg["development_bundle"] / "development.parquet")
    assert len(f) == 3706 and f.n_ema_id.is_unique and f.n_partition.eq("development").all()
    assert f.groupby("n_participant").n_fold.nunique().eq(1).all()
    for name, value in zip(
        ITEMS, [f.item_nervous, 6 - f.item_cope, 6 - f.item_control], strict=True
    ):
        f["ordinal_" + name] = value
        f["target_" + name] = value.ge(3).where(value.notna())
    for cut in [5, 10]:
        f[f"target_composite{cut}"] = f.pss_combo.ge(cut).where(f.pss_combo.notna())
    f["matched"] = f.pss_combo.notna() & f.context_available & f[HR + [STEP]].notna().all(axis=1)
    return f, cfg


def independent_hr(t, v):
    """Independent reference arithmetic; timestamp order and ties checked explicitly."""
    t, v = list(map(float, t)), list(map(float, v))
    assert all(t[i + 1] > t[i] for i in range(len(t) - 1))
    out = dict.fromkeys(HR, np.nan)
    if not t:
        return out
    out[HR[2]] = v[-1]
    if len(t) >= 2 and t[-1] - t[0] >= 1:
        mean = sum(v) / len(v)
        out[HR[0]] = mean
        out[HR[1]] = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
    if len(t) >= 3 and t[-1] - t[0] >= 60:
        x = [(s - t[0]) / 60 for s in t]
        xm, ym = sum(x) / len(x), sum(v) / len(v)
        out[HR[3]] = sum((a - xm) * (b - ym) for a, b in zip(x, v)) / sum(
            (a - xm) ** 2 for a in x
        )
    return out


def interval_support(intervals, start, end):
    """Union length and largest unsupported edge/internal gap; never double count."""
    merged = []
    for a, b in sorted(intervals):
        if not start <= a < b < end:
            raise ValueError("Only wholly contained original pairs may contribute.")
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    cursor, gap, support = start, 0.0, 0.0
    for a, b in merged:
        gap = max(gap, a - cursor)
        support += b - a
        cursor = b
    return support, max(gap, end - cursor)


def probe_window(hr, pairs, start, end, cfg):
    h = hr[hr.n_epoch_s.ge(start) & hr.n_epoch_s.lt(end)]
    h = h[h.a_first & h.a_acceptable]
    t, v = h.n_epoch_s.to_numpy(float), h.n_value.to_numpy(float)
    p = pairs[
        pairs.a_pair_accepted & pairs.a_pair_start.ge(start) & pairs.n_epoch_s.lt(end)
    ]
    support, gap = interval_support(zip(p.a_pair_start, p.n_epoch_s), start, end)
    hr_gap = float(np.diff(np.r_[start, t, end]).max())
    occupied = len(set(np.floor((t - start) / 60)))
    rate = 60 * float(p.a_pair_delta.sum()) / support if support else np.nan
    features = independent_hr(t, v)
    reasons = []
    if occupied < cfg["activity_min_hr_occupied_minutes"]:
        reasons.append("hr_occupancy")
    if hr_gap > cfg["activity_max_hr_gap_seconds"]:
        reasons.append("hr_gaps")
    if not np.isfinite(list(features.values())).all():
        reasons.append("hr_feature_support")
    if support < cfg["activity_min_supported_seconds"]:
        reasons.append("step_support")
    if gap > cfg["activity_max_unsupported_gap_seconds"]:
        reasons.append("step_gaps")
    target = (
        0 if rate == 0 else 1 if rate >= cfg["activity_active_steps_per_supported_minute"] else np.nan
    )
    if not np.isfinite(target):
        reasons.append("intermediate_or_unobserved_activity")
    return dict(
        **features, target=target, eligible=not reasons, reasons="|".join(reasons),
        supported_seconds=support, step_gap_seconds=gap, hr_gap_seconds=hr_gap,
        occupied_minutes=occupied, accepted_hr=len(t), step_rate=rate,
        hr_latest_age_seconds=float(end - t[-1]) if len(t) else np.nan,
    )


def build_probe(root, f, cfg):
    root = Path(root)
    mapping = json.loads((root / cfg["normalized_bundle"] / "participant_directories.json").read_text())
    rows = []
    for i, person in enumerate(mapping):
        ledger = root / cfg["ledger_bundle"]
        hr = pd.read_parquet(ledger / "reading_dispositions" / person["directory"] / "hr.parquet")
        pairs = pd.read_parquet(ledger / "counter_pairs" / person["directory"] / "steps.parquet")
        if len(hr):
            assert hr.n_partition.eq("development").all()
            assert hr.n_participant.eq(person["participant"]).all()
        # Fixed UTC grid; each source-time bin appears once per participant.
        ends = set()
        for cutoff in f.loc[f.n_participant.eq(person["participant"]), "n_cutoff_epoch"]:
            first = math.ceil((cutoff - 3600) / 300) * 300
            for start in range(first, math.floor(cutoff / 300) * 300, 300):
                ends.add(start + 300)
        for end in sorted(ends):
            start = end - 300
            record = probe_window(hr, pairs, start, end, cfg)
            rows.append(dict(
                n_ema_id=f"probe_{i}_{end}", n_participant=person["participant"],
                n_fold=person["fold"], n_survey_category="Activity probe", n_start=start,
                n_end=end, local_time_sin=0.0, local_time_cos=0.0, **record,
            ))
        if (i + 1) % 25 == 0:
            print("Development activity partitions", i + 1, flush=True)
    return pd.DataFrame(rows)


def recall_gate(row, contract):
    """No current-survey answers or guessed prompt histories can open this gate."""
    required = ["sensor_acquisition_verified", "utc_conversion_verified", "boundary_semantics_verified"]
    reasons = [k for k in required if contract.get(k) is not True]
    if row.get("boundary_source") == "current_survey_answer":
        reasons.append("retrospective_answer")
    if row.get("boundary_source") not in ["verified_wake_event", "verified_delivered_prompt"]:
        reasons.append("unverified_boundary_source")
    if row.get("boundary_source") == "verified_delivered_prompt" and contract.get("complete_prompt_log") is not True:
        reasons.append("incomplete_prompt_log")
    fields = [row.get(k, np.nan) for k in ["start_epoch", "available_epoch", "cutoff_epoch"]]
    if not all(isinstance(x, (float, int)) and np.isfinite(x) for x in fields):
        reasons.append("missing_clock")
    elif not fields[0] <= fields[1] < fields[2]:
        reasons.append("boundary_not_known_before_cutoff")
    return reasons


def aligned_features(hr, row, contract):
    reasons = recall_gate(row, contract)
    if reasons:
        raise ValueError("Recall evidence gate closed: " + ",".join(reasons))
    w = hr[
        hr.a_first & hr.a_acceptable & hr.n_epoch_s.ge(row["start_epoch"])
        & hr.n_epoch_s.lt(row["cutoff_epoch"])
    ]
    return independent_hr(w.n_epoch_s, w.n_value)
