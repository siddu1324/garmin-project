"""Versioned original-sheet completion using cached features and past-only HR partitions."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_ema_baseline.linkage import check_development
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import (
    digest_object,
    frozen_assignments,
    publish_bundle,
    verify_bundle,
    write_json,
)
from .extract import REFERENCE

HISTORY_METRICS = [
    "reference_mean",
    "reference_sd",
    "mean_deviation",
    "latest_deviation",
    "standardized_mean_deviation",
    "standardized_latest_deviation",
]


def history_name(metric, m):
    legacy = {
        "reference_mean": "reference_mean_bpm",
        "mean_deviation": "hr_mean_deviation_bpm",
        "latest_deviation": "hr_latest_deviation_bpm",
    }
    return legacy[metric] if m == 60 and metric in legacy else f"history__{metric}__{m}m"


def past_reference(g, cutoff, m, current_mean, current_latest):
    start, end = cutoff - (m + 1440) * 60, cutoff - m * 60
    t = g.n_epoch_s.to_numpy(float)
    a, b = np.searchsorted(t, [start, end])
    w = g.iloc[a:b]
    v = w[w.a_acceptable]
    ts, x = v.n_epoch_s.to_numpy(float), v.n_value.to_numpy(float)
    bins = np.floor(ts / 60).astype(np.int64)
    _, inverse = np.unique(bins, return_inverse=True)
    means = np.bincount(inverse, weights=x) / np.bincount(inverse) if len(ts) else np.array([])
    hours = len(np.unique(np.floor(ts / 3600)))
    out = dict.fromkeys(HISTORY_METRICS, np.nan)
    enough = len(means) >= 30 and hours >= 3
    if enough:
        mu, sd = means.mean(), means.std(ddof=1)
        out.update(
            reference_mean=mu,
            reference_sd=sd,
            mean_deviation=current_mean - mu,
            latest_deviation=current_latest - mu,
        )
        if sd > 0:
            out.update(
                standardized_mean_deviation=(current_mean - mu) / sd,
                standardized_latest_deviation=(current_latest - mu) / sd,
            )
    q = dict(
        accepted_readings=len(ts),
        occupied_minutes=len(means),
        occupied_fraction=len(means) / 1440,
        occupied_hours=hours,
        span_seconds=ts[-1] - ts[0] if len(ts) else np.nan,
        latest_age_seconds=cutoff - ts[-1] if len(ts) else np.nan,
        maximum_gap_seconds=np.diff(np.r_[start, ts, end]).max(),
        history_status="available" if enough else "insufficient_history",
        reference_start=start,
        reference_end=end,
    )
    return out, q


def augment(root, base):
    root, base = Path(root).resolve(), Path(base).resolve()
    cfg = json.loads((root / "config/comprehensive_features.json").read_text())
    verify_bundle(base)
    assignments, dev = frozen_assignments(root, REFERENCE)
    frame = pd.read_parquet(base / "development_features.parquet")
    check_development(frame, assignments, 3706)
    original = json.loads((root / "private/comprehensive_features/original_sheet.json").read_text())
    identity = dict(
        base_manifest=sha256(base / "manifest.json"),
        config=cfg,
        code=sha256(Path(__file__)),
        specification=sha256(root / "reports/comprehensive_features/specification.md"),
        original_sheet=original["source_sha256"],
        original_family_rows=original["family_rows"],
    )
    dest = root / "private/comprehensive_features/features" / digest_object(identity)[:16]
    if dest.exists():
        verify_bundle(dest)
        return dest
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=dest.parent))
    normalized = root / cfg["normalized_bundle"]
    old = root / cfg["feature_bundle"]
    people = json.loads((normalized / "participant_directories.json").read_text())
    assert {p["participant"] for p in people} == dev
    additions, qs = [], []
    agreement = 0
    for person in people:
        ledger = pd.read_parquet(old / "reading_dispositions" / person["directory"] / "hr.parquet")
        if len(ledger):
            assert (
                ledger.n_participant.eq(person["participant"]).all()
                and ledger.n_partition.eq("development").all()
            )
            assert ledger.n_fold.eq(person["fold"]).all()
        g = ledger[ledger.a_first & np.isfinite(ledger.n_epoch_s)].copy()
        e = frame[frame.n_participant.eq(person["participant"])]
        for row in e.itertuples(index=False):
            output = {"n_ema_id": row.n_ema_id}
            for m in cfg["lookbacks_minutes"]:
                end = float(row.n_cutoff_epoch)
                if not np.isfinite(end) or row.n_invalid_clock:
                    continue
                values, q = past_reference(
                    g, end, m, getattr(row, f"hr__mean__{m}m"), getattr(row, f"hr__latest__{m}m")
                )
                for metric, value in values.items():
                    name = history_name(metric, m)
                    if name in frame:
                        assert np.isclose(value, getattr(row, name), equal_nan=True)
                        agreement += 1
                    else:
                        output[name] = value
                qs.append(
                    dict(
                        n_ema_id=row.n_ema_id,
                        sensor="history",
                        lookback_minutes=m,
                        **q,
                        reasons_json=json.dumps(
                            {
                                k: "available"
                                if np.isfinite(v)
                                else "insufficient_history_or_current_support_or_zero_reference_sd"
                                for k, v in values.items()
                            }
                        ),
                    )
                )
            additions.append(output)
    frame = frame.merge(pd.DataFrame(additions), on="n_ema_id", validate="one_to_one")
    entries = [
        d for d in json.loads((base / "dictionary.json").read_text()) if d["modality"] != "history"
    ]
    for m in cfg["lookbacks_minutes"]:
        for metric in HISTORY_METRICS:
            entries.append(
                dict(
                    name=history_name(metric, m),
                    modality="history",
                    metric=metric,
                    lookback_minutes=m,
                    units="dimensionless" if metric.startswith("standardized") else "bpm",
                    formula="Reference mean/sample SD of equally weighted occupied UTC-minute means in [c-(L+1440)m,c-Lm); current mean/latest minus reference, optionally /reference SD",
                    requirements=">=30 occupied minutes in >=3 UTC-hour bins; standardized deviation requires reference SD>0 and current input",
                    missing="NaN if support/current value unavailable or denominator zero",
                    source="accepted HR ledger; strictly preceding current window",
                    status="supported_observed_summary",
                    limitation="Earlier-recording reference; not resting, complete daily or full-person baseline",
                )
            )
    frame.to_parquet(build / "development_features.parquet", index=False)
    q = pd.concat([pd.read_parquet(base / "quality.parquet"), pd.DataFrame(qs)], ignore_index=True)
    q.to_parquet(build / "quality.parquet", index=False)
    # Existing source-row slices are preserved; history stores explicit disjoint interval bounds in quality.
    for name in ["lineage.parquet", "source_partitions.json"]:
        (build / name).write_bytes((base / name).read_bytes())
    write_json(build / "dictionary.json", entries)
    coverage = []
    for d in entries:
        for survey in ["All", "Morning", "Midday", "Endday"]:
            f = frame if survey == "All" else frame[frame.n_survey_category.eq(survey)]
            ok = f[d["name"]].notna()
            coverage.append(
                dict(
                    feature=d["name"],
                    modality=d["modality"],
                    lookback_minutes=d["lookback_minutes"],
                    survey=survey,
                    responses=len(f),
                    available=int(ok.sum()),
                    missing=int((~ok).sum()),
                    available_participants=f.loc[ok, "n_participant"].nunique(),
                    missing_fraction=float((~ok).mean()),
                )
            )
    write_json(build / "coverage.json", coverage)
    summary = json.loads((base / "summary.json").read_text())
    summary.update(
        feature_count=len(entries),
        history_legacy_agreement_checks=agreement,
        count_by_modality=pd.Series([d["modality"] for d in entries]).value_counts().to_dict(),
        count_by_lookback=pd.Series([str(d["lookback_minutes"]) for d in entries])
        .value_counts()
        .to_dict(),
        entirely_missing=[d["name"] for d in entries if not frame[d["name"]].notna().any()],
        complete_all_supported=int(
            frame[[d["name"] for d in entries if d["modality"] != "pulseox"]]
            .notna()
            .all(axis=1)
            .sum()
        ),
    )
    write_json(build / "summary.json", summary)
    write_json(build / "original_sheet.json", original)
    publish_bundle(build, dest, dict(inputs=identity))
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base")
    args = p.parse_args()
    print(augment(Path.cwd(), args.base))
