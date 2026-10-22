"""Read immutable development sources; add two bounded cached-ledger summaries."""

import json
import tempfile
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd

from garmin_binary_6hour.run import data as prior_data
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle, write_json
from .history import build_history

RAW = [f"hr__{x}__60m" for x in ["mean", "sd", "latest", "slope"]] + ["steps__supported_rate__60m"]
REFERENCE = ["reference_mean_bpm"]
PERSONAL = REFERENCE + [
    "hr_mean_deviation_bpm",
    "hr_latest_deviation_bpm",
    "hr__sd__60m",
    "hr__slope__60m",
    "steps__supported_rate__60m",
    "hr__observed_minute_sd__60m",
    "hr__half_mean_change__60m",
    "multiscale__adjacent_mean_difference__60m",
    "interaction__hr_zero_step_mean__60m",
]
EXTRA = ["clock_reference_mean", "clock_mean_deviation", "step_block_rate_change"]
QUALITY = [
    f"quality__hr__{s}__60m"
    for s in [
        "accepted_readings",
        "occupied_fraction",
        "span_seconds",
        "maximum_gap_seconds",
        "latest_age_seconds",
    ]
]


def clock_reference(epochs, values, cutoff, cfg):
    start, end = cutoff - cfg["clock_reference_days"] * 86400, cutoff - 3600
    keep = (epochs >= start) & (epochs < end)
    t, v = epochs[keep], values[keep]
    phase = np.abs((t % 86400) - (cutoff % 86400))
    keep = np.minimum(phase, 86400 - phase) <= cfg["clock_reference_radius_minutes"] * 60
    t, v = t[keep], v[keep]
    minutes = pd.Series(v).groupby(np.floor(t / 60)).mean()
    dates = len(np.unique(np.floor(t / 86400)))
    ok = (
        len(minutes) >= cfg["clock_reference_min_minutes"]
        and dates >= cfg["clock_reference_min_dates"]
    )
    return (float(minutes.mean()) if ok else np.nan), dict(
        recorded_minutes=len(minutes),
        utc_dates=dates,
        start=start,
        end=end,
        selected_first=float(t.min()) if len(t) else None,
        selected_last=float(t.max()) if len(t) else None,
        reason="available" if ok else "insufficient_clock_matched_history",
    )


def block_rate(pairs, start, end, minimum):
    p = pairs[pairs.a_pair_accepted & pairs.a_pair_start.ge(start) & pairs.n_epoch_s.lt(end)]
    seconds = float(p.a_pair_duration.sum())
    return (float(60 * p.a_pair_delta.sum() / seconds) if seconds >= minimum else np.nan), seconds


@lru_cache(maxsize=1)
def data(root):
    root = Path(root)
    cfg = json.loads((root / "config/temporal_personalization.json").read_text())
    f, _, _ = prior_data(root)
    f = f.copy()
    source = root / cfg["normalized_bundle"] / "ema/development_ema.parquet"
    cols = [
        "n_ema_id",
        "n_participant",
        "n_partition",
        "n_fold",
        "n_actual_utc",
        "n_actual_local",
        "actual_start_local",
        "timezone_offset",
        "n_cutoff_epoch",
        "PSS_NERVOUS",
        "PSS_COPE",
        "PSS_CONTROL",
    ]
    e = pd.read_parquet(source, columns=cols).set_index("n_ema_id").loc[f.n_ema_id]
    assert e.n_partition.eq("development").all()
    assert np.array_equal(e.n_participant, f.n_participant) and np.array_equal(e.n_fold, f.n_fold)
    np.testing.assert_allclose(e.n_cutoff_epoch, f.n_cutoff_epoch)
    items = e[["PSS_NERVOUS", "PSS_COPE", "PSS_CONTROL"]].apply(pd.to_numeric, errors="coerce")
    valid = items.isin([1, 2, 3, 4, 5]).all(axis=1)
    score = (12 + items.PSS_NERVOUS - items.PSS_COPE - items.PSS_CONTROL).where(valid)
    np.testing.assert_allclose(score, f.pss_combo, equal_nan=True)
    for dest, col in zip(["item_nervous", "item_cope", "item_control"], items, strict=True):
        f[dest] = items[col].where(items[col].isin([1, 2, 3, 4, 5])).to_numpy()
    f["n_actual_epoch"] = e.n_actual_utc.astype("int64").to_numpy() / 1e9
    f["n_available_epoch"] = np.nan
    f["n_availability_verified"] = False
    quality = pd.read_parquet(root / cfg["feature_bundle"] / "quality.parquet")
    q = (
        quality[quality.sensor.eq("hr") & quality.lookback_minutes.eq(60)]
        .set_index("n_ema_id")
        .loc[f.n_ema_id]
    )
    for col in QUALITY:
        f[col] = q[col.split("__")[2]].to_numpy()
    f["supported60"] = (
        q.occupied_fraction.ge(cfg["support_occupied_fraction"])
        & q.latest_age_seconds.le(cfg["support_latest_age_seconds"])
    ).to_numpy() & f[RAW[:4]].notna().all(axis=1)
    return f, cfg


def build(root):
    root = Path(root)
    f, cfg = data(root)
    ledger = root / cfg["ledger_bundle"]
    verify_bundle(ledger)
    identity = dict(
        config=cfg,
        specification=sha256(root / "reports/temporal_personalization/specification.md"),
        sources={
            n: sha256(root / cfg[n] / "manifest.json")
            for n in ["feature_bundle", "normalized_bundle", "ledger_bundle"]
        },
        assignments=sha256(root / "private/frozen_participant_assignments.parquet"),
        source={
            p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("history.py")]
        },
    )
    dest = root / "private/temporal_personalization/features" / digest_object(identity)[:16]
    dest.parent.mkdir(exist_ok=True, parents=True)
    if dest.exists():
        verify_bundle(dest)
        return dest
    tmp = Path(tempfile.mkdtemp(prefix=".building_", dir=dest.parent))
    people = json.loads(
        (root / cfg["normalized_bundle"] / "participant_directories.json").read_text()
    )
    assert {p["participant"] for p in people} == set(f.n_participant)
    records = []
    hashes = {}
    for i, person in enumerate(people):
        hr = ledger / "reading_dispositions" / person["directory"] / "hr.parquet"
        step = ledger / "counter_pairs" / person["directory"] / "steps.parquet"
        hashes[str(hr.relative_to(root))] = sha256(hr)
        hashes[str(step.relative_to(root))] = sha256(step)
        g = pd.read_parquet(hr)
        if len(g):
            assert (
                g.n_partition.eq("development").all()
                and g.n_participant.eq(person["participant"]).all()
            )
        g = g[g.a_first & g.a_acceptable & np.isfinite(g.n_epoch_s)]
        t, v = g.n_epoch_s.to_numpy(float), g.n_value.to_numpy(float)
        pairs = pd.read_parquet(step)
        for r in f[f.n_participant.eq(person["participant"])].itertuples():
            c = r.n_cutoff_epoch
            ref, q = clock_reference(t, v, c, cfg)
            recent, qs = block_rate(pairs, c - 86400, c, cfg["step_block_min_supported_seconds"])
            earlier, qe = block_rate(
                pairs, c - 172800, c - 86400, cfg["step_block_min_supported_seconds"]
            )
            records.append(
                dict(
                    n_ema_id=r.n_ema_id,
                    clock_reference_mean=ref,
                    clock_mean_deviation=getattr(r, "hr__mean__60m") - ref,
                    step_block_rate_change=recent - earlier,
                    **{"q_clock_" + k: val for k, val in q.items()},
                    q_recent_step_seconds=qs,
                    q_earlier_step_seconds=qe,
                )
            )
        if (i + 1) % 25 == 0:
            print("Cached development partitions processed", i + 1, flush=True)
    f = f.merge(pd.DataFrame(records), on="n_ema_id", validate="one_to_one")
    history, lineage = build_history(f)
    f = f.merge(history, on="n_ema_id", validate="one_to_one")
    assert len(f) == 3706 and len(lineage) == 0
    f.to_parquet(tmp / "development.parquet", index=False)
    lineage.to_parquet(tmp / "ema_history_lineage.parquet", index=False)
    write_json(tmp / "source_hashes.json", hashes)
    write_json(tmp / "identity.json", identity)
    publish_bundle(tmp, dest, dict(identity=identity))
    write_json(
        root / "private/temporal_personalization/features_current.json",
        {"path": str(dest.relative_to(root))},
    )
    return dest


if __name__ == "__main__":
    print(build(Path.cwd()))
