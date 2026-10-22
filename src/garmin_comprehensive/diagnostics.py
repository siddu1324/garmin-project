"""Read-only interpretation diagnostics; no outcome-driven feature definitions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle, write_json
from garmin_inventory.privacy import sha256
import tempfile


def summarize(root):
    root = Path(root)
    current = json.loads((root / "private/comprehensive_features/current.json").read_text())
    fp = root / current["features"]
    verify_bundle(fp)
    data = pd.read_parquet(fp / "development_features.parquet")
    q = pd.read_parquet(fp / "quality.parquet")
    cfg = json.loads((root / "config/comprehensive_features.json").read_text())
    normalized = root / cfg["normalized_bundle"]
    old = root / cfg["feature_bundle"]
    identity = dict(features=sha256(fp / "manifest.json"), code=sha256(Path(__file__)))
    dest = root / "private/comprehensive_features/diagnostics" / digest_object(identity)[:16]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        verify_bundle(dest)
        return dest
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=dest.parent))
    zero = []
    for person in json.loads((normalized / "participant_directories.json").read_text()):
        for sensor, raw in [("hr", "hr"), ("pulseox", "pulseOx")]:
            path = (
                old / "reading_dispositions" / person["directory"] / "hr.parquet"
                if sensor == "hr"
                else normalized / "logical" / person["directory"] / f"{raw}.parquet"
            )
            if path.exists() and pq.read_metadata(path).num_rows:
                f = pd.read_parquet(
                    path, columns=["n_epoch_s", "n_value", "n_partition", "n_participant", "n_fold"]
                )
                if len(f):
                    assert (
                        f.n_partition.eq("development").all()
                        and f.n_participant.eq(person["participant"]).all()
                    )
                    assert f.n_fold.eq(person["fold"]).all()
                # Fractions of logical records, NOT estimated time at zero or physiological values.
                f = f[np.isfinite(f.n_epoch_s)]
                ts = f.n_epoch_s.to_numpy(float)
                vs = f.n_value.to_numpy(float)
            else:
                ts = np.array([])
                vs = np.array([])
            for r in data[data.n_participant.eq(person["participant"])].itertuples():
                for m in cfg["lookbacks_minutes"]:
                    a, b = np.searchsorted(ts, [r.n_cutoff_epoch - m * 60, r.n_cutoff_epoch])
                    values = vs[a:b]
                    zero.append(
                        dict(
                            n_ema_id=r.n_ema_id,
                            sensor=sensor,
                            lookback_minutes=m,
                            zero_logical_record_fraction=float((values == 0).mean())
                            if len(values)
                            else np.nan,
                            logical_records=len(values),
                        )
                    )
    z = pd.DataFrame(zero)
    z.to_parquet(build / "zero_quality.parquet", index=False)
    rows = []
    cols = [
        "accepted_readings",
        "accepted_fraction",
        "occupied_fraction",
        "span_seconds",
        "latest_age_seconds",
        "maximum_gap_seconds",
        "supported_seconds",
        "supported_pairs",
    ]
    for (sensor, m), g in q.groupby(["sensor", "lookback_minutes"]):
        for col in cols:
            x = g[col].dropna()
            if len(x):
                rows.append(
                    dict(
                        sensor=sensor,
                        lookback_minutes=int(m),
                        measure=col,
                        responses=len(x),
                        p10=float(x.quantile(0.1)),
                        median=float(x.median()),
                        p90=float(x.quantile(0.9)),
                    )
                )
    for (sensor, m), g in z.groupby(["sensor", "lookback_minutes"]):
        x = g.zero_logical_record_fraction.dropna()
        rows.append(
            dict(
                sensor=sensor,
                lookback_minutes=int(m),
                measure="zero_logical_record_fraction",
                responses=len(x),
                p10=float(x.quantile(0.1)),
                median=float(x.median()),
                p90=float(x.quantile(0.9)),
            )
        )
    # Repeated participants are the resampling units for explicitly retrospective associations.
    associations = []
    for col in [
        "hr__mean__60m",
        "hr__slope__60m",
        "reference_mean_bpm",
        "hr_mean_deviation_bpm",
        "steps__added_supported__60m",
        "pulseox__mean__60m",
    ]:
        f = data[["n_participant", "pss_combo", col]].dropna().copy()
        person = f.groupby("n_participant")[["pss_combo", col]].mean()
        f["center_y"] = f.pss_combo - f.n_participant.map(person.pss_combo)
        f["center_x"] = f[col] - f.n_participant.map(person[col])
        groups = [g for _, g in f.groupby("n_participant")]
        sufficient = np.array(
            [
                [
                    len(g),
                    g.center_x.sum(),
                    g.center_y.sum(),
                    (g.center_x**2).sum(),
                    (g.center_y**2).sum(),
                    (g.center_x * g.center_y).sum(),
                ]
                for g in groups
            ]
        )

        def corr(s):
            n, sx, sy, sxx, syy, sxy = np.moveaxis(s, -1, 0)
            den = np.sqrt(np.maximum(sxx - sx * sx / n, 0) * np.maximum(syy - sy * sy / n, 0))
            return np.divide(sxy - sx * sy / n, den, out=np.full_like(den, np.nan), where=den > 0)

        sample = np.random.default_rng(cfg["seed"]).integers(
            0, len(groups), size=(1000, len(groups))
        )
        low, high = np.nanquantile(corr(sufficient[sample].sum(axis=1)), [0.025, 0.975])
        associations.append(
            dict(
                feature=col,
                responses=len(f),
                participants=len(person),
                between_person_correlation=float(person[col].corr(person.pss_combo)),
                retrospective_within_person_correlation=float(corr(sufficient.sum(axis=0))),
                within_cluster_low=float(low),
                within_cluster_high=float(high),
                warning="Retrospective full-record centering for diagnosis only; never a predictor or validated prediction",
            )
        )
    write_json(build / "recording_support.json", rows)
    write_json(build / "retrospective_associations.json", associations)
    publish_bundle(build, dest, dict(inputs=identity))
    return dest


if __name__ == "__main__":
    print(summarize(Path.cwd()))
