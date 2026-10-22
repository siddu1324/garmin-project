"""Validated saved inputs, explicit cohorts and no raw sensor extraction."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_binary_6hour.run import data as old_data
from garmin_preprocessing.common import verify_bundle
from .design import blocks


@lru_cache(maxsize=1)
def data(root):
    root = Path(root)
    cfg = json.loads((root / "config/binary_improvement.json").read_text())
    f, _, _ = old_data(root)
    f = f.copy()
    entries = json.loads((root / cfg["feature_bundle"] / "dictionary.json").read_text())
    b = blocks(entries)
    q = pd.read_parquet(
        root / cfg["feature_bundle"] / "quality.parquet", filters=[("sensor", "=", "hr")]
    )
    q360 = q[q.lookback_minutes.eq(360)].set_index("n_ema_id").reindex(f.n_ema_id)
    f["supported"] = (
        (q360.occupied_fraction >= cfg["support_occupied_fraction"])
        & (q360.latest_age_seconds <= cfg["support_latest_age_seconds"])
    ).to_numpy()
    f["supported"] &= f[b["original360"]].notna().all(axis=1)
    verify_bundle(root / cfg["morning_bundle"])
    wake = pd.read_parquet(root / cfg["morning_bundle"] / "features.parquet")
    assert wake.n_partition.eq("development").all()
    for short, label in [
        ("nominal", "reported_minute_start"),
        ("later", "reported_minute_end_sensitivity"),
    ]:
        w = wake[wake.scenario.eq(label)].set_index("n_ema_id").reindex(f.n_ema_id)
        for source, new in zip(
            [
                "hr_mean_bpm__wake",
                "hr_sd_bpm__wake",
                "hr_latest_bpm__wake",
                "hr_slope_bpm_per_minute__wake",
            ],
            b["wake"],
            strict=True,
        ):
            f[f"{new}_{short}"] = w[source].to_numpy()
        known = w.n_participant.notna().to_numpy()
        assert np.array_equal(f.loc[known, "n_participant"], w.loc[known, "n_participant"])
        np.testing.assert_allclose(
            f.loc[known, "pss_combo"], w.loc[known, "pss_combo"], equal_nan=True
        )
        f[f"retro_{short}"] = (
            known
            & w.candidate_available.eq(True).to_numpy(bool)
            & w[
                [
                    "hr_mean_bpm__wake",
                    "hr_sd_bpm__wake",
                    "hr_latest_bpm__wake",
                    "hr_slope_bpm_per_minute__wake",
                ]
            ]
            .notna()
            .all(axis=1)
            .to_numpy()
            & f[b["original60"]].notna().all(axis=1).to_numpy()
        )
    for name, cols in b.items():
        if name != "wake":
            assert set(cols) <= set(f), name
    return f, b, cfg


def cohort(frame, spec):
    keep = frame.pss_combo.notna() & frame.context_available
    if spec["cohort"] != "all":
        keep &= frame[spec["cohort"]]
    f = frame[keep].copy()
    if spec["cohort"].startswith("retro_"):
        scenario = spec["cohort"].split("_")[1]
        for s in ["mean", "sd", "latest", "slope"]:
            f[f"retrospective__{s}"] = f[f"retrospective__{s}_{scenario}"]
    return f
