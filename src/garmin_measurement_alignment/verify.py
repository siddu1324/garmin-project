"""Independent arithmetic, stored-input provenance and preservation checks."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from garmin_comprehensive.extract import REFERENCE
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import verify_bundle, write_json
from garmin_preprocessing.eligibility import window_ends
from garmin_root_cause.preservation import completed as previous_completed


def local_parse(text):
    for fmt in ["%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError("Unrecognized clock format; values withheld")


def verify(root):
    root = Path(root)
    base = root / "private/measurement_alignment"
    ptr = json.loads((base / "current.json").read_text())
    run = root / ptr["run"]
    out = root / ptr["public"]
    identity = json.loads((run / "identity.json").read_text())
    cfg = identity["config"]
    for key, h in identity["input_manifests"].items():
        assert sha256(root / cfg[key] / "manifest.json") == h
        verify_bundle(root / cfg[key])
    for name, h in identity["code"].items():
        assert sha256(root / "src/garmin_measurement_alignment" / name) == h
    assert previous_completed(root)
    ema = pd.read_parquet(root / cfg["normalized_bundle"] / "ema/development_ema.parquet")
    dev = pd.read_parquet(root / cfg["development_bundle"] / "development.parquet")
    assignments = pd.read_parquet(
        root / "private/frozen_participant_assignments.parquet"
    ).set_index("participant")
    assert (
        len(ema) == len(dev) == 3706
        and dev.n_ema_id.is_unique
        and dev.n_partition.eq("development").all()
    )
    assert (
        dev.n_participant.nunique() == 125
        and dev.n_fold.eq(dev.n_participant.map(assignments.fold)).all()
    )
    timing = pd.read_parquet(run / "timing.parquet").set_index("n_ema_id")
    for r in ema.itertuples():
        actual = local_parse(r.actual_start_local)
        scheduled = local_parse(r.scheduled_start_local)
        utc = (actual - timedelta(minutes=float(r.timezone_offset))).replace(tzinfo=timezone.utc)
        cutoff = utc - timedelta(seconds=60 if r.n_start_precision == "minute" else 0)
        assert r.n_cutoff_epoch == cutoff.timestamp()
        assert timing.loc[r.n_ema_id, "delay_minutes"] == (actual - scheduled).total_seconds() / 60
    # Original corrected engine against Python datetime, including both old overflow sites and 24h edge.
    boundaries = 0
    for cutoff in [0.0, 1700000035.0]:
        for minutes in cfg["lookbacks_minutes"]:
            for k, end in enumerate(window_ends(cutoff, minutes)):
                expected = (
                    datetime.fromtimestamp(cutoff, timezone.utc) - timedelta(minutes=int(k))
                ).timestamp()
                assert (
                    end == expected
                    and end - (end - 300) == 300
                    and cutoff - int(minutes) * 60 <= end - 300 < end <= cutoff
                )
                boundaries += 1
    q = pd.read_parquet(root / cfg["feature_bundle"] / "quality.parquet")
    sensitivity = pd.read_parquet(run / "timing_sensitivity.parquet")
    baseline = (
        sensitivity[sensitivity.shift_seconds.eq(0)]
        .set_index(["n_ema_id", "lookback_minutes"])
        .sort_index()
    )
    old = q[q.sensor.eq("hr")].set_index(["n_ema_id", "lookback_minutes"]).sort_index()
    np.testing.assert_array_equal(baseline.accepted_readings, old.accepted_readings)
    np.testing.assert_allclose(baseline.occupied_fraction, old.occupied_fraction, equal_nan=True)
    # Deterministic independent boolean-mask check of every shift and lookback on ten responses.
    directories = json.loads(
        (root / cfg["normalized_bundle"] / "participant_directories.json").read_text()
    )
    mapping = {d["participant"]: d["directory"] for d in directories}
    checked = 0
    for fold in range(5):
        for r in dev[dev.n_fold.eq(fold)].sort_values("n_ema_id").head(2).itertuples():
            h = pd.read_parquet(
                root
                / cfg["ledger_bundle"]
                / "reading_dispositions"
                / mapping[r.n_participant]
                / "hr.parquet"
            )
            times = h.loc[h.a_first & h.a_acceptable, "n_epoch_s"].dropna().to_numpy(float)
            for minutes in cfg["lookbacks_minutes"]:
                for shift in cfg["sensor_clock_sensitivity_seconds"]:
                    shifted = times + shift
                    mask = (
                        (times < r.n_cutoff_epoch)
                        & (shifted >= r.n_cutoff_epoch - int(minutes) * 60)
                        & (shifted < r.n_cutoff_epoch)
                    )
                    bins = {
                        int((float(t) - (r.n_cutoff_epoch - int(minutes) * 60)) // 60)
                        for t in shifted[mask]
                    }
                    saved = sensitivity[
                        sensitivity.n_ema_id.eq(r.n_ema_id)
                        & sensitivity.lookback_minutes.eq(minutes)
                        & sensitivity.shift_seconds.eq(shift)
                    ].iloc[0]
                    assert (
                        int(mask.sum()) == saved.accepted_readings
                        and len(bins) / minutes == saved.occupied_fraction
                    )
                    checked += 1
    gate = pd.read_parquet(run / "recall_gate.parquet")
    assert gate.n_ema_id.tolist() == dev.n_ema_id.tolist() and gate.status.eq("BLOCKED").all()
    before = json.loads((base / "before.json").read_text())
    for group in ["files", "manifests"]:
        assert all(sha256(root / p) == h for p, h in before[group].items())
    baseline = json.loads((root / "private/integrity_baseline.json").read_text())
    raw = 0
    for label, folder in [("data", root / "data"), ("reference", REFERENCE)]:
        for name, r in baseline[label].items():
            stat = (folder / name).stat()
            assert stat.st_size == r["size"] and stat.st_mtime_ns == r["mtime_ns"]
            raw += 1
    result = dict(
        development_rows=len(dev),
        development_participants=125,
        all_ema_clock_delay_checks=len(ema),
        corrected_scalar_boundary_checks=boundaries,
        zero_shift_quality_checks=3706 * 8,
        independent_shift_support_checks=checked,
        raw_reference_metadata_checks=raw,
        prior_files_preserved=len(before["files"]),
        prior_manifests_preserved=len(before["manifests"]),
        previous_checkpoint_seal_verified=True,
        new_real_predictive_fits=0,
        final_test_evaluated=False,
    )
    p = out / "verification.json"
    if (out / "manifest.json").exists():
        assert json.loads(p.read_text()) == result
    else:
        write_json(p, result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(Path.cwd()), indent=2))
