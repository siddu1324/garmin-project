"""Independent arithmetic, estimator replays, immutable inputs and separation."""

import json
import csv
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from garmin_comprehensive.extract import REFERENCE
from garmin_inventory.privacy import sha256
from garmin_inventory.sources import discover
from garmin_preprocessing.common import verify_bundle, write_json
from .data import HR, STEP
from .models import fit_model


def verify(root):
    root = Path(root)
    ptr = json.loads((root / "private/root_cause_analysis/current.json").read_text())
    data, run = root / ptr["data"], root / ptr["run"]
    verify_bundle(data)
    cfg = json.loads((run / "identity.json").read_text())["config"]
    f = pd.read_parquet(data / "development.parquet")
    a = pd.read_parquet(root / "private/frozen_participant_assignments.parquet")
    mapping = a.set_index("participant")
    assert f.n_participant.nunique() == 125 and f.n_ema_id.is_unique and len(f) == 3706
    assert f.n_partition.eq("development").all()
    assert np.array_equal(f.n_participant.map(mapping.fold), f.n_fold)
    probe = pd.read_parquet(data / "activity_probe.parquet")
    assert set(probe.n_participant) <= set(f.n_participant)
    assert probe.n_fold.eq(probe.n_participant.map(mapping.fold)).all()
    assert not probe.duplicated(["n_participant", "n_start", "n_end"]).any()
    assert probe.n_end.sub(probe.n_start).eq(300).all()
    assert probe.loc[probe.eligible, "supported_seconds"].ge(240).all()
    assert probe.loc[probe.eligible, "hr_gap_seconds"].le(90).all()
    for person, g in probe.groupby("n_participant"):
        c = f.loc[f.n_participant.eq(person), "n_cutoff_epoch"].to_numpy()
        assert all(((c >= r.n_end) & (c - 3600 <= r.n_start)).any() for r in g.itertuples())
        assert g.sort_values("n_start").n_start.diff().dropna().ge(300).all()
    replays = inner = fresh = 0
    memberships = {}
    for path in sorted((run / "jobs").glob("*/summary.json")):
        verify_bundle(path.parent)
        info = json.loads(path.read_text())
        if info["status"] != "complete":
            continue
        target, kind, spec, fold = info["target"], info["kind"], info["spec"], info["fold"]
        g = probe[probe.eligible] if target == "target" else f[f.matched]
        train, valid = g[g.n_fold.ne(fold)], g[g.n_fold.eq(fold)]
        assert not set(train.n_participant) & set(valid.n_participant)
        fitted = joblib.load(path.parent / "estimator.joblib")
        assert set(spec["columns"]) <= set(HR + [STEP])
        assert all(
            n in HR + [STEP, "local_time_sin", "local_time_cos"]
            or n.startswith("n_survey_category_")
            for n in fitted.transform.names
        )
        base = fitted.transform.base
        pd.testing.assert_series_equal(base.medians, train[base.kept].median())
        if base.scaler is not None:
            np.testing.assert_allclose(base.scaler.mean_, train[base.kept].mean())
        with threadpool_limits(limits=1):
            for role, z in [("train", train), ("validation", valid)]:
                p = pd.read_parquet(path.parent / f"{role}.parquet")
                assert p.n_ema_id.tolist() == z.n_ema_id.tolist()
                np.testing.assert_array_equal(p.target, z[target])
                saved = (
                    p[[f"p{k}" for k in range(1, 6)]].to_numpy()
                    if kind == "ordinal"
                    else p["probability" if kind == "binary" else "prediction"]
                )
                np.testing.assert_allclose(saved, fitted.predict(z), atol=1e-12, rtol=1e-12)
            if (
                fold == 0
                and spec["name"] == "context_hr"
                and target in ["target_nervous", "ordinal_nervous", "pss_combo"]
            ):
                fit, _, _ = fit_model(train, target, spec, kind, cfg)
                np.testing.assert_allclose(
                    fit.predict(valid), fitted.predict(valid), atol=1e-12, rtol=1e-12
                )
                fresh += 1
        if (path.parent / "inner_membership.parquet").exists():
            m = pd.read_parquet(path.parent / "inner_membership.parquet")
            key = (target, fold)
            if key in memberships:
                pd.testing.assert_frame_equal(m, memberships[key])
            else:
                memberships[key] = m
            for _, part in m.groupby("inner"):
                x, y = part[part.role.eq("train")], part[part.role.eq("validation")]
                assert not set(x.n_participant) & set(y.n_participant)
                assert set(x.n_ema_id) | set(y.n_ema_id) == set(train.n_ema_id)
                inner += 1
        replays += 1
    # Independent counter-pair mask reconstruction, preserving rejected endpoints.
    directories = json.loads(
        (root / cfg["normalized_bundle"] / "participant_directories.json").read_text()
    )
    counter_rows = counter_values = 0
    for person in [p for p in directories if p["has_garmin"]][:10]:
        directory = person["directory"]
        logical = root / cfg["normalized_bundle"] / "logical" / directory / "steps.parquet"
        if not logical.exists():
            continue
        z = pd.read_parquet(logical)
        saved = pd.read_parquet(
            root / cfg["ledger_bundle"] / "counter_pairs" / directory / "steps.parquet"
        )
        rows = []
        for _, part in z[np.isfinite(z.n_epoch_s)].groupby("n_epoch_s", sort=True):
            r = part.iloc[0]
            good = (
                part.n_value.nunique(dropna=False) == 1
                and not part.q_conflicting_timestamp_value.any()
                and not part[
                    [
                        "q_invalid_epoch",
                        "q_invalid_text_clock",
                        "q_clock_discrepancy",
                        "q_unusual_offset",
                    ]
                ]
                .any()
                .any()
                and all(np.isfinite(v) and v >= 0 and v == int(v) for v in part.n_value)
            )
            rows.append((r, bool(good)))
        assert len(rows) == len(saved)
        for j, ((r, good), s) in enumerate(zip(rows, saved.itertuples())):
            mask = 1 if j == 0 else 0
            prev, pg = rows[j - 1] if j else (None, False)
            if not (good and pg):
                mask |= 2
            if prev is None or r.n_text_local.date() != prev.n_text_local.date():
                mask |= 4
            if prev is None or r.n_inferred_offset_minutes != prev.n_inferred_offset_minutes:
                mask |= 8
            if prev is None or r.n_source_sha256 != prev.n_source_sha256:
                mask |= 16
            if prev is None or not r.n_epoch_s > prev.n_epoch_s:
                mask |= 32
            if prev is not None and r.n_epoch_s - prev.n_epoch_s > 300:
                mask |= 64
            if prev is not None and r.n_value - prev.n_value < 0:
                mask |= 128
            assert mask == s.a_pair_rejection_mask and (mask == 0) == s.a_pair_accepted
            counter_rows += 1
        ema = f[f.n_participant.eq(person["participant"])].sort_values("n_ema_id").iloc[-1]
        for m in [5, 15, 30, 60, 120, 360, 720, 1440]:
            seconds = added = 0.0
            for p in saved.itertuples():
                if (
                    p.a_pair_accepted
                    and ema.n_cutoff_epoch - m * 60 <= p.a_pair_start
                    and p.n_epoch_s < ema.n_cutoff_epoch
                ):
                    seconds += p.a_pair_duration
                    added += p.a_pair_delta
            rate = 60 * added / seconds if seconds else np.nan
            np.testing.assert_allclose(rate, ema[f"steps__supported_rate__{m}m"], equal_nan=True)
            counter_values += 1
    # Close one complete ORIGINAL sensor + ORIGINAL answer -> this fold's prediction chain.
    trace = pd.read_parquet(data / "private_source_trace.parquet")
    keys = sorted(set(trace.n_ema_id) & set(f.loc[f.matched, "n_ema_id"]))
    assert keys
    chosen = f[f.n_ema_id.eq(keys[0])]
    original = pd.read_parquet(root / cfg["normalized_bundle"] / "ema/development_ema.parquet")
    row = original[original.n_ema_id.eq(keys[0])].iloc[0]
    source = discover(root / "data")[0][int(row.n_source_id.removeprefix("ema_"))]
    assert source.content_hash() == row.n_source_sha256
    found = False
    with source.open() as stream:
        reader = csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
        header = next(reader)
        for number, values in enumerate(reader, 1):
            if number != row.n_source_row:
                continue
            raw = dict(zip(header, values))
            assert raw["mbl_cod"].strip() == chosen.n_participant.iloc[0]
            for key in [
                "PSS_NERVOUS",
                "PSS_COPE",
                "PSS_CONTROL",
                "actual_start_local",
                "timezone_offset",
            ]:
                assert raw[key] == row[key]
            score = (
                float(raw["PSS_NERVOUS"]) + 12 - float(raw["PSS_COPE"]) - float(raw["PSS_CONTROL"])
            )
            assert score == chosen.pss_combo.iloc[0]
            found = True
            break
    assert found
    fold = int(chosen.n_fold.iloc[0])
    job = run / "jobs" / f"binary__target_nervous__context_hr__fold{fold}"
    model = joblib.load(job / "estimator.joblib")
    x = model.transform.transform(chosen)[0]
    linear = float(model.model.intercept_[0] + sum(a * b for a, b in zip(x, model.model.coef_[0])))
    expected_probability = 1 / (1 + np.exp(-linear))
    saved = pd.read_parquet(job / "validation.parquet")
    saved = saved[saved.n_ema_id.eq(keys[0])]
    np.testing.assert_allclose(saved.probability.iloc[0], expected_probability, atol=1e-12)
    assert saved.target.iloc[0] == int(float(raw["PSS_NERVOUS"]) >= 3)
    before = json.loads((root / "private/root_cause_analysis/before.json").read_text())
    assert all(sha256(root / p) == h for p, h in before["files"].items())
    assert all(sha256(root / p) == h for p, h in before["manifests"].items())
    baseline = json.loads((root / "private/integrity_baseline.json").read_text())
    raw = 0
    for label, folder in [("data", root / "data"), ("reference", REFERENCE)]:
        for name, record in baseline[label].items():
            s = (folder / name).stat()
            assert s.st_size == record["size"] and s.st_mtime_ns == record["mtime_ns"]
            raw += 1
    report = dict(
        training_replays=replays,
        validation_replays=replays,
        independent_fresh_fits=fresh,
        inner_separation_checks=inner,
        counter_pair_mask_checks=counter_rows,
        counter_feature_checks=counter_values,
        complete_original_to_prediction_traces=1,
        raw_metadata_checks=raw,
        prior_files_preserved=len(before["files"]),
        prior_manifests_preserved=len(before["manifests"]),
        all_responses=len(f),
        final_test_evaluated=False,
        new_confirmed_defects=[],
    )
    output = root / "reports/root_cause_analysis" / run.name / "verification.json"
    if (output.parent / "manifest.json").exists():
        assert json.loads(output.read_text()) == report
    else:
        write_json(output, report)
    return report


if __name__ == "__main__":
    print(json.dumps(verify(Path.cwd()), indent=2))
