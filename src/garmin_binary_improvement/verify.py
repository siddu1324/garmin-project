"""Independent replay, donor exclusion, provenance and preservation checks."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from garmin_binary_6hour.models import estimator, checked_fit
from garmin_comprehensive.extract import REFERENCE
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import verify_bundle, write_json
from .data import data, cohort
from .models import prepare, training_labels, threshold


def verify(root):
    root = Path(root)
    run = root / json.loads((root / "private/binary_improvement/current.json").read_text())["run"]
    out = root / "reports/binary_improvement" / run.name
    if (run / "manifest.json").exists():
        manifest = verify_bundle(run)
        assert all(sha256(run / p) == h for p, h in manifest["job_manifest_sha256"].items())
    frame, blocks, cfg = data(root)
    primary = pd.read_parquet(run / "primary_all_predictions.parquet")
    combined = pd.read_parquet(run / "all_predictions.parquet")
    order = ["task", "model", "n_ema_id"]
    pd.testing.assert_frame_equal(
        primary.sort_values(order).reset_index(drop=True),
        combined[combined.model.isin(primary.model.unique())]
        .sort_values(order)
        .reset_index(drop=True),
    )
    for filename, keys in [
        ("metrics.json", ["task", "model"]),
        ("paired.json", ["task", "model", "reference"]),
    ]:
        old = json.loads((run / "primary_stage_aggregates" / filename).read_text())
        current = {
            tuple(row[k] for k in keys): row for row in json.loads((out / filename).read_text())
        }
        assert all(current[tuple(row[k] for k in keys)] == row for row in old)
    original_items = (
        pd.read_parquet(
            root / "private/preprocessing/normalized/31e51c934bcdeffa/ema/development_ema.parquet",
            columns=["n_ema_id", "PSS_NERVOUS", "PSS_COPE", "PSS_CONTROL"],
        )
        .set_index("n_ema_id")
        .loc[frame.n_ema_id]
    )
    numeric_items = original_items.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid_items = np.isin(numeric_items, [1, 2, 3, 4, 5]).all(axis=1)
    independent_score = np.where(
        valid_items, 12 + numeric_items[:, 0] - numeric_items[:, 1] - numeric_items[:, 2], np.nan
    )
    np.testing.assert_allclose(independent_score, frame.pss_combo, equal_nan=True)
    identity = json.loads((run / "identity.json").read_text())
    assert identity["features"] == sha256(root / cfg["feature_bundle"] / "manifest.json")
    assert identity["previous"] == sha256(root / cfg["previous_run"] / "manifest.json")
    assert identity["assignments"] == sha256(
        root / "private/frozen_participant_assignments.parquet"
    )
    for name, h in identity["code"].items():
        assert sha256(Path(__file__).parent / name) == sha256(run / "executed_source" / name) == h
    assert all(sha256(root / p) == h for p, h in identity["reused_source"].items())
    replay = fresh = inner_checks = donor_checks = old_replays = nonconverged = 0
    failures = []
    controls = run / json.loads((run / "context_controls.json").read_text())["path"]
    cm = verify_bundle(controls)
    assert all(sha256(controls / p) == h for p, h in cm["job_manifest_sha256"].items())
    ci = json.loads((controls / "identity.json").read_text())
    assert ci["source"] == sha256(Path(__file__).parent / "controls.py")
    assert ci["amendment"] == sha256(
        root / "reports/binary_improvement/context_control_amendment.md"
    )
    assert all(sha256(Path(__file__).parent / p) == h for p, h in ci["fitting_source"].items())
    paths = sorted((run / "jobs").glob("*/summary.json"))
    control_paths = sorted((controls / "jobs").glob("*/summary.json"))
    assert len(paths) == 600 and len(control_paths) == 40
    matched_controls = candidate_evaluations = 0
    for path in paths + control_paths:
        verify_bundle(path.parent)
        info = json.loads(path.read_text())
        if info["status"] != "complete":
            failures.append(dict(task=info["task"], model=info["spec"]["name"], fold=info["fold"]))
            continue
        spec, task, fold = info["spec"], info["task"], info["fold"]
        if spec["name"].startswith("context_control_"):
            paired = run / "jobs" / path.parent.name.replace("context_control_", "context_compact_")
            for filename in [
                "inner_membership.parquet",
                "train_predictions.parquet",
                "validation_predictions.parquet",
            ]:
                a = pd.read_parquet(path.parent / filename)
                b = pd.read_parquet(paired / filename)
                columns = [
                    c
                    for c in ["n_ema_id", "n_participant", "inner_fold", "role", "target"]
                    if c in a
                ]
                pd.testing.assert_frame_equal(a[columns], b[columns])
            a = pd.read_parquet(path.parent / "augmentation_lineage.parquet")
            b = pd.read_parquet(paired / "augmentation_lineage.parquet")
            if len(a):
                pd.testing.assert_frame_equal(
                    a[["inner_fold", "base_ema_key"]], b[["inner_fold", "base_ema_key"]]
                )
            a = json.loads((path.parent / "resampling.json").read_text())
            b = json.loads((paired / "resampling.json").read_text())
            for left, right in zip(a, b, strict=True):
                for key in ["original_counts", "resampled_counts"]:
                    assert left[key] == right[key]
            matched_controls += 1
        f = cohort(frame, spec)
        train, valid = f[f.n_fold.ne(fold)], f[f.n_fold.eq(fold)]
        assert not set(train.n_participant) & set(valid.n_participant)
        fitted = joblib.load(path.parent / "estimator.joblib")
        t = fitted.transform.base
        pd.testing.assert_series_equal(t.medians, train[t.kept].median())
        if t.scaler is not None:
            np.testing.assert_allclose(
                t.scaler.mean_, train[t.kept].fillna(t.medians).to_numpy(float).mean(axis=0)
            )
        assert set(t.encoder.categories_[0]) == set(train.n_survey_category)
        for role, g in [("train", train), ("validation", valid)]:
            p = pd.read_parquet(path.parent / f"{role}_predictions.parquet")
            assert p.n_ema_id.tolist() == g.n_ema_id.tolist()
            assert np.array_equal(p.target, g[f"target_{task}"].to_numpy(int))
            with threadpool_limits(limits=1):
                predicted = fitted.probability(
                    g.drop(columns=[c for c in g if c.startswith("target_") or c == "pss_combo"])
                )
            np.testing.assert_allclose(p.probability, predicted, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(
                p.prior_corrected_probability, fitted.corrected(predicted), atol=1e-12, rtol=1e-12
            )
        inner = pd.read_parquet(path.parent / "inner_membership.parquet")
        for _, part in inner.groupby("inner_fold"):
            a, b = part[part.role.eq("train")], part[part.role.eq("validation")]
            assert not set(a.n_participant) & set(b.n_participant)
            assert set(a.n_ema_id) | set(b.n_ema_id) == set(train.n_ema_id)
            inner_checks += 1
        inner_p = pd.read_parquet(path.parent / "inner_oof.parquet")
        np.testing.assert_array_equal(inner_p.training_target, training_labels(train, task, spec))
        assert threshold(inner_p.training_target, inner_p.probability) == fitted.threshold
        donor = pd.read_parquet(path.parent / "augmentation_lineage.parquet")
        for split, g in donor.groupby("inner_fold"):
            allowed = (
                set(train.n_ema_id)
                if split == "outer_refit"
                else set(
                    inner.loc[inner.inner_fold.eq(int(split)) & inner.role.eq("train"), "n_ema_id"]
                )
            )
            assert set(g.base_ema_key) <= allowed and set(g.neighbor_ema_key) <= allowed
            assert not set(g.base_ema_key) & set(valid.n_ema_id)
            if spec["sampler"] == "conditional":
                a = train.iloc[g.base.to_numpy(int)]
                b = train.iloc[g.neighbor.to_numpy(int)]
                np.testing.assert_array_equal(a.n_participant, b.n_participant)
                np.testing.assert_array_equal(a.n_survey_category, b.n_survey_category)
            donor_checks += len(g)
        trials = json.loads((path.parent / "trials.json").read_text())
        candidate_evaluations += len(trials)
        nonconverged += sum(not all(t["converged"]) for t in trials)
        if fold == 0 and spec["name"] in [
            "context",
            "full1440",
            "compact_smote",
            "context_compact_conditional",
        ]:
            y = training_labels(train, task, spec)
            trans, x, yt, _, _ = prepare(train, blocks[spec["block"]], spec, cfg, y, 100)
            model = estimator(info["best"]["parameters"], spec["sampler"] == "weighted", cfg)
            with threadpool_limits(limits=1):
                assert checked_fit(model, x, yt)
                pred = model.predict_proba(trans.transform(valid))[:, 1]
            np.testing.assert_allclose(pred, fitted.probability(valid), atol=1e-12, rtol=1e-12)
            fresh += 1
        old_replays += int(info["previous_prediction_replay"])
        replay += 1
    # Broader independent source sample: one sorted EMA from each of nine distinct participants.
    eligible = frame[frame["hr__mean__1440m"].notna()]
    sample = eligible.groupby("n_participant", sort=True).head(1).head(9)
    lineage = pd.read_parquet(
        root / cfg["feature_bundle"] / "lineage.parquet", filters=[("sensor", "=", "hr")]
    )
    selected = lineage[
        lineage.n_ema_id.isin(sample.n_ema_id) & lineage.lookback_minutes.isin([60, 360, 720, 1440])
    ]
    traces = 0
    for directory, part in selected.groupby("directory"):
        p = (
            root
            / "private/hr_activity/features/76f5885b3955d5b0/reading_dispositions"
            / directory
            / "hr.parquet"
        )
        ledger = pd.read_parquet(p)
        g = ledger[ledger.a_first & np.isfinite(ledger.n_epoch_s)].reset_index(drop=True)
        for r in part.itertuples():
            expected = frame.set_index("n_ema_id").loc[r.n_ema_id]
            w = g[(g.n_epoch_s >= r.start_epoch) & (g.n_epoch_s < r.end_epoch)]
            assert w.index.tolist() == list(range(r.group_start, r.group_stop))
            assert w.n_participant.eq(expected.n_participant).all()
            v = w[w.a_acceptable]
            mean = (
                v.n_value.mean()
                if len(v) >= 2 and v.n_epoch_s.iloc[-1] - v.n_epoch_s.iloc[0] >= 1
                else np.nan
            )
            np.testing.assert_allclose(
                mean, expected[f"hr__mean__{r.lookback_minutes}m"], equal_nan=True
            )
            minute = v.groupby(np.floor(v.n_epoch_s / 60)).n_value.mean()
            mu = minute.mean() if len(minute) >= 2 else np.nan
            np.testing.assert_allclose(
                mu, expected[f"hr__observed_minute_mean__{r.lookback_minutes}m"], equal_nan=True
            )
            traces += 1
    before = json.loads((root / "private/binary_improvement/before.json").read_text())
    assert all(sha256(root / p) == h for p, h in before["files"].items())
    assert all(sha256(root / p) == h for p, h in before["manifests"].items())
    raw = json.loads((root / "private/integrity_baseline.json").read_text())
    raw_checks = 0
    for label, folder in [("data", root / "data"), ("reference", REFERENCE)]:
        for name, record in raw[label].items():
            st = (folder / name).stat()
            assert st.st_size == record["size"] and st.st_mtime_ns == record["mtime_ns"]
            raw_checks += 1
    results = dict(
        validation_replays=replay,
        matched_context_control_checks=matched_controls,
        primary_prediction_rows_preserved=len(primary),
        candidate_evaluations=candidate_evaluations,
        training_replays=replay,
        independent_refits=fresh,
        original_checkpoint_reproductions=old_replays,
        inner_participant_checks=inner_checks,
        augmentation_donors_checked=donor_checks,
        nonconverged_candidates=nonconverged,
        failed_jobs=failures,
        extra_source_traces=traces,
        extra_source_participants=sample.n_participant.nunique(),
        preserved_files=len(before["files"]),
        preserved_manifests=len(before["manifests"]),
        raw_metadata_checks=raw_checks,
        final_test_evaluated=False,
    )
    write_json(root / "private/binary_improvement/verification.json", results)
    if not (out / "manifest.json").exists():
        write_json(out / "verification.json", results)
    return results


if __name__ == "__main__":
    print(json.dumps(verify(Path.cwd()), indent=2))
