"""Replay, fold separation, training-only transforms and preservation checks."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from garmin_comprehensive.extract import REFERENCE
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import verify_bundle, write_json
from .models import checked_fit, cohort, estimator, make_transform
from .run import data


def verify(root):
    root = Path(root)
    run = root / json.loads((root / "private/binary_6hour/current.json").read_text())["run"]
    out = root / "reports/binary_6hour" / run.name
    frame, blocks, cfg = data(root)
    identity = json.loads((run / "identity.json").read_text())
    assert identity["features"] == sha256(root / cfg["feature_bundle"] / "manifest.json")
    assert identity["assignments"] == sha256(
        root / "private/frozen_participant_assignments.parquet"
    )
    for name, h in identity["code"].items():
        assert sha256(Path(__file__).parent / name) == h == sha256(run / "executed_source" / name)
    assert all(sha256(root / p) == h for p, h in identity["reused_source"].items())
    replay = fresh = inner_checks = candidate_count = nonconverged = 0
    max_error = 0.0
    supports = []
    for path in sorted((run / "jobs").glob("*/summary.json")):
        verify_bundle(path.parent)
        info = json.loads(path.read_text())
        assert info["status"] == "complete", "Do not conceal failed jobs."
        spec, task, fold = info["spec"], info["task"], info["fold"]
        f = cohort(frame, blocks, spec["cohort"])
        train, valid = f[f.n_fold.ne(fold)], f[f.n_fold.eq(fold)]
        assert not set(train.n_participant) & set(valid.n_participant)
        target = f"target_{task}"
        saved = pd.read_parquet(path.parent / "validation_predictions.parquet")
        assert saved.n_ema_id.tolist() == valid.n_ema_id.tolist()
        np.testing.assert_array_equal(saved.target, valid[target].to_numpy(int))
        for role, rows in [("train", train), ("validation", valid)]:
            assert (
                np.bincount(rows[target].to_numpy(int), minlength=2).tolist()
                == info[f"{role}_support"]
            )
        supports.append(
            (
                task,
                spec["cohort"],
                fold,
                tuple(info["train_support"]),
                tuple(info["validation_support"]),
            )
        )
        fitted = joblib.load(path.parent / "estimator.joblib")
        valid_x = valid.drop(
            columns=[c for c in valid if c == "pss_combo" or c.startswith("target_")]
        )
        with threadpool_limits(limits=1):
            p = fitted.probability(valid_x)
            training_probability = fitted.probability(train)
        saved_train = pd.read_parquet(path.parent / "train_predictions.parquet")
        assert saved_train.n_ema_id.tolist() == train.n_ema_id.tolist()
        np.testing.assert_array_equal(saved_train.target, train[target].to_numpy(int))
        np.testing.assert_allclose(
            training_probability, saved_train.probability, atol=1e-12, rtol=1e-12
        )
        error = float(np.max(np.abs(p - saved.probability)))
        max_error = max(max_error, error)
        np.testing.assert_allclose(p, saved.probability, atol=1e-12, rtol=1e-12)
        t = fitted.transform
        pd.testing.assert_series_equal(t.medians, train[t.kept].median())
        if t.kept:
            np.testing.assert_allclose(
                t.scaler.mean_, train[t.kept].fillna(t.medians).to_numpy(float).mean(axis=0)
            )
        assert t.flag_columns == (
            [c for c in t.columns if train[c].isna().nunique() > 1] if t.flags else []
        )
        assert not any(
            c.startswith(("n_", "target_", "PSS_")) or c == "pss_combo" for c in t.columns
        )
        assert fitted.model.class_weight == ("balanced" if spec["weighted"] else None)
        membership = pd.read_parquet(path.parent / "inner_membership.parquet")
        for _, g in membership.groupby("inner_fold"):
            a, b = g[g.role.eq("train")], g[g.role.eq("validation")]
            assert not set(a.n_participant) & set(b.n_participant)
            assert set(g.n_ema_id) == set(train.n_ema_id)
            assert not set(g.n_participant) & set(valid.n_participant)
            inner_checks += 1
        trials = json.loads((path.parent / "trials.json").read_text())
        candidate_count += len(trials)
        nonconverged += sum(not all(t["converged"]) for t in trials)
        expected = max((t for t in trials if all(t["converged"])), key=lambda t: t["mean_auroc"])
        assert expected == info["best"]
        if fold in [0, 4] and spec["name"] in ["context_u", "full360_u", "full360_w"]:
            with threadpool_limits(limits=1):
                newt = make_transform(train, blocks[spec["block"]], spec, cfg)
                newm = estimator(expected["parameters"], spec["weighted"], cfg)
                assert checked_fit(newm, newt.transform(train), train[target].to_numpy(int))
                np.testing.assert_allclose(
                    newm.predict_proba(newt.transform(valid_x))[:, 1],
                    saved.probability,
                    atol=1e-10,
                    rtol=1e-10,
                )
            fresh += 1
        replay += 1
    assert replay == len(json.loads((run / "specifications.json").read_text())) * 10
    # Every model in a cohort/task/fold must have identical class support.
    s = pd.DataFrame(supports, columns=["task", "cohort", "fold", "train", "validation"])
    assert (
        s.groupby(["task", "cohort", "fold"])[["train", "validation"]].nunique().eq(1).all().all()
    )
    before = json.loads((root / "private/binary_6hour/before.json").read_text())
    assert all(sha256(root / p) == h for p, h in before["files"].items())
    assert all(sha256(root / p) == h for p, h in before["manifests"].items())
    for report in ["comprehensive_features", "comprehensive_regression"]:
        seal = json.loads((root / f"reports/{report}/completion.json").read_text())
        assert all(sha256(root / p) == h for p, h in seal["presentation_sha256"].items())
    raw = json.loads((root / "private/integrity_baseline.json").read_text())
    raw_checks = 0
    for label, folder in [("data", root / "data"), ("reference", REFERENCE)]:
        for name, record in raw[label].items():
            stat = (folder / name).stat()
            assert stat.st_size == record["size"] and stat.st_mtime_ns == record["mtime_ns"]
            raw_checks += 1
    verify_bundle(root / cfg["feature_bundle"])
    result = dict(
        estimator_replays=replay,
        training_prediction_replays=replay,
        max_prediction_error=max_error,
        independent_fresh_fits=fresh,
        inner_membership_checks=inner_checks,
        tuning_candidates=candidate_count,
        nonconverged_candidates=nonconverged,
        old_files_preserved=len(before["files"]),
        old_private_manifests_preserved=len(before["manifests"]),
        raw_metadata_checks=raw_checks,
        final_test_evaluated=False,
        raw_contents_reread=False,
        feature_bundle="unchanged",
        frozen_assignments="unchanged",
        notebook13_and_14_seals="unchanged",
    )
    write_json(root / "private/binary_6hour/verification.json", result)
    if not (out / "manifest.json").exists():
        write_json(out / "verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(Path.cwd()), indent=2))
