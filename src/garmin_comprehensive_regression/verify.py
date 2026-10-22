"""Prediction replay, fresh fitting, membership and immutable input verification."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import verify_bundle, write_json
from .models import estimator, fit_checked, predict
from .run import TARGET, basic, cohort, data


def verify(root):
    root = Path(root)
    run = (
        root
        / json.loads((root / "private/comprehensive_regression/current.json").read_text())["run"]
    )
    identity = json.loads((run / "run_identity.json").read_text())
    cfg = identity["config"]
    assert sha256(root / cfg["feature_bundle"] / "manifest.json") == identity["features"]
    assert (
        sha256(root / "private/frozen_participant_assignments.parquet") == identity["assignments"]
    )
    assert all(sha256(run / "executed_source" / name) == h for name, h in identity["code"].items())
    frame, blocks = data(root, cfg)
    replays = fresh = inner_checks = failed = candidate_failures = trials = 0
    max_error = 0.0
    for path in sorted((run / "jobs").glob("*/summary.json")):
        verify_bundle(path.parent)
        info = json.loads(path.read_text())
        spec = info["spec"]
        fold = info["fold"]
        if info["status"] != "complete":
            failed += 1
            continue
        f = cohort(frame, blocks, spec["cohort"])
        train, valid = f[f.n_fold.ne(fold)], f[f.n_fold.eq(fold)]
        saved = pd.read_parquet(path.parent / "predictions.parquet")
        saved_train = pd.read_parquet(path.parent / "training_predictions.parquet")
        assert list(saved.n_ema_id) == list(valid.n_ema_id)
        assert list(saved_train.n_ema_id) == list(train.n_ema_id)
        assert np.array_equal(saved[TARGET], valid[TARGET])
        assert not set(train.n_participant) & set(valid.n_participant)
        assert saved.dummy_mean.eq(float(train[TARGET].mean())).all()
        assert saved.dummy_median.eq(float(train[TARGET].median())).all()
        membership = pd.read_parquet(path.parent / "inner_membership.parquet")
        for _, g in membership.groupby("inner_fold"):
            a, b = g[g.role.eq("train")], g[g.role.eq("validation")]
            assert not set(a.n_participant) & set(b.n_participant)
            assert set(g.n_ema_id) == set(train.n_ema_id)
            assert not set(g.n_ema_id) & set(valid.n_ema_id)
            inner_checks += 1
        model = joblib.load(path.parent / "estimator.joblib")
        t = model.transform
        np.testing.assert_allclose(t.medians, train[t.kept].median(), atol=1e-12)
        if t.kept:
            np.testing.assert_allclose(
                t.scaler.mean_, train[t.kept].fillna(t.medians).mean(), atol=1e-12
            )
        assert t.encoder.categories_[0].tolist() == sorted(train.n_survey_category.unique())
        expected_flags = [c for c in t.columns if train[c].isna().nunique() > 1] if t.flags else []
        assert t.flag_columns == expected_flags
        with threadpool_limits(limits=1):
            prediction = model.predict(valid.drop(columns=TARGET))
        error = float(np.max(np.abs(prediction - saved.prediction)))
        max_error = max(max_error, error)
        assert error < 1e-9
        actual = basic(valid[TARGET].to_numpy(), prediction)
        for key in ["mae", "rmse", "r2"]:
            assert abs(actual[key] - info["validation"][key]) < 1e-10
        replays += 1
        candidate = json.loads((path.parent / "trials.json").read_text())
        trials += len(candidate)
        candidate_failures += sum(not x["converged"] for x in candidate)
        if fold == 0 and spec["cohort"] == "all_valid" and spec["view"] in ["original60", "full"]:
            with threadpool_limits(limits=1):
                x = t.transform(train)[:, model.selected]
                refit, ok, _ = fit_checked(
                    estimator(spec["family"], info["best"]["setting"], cfg),
                    x,
                    train[TARGET].to_numpy(float),
                )
                assert ok
                np.testing.assert_allclose(
                    predict(refit, t.transform(valid)[:, model.selected]),
                    saved.prediction,
                    atol=1e-8,
                    rtol=1e-10,
                )
            fresh += 1
    expected = len(json.loads((run / "specifications.json").read_text())) * 5
    assert replays + failed == expected
    snapshot = json.loads((root / "private/comprehensive_regression/before.json").read_text())
    changed = [p for p, h in snapshot["files"].items() if sha256(root / p) != h]
    assert set(changed) <= {"src/garmin_inventory/privacy.py"}
    assert all(sha256(root / p) == h for p, h in snapshot["manifests"].items())
    # Pin the normalized/source manifests without parsing any final-test sensor/outcome records.
    verify_bundle(root / cfg["feature_bundle"])
    complete13 = json.loads((root / "reports/comprehensive_features/completion.json").read_text())
    assert all(sha256(root / p) == h for p, h in complete13["presentation_sha256"].items())
    # Preserve sealing: inspect filesystem metadata, not raw participant contents.
    from garmin_comprehensive.extract import REFERENCE

    baseline = json.loads((root / "private/integrity_baseline.json").read_text())
    raw_metadata_checks = 0
    for label, directory in [("data", root / "data"), ("reference", REFERENCE)]:
        for name, record in baseline[label].items():
            stat = (directory / name).stat()
            assert stat.st_size == record["size"] and stat.st_mtime_ns == record["mtime_ns"]
            raw_metadata_checks += 1
    result = dict(
        estimator_replays=replays,
        failed_jobs=failed,
        max_prediction_replay_error=max_error,
        independent_fresh_fits=fresh,
        inner_membership_checks=inner_checks,
        tuning_candidates=trials,
        nonconverged_candidates=candidate_failures,
        prior_files_preserved=len(snapshot["files"]) - len(changed),
        intentional_existing_change="Notebook 14 aggregate-output privacy authorization only"
        if changed
        else None,
        prior_private_manifests_preserved=len(snapshot["manifests"]),
        notebook13_seal="unchanged",
        feature_bundle="verified unchanged",
        frozen_assignments="verified unchanged",
        final_test_evaluated=False,
        raw_reference_metadata_checks=raw_metadata_checks,
        raw_contents_reread=False,
    )
    write_json(root / "private/comprehensive_regression/verification.json", result)
    public = root / "reports/comprehensive_regression" / run.name
    if public.exists() and not (public / "manifest.json").exists():
        write_json(public / "verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(Path.cwd()), indent=2))
