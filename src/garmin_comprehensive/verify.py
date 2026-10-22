"""Independent replay, privacy, provenance and held-out participant checks."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from garmin_inventory.privacy import sha256, contains_private_token
from garmin_preprocessing.common import frozen_assignments, input_guard, verify_bundle, write_json
from .extract import REFERENCE
from .models import fit
from .run import get_data


def verify(root):
    root = Path(root)
    pointers = json.loads((root / "private/comprehensive_features/current.json").read_text())
    fp, run = root / pointers["features"], root / pointers["models"]
    verify_bundle(fp)
    input_guard(root, REFERENCE, root / "data")
    assignments, dev = frozen_assignments(root, REFERENCE)
    frame, blocks = get_data(fp)
    assert set(frame.n_participant) == dev and frame.n_partition.eq("development").all()
    cfg = json.loads((root / "config/comprehensive_features.json").read_text())
    replays = 0
    fresh = 0
    inner = 0
    max_error = 0.0
    trials = 0
    failed_candidates = 0
    for p in sorted((run / "jobs").glob("*/summary.json")):
        info = json.loads(p.read_text())
        job = p.parent
        verify_bundle(job)
        if info["status"] != "complete":
            continue
        target = "target_" + info["task"]
        fold = info["fold"]
        mask = frame[target].notna() & frame.context_available
        if info["spec"]["handling"] == "complete":
            mask &= frame[blocks["original60"]].notna().all(axis=1)
        f = frame[mask]
        train = f[f.n_fold.ne(fold)]
        valid = f[f.n_fold.eq(fold)]
        saved = pd.read_parquet(job / "predictions.parquet")
        assert list(saved.n_ema_id) == list(valid.n_ema_id)
        assert not set(train.n_participant) & set(valid.n_participant)
        mem = pd.read_parquet(job / "training_membership.parquet")
        assert list(mem.n_ema_id) == list(train.n_ema_id)
        membership = pd.read_parquet(job / "inner_membership.parquet")
        for _, m in membership.groupby("inner_fold"):
            a = m[m.role.eq("train")]
            b = m[m.role.eq("validation")]
            assert not set(a.n_participant) & set(b.n_participant)
            assert set(a.n_ema_id) | set(b.n_ema_id) == set(train.n_ema_id)
            assert not set(m.n_ema_id) & set(valid.n_ema_id)
            inner += 1
        obj = joblib.load(job / "estimator.joblib")
        t, model = obj["transform"], obj["model"]
        np.testing.assert_allclose(t.medians.to_numpy(), train[t.kept].median().to_numpy())
        if t.kept:
            numeric = train[t.kept].fillna(t.medians).to_numpy(float)
            np.testing.assert_allclose(t.scaler.mean_, numeric.mean(axis=0))
        expected = saved[[c for c in saved if c.startswith("p") and c[1:].isdigit()]].to_numpy(
            float
        )
        with threadpool_limits(limits=1):
            actual = model.predict_proba(t.transform(valid))
        error = float(np.max(np.abs(actual - expected)))
        max_error = max(max_error, error)
        assert error < 1e-10 and np.allclose(actual.sum(axis=1), 1)
        assert np.array_equal(saved.target, valid[target].to_numpy(int))
        replays += 1
        trial = json.loads((job / "trials.json").read_text())
        trials += len(trial)
        failed_candidates += sum(not x["converged"] for x in trial)
        if fold == 0 and info["spec"]["name"] in ["original60", "full_corr"]:
            with threadpool_limits(limits=1):
                refit, ok = fit(t.transform(train), train[target].to_numpy(int), info["best"], cfg)
                assert ok
                np.testing.assert_allclose(
                    refit.predict_proba(t.transform(valid)), expected, atol=1e-10
                )
            fresh += 1
    snapshot = json.loads((root / "private/comprehensive_features/before.json").read_text())
    changed = [p for p, h in snapshot["files"].items() if sha256(root / p) != h]
    assert changed == ["src/garmin_inventory/privacy.py"] or set(changed) == {
        "src/garmin_inventory/privacy.py"
    }
    assert all(sha256(root / p) == h for p, h in snapshot["manifests"].items())
    tokens = json.loads((root / "private/privacy_tokens.json").read_text())
    scanned = 0
    for directory in [root / "reports/comprehensive_features", root / "src/garmin_comprehensive"]:
        for p in directory.rglob("*"):
            if p.suffix in [".json", ".md", ".html", ".py", ".svg"]:
                assert not contains_private_token(
                    p.read_text(), tokens["identifiers"], tokens["names"]
                ), "Private token detected; details withheld."
                scanned += 1
    result = dict(
        estimator_replays=replays,
        independent_fresh_fits=fresh,
        inner_split_checks=inner,
        max_replay_probability_error=max_error,
        tuning_candidates=trials,
        nonconverged_candidates=failed_candidates,
        existing_files_preserved=len(snapshot["files"]) - 1,
        intentional_existing_change="privacy.py adds explicitly authorized aggregate Notebook13 output validation",
        existing_private_manifests_preserved=len(snapshot["manifests"]),
        public_source_files_scanned=scanned,
        source_integrity="passed",
        frozen_assignments="passed",
        final_test_evaluation=False,
        feature_manifest_sha256=sha256(fp / "manifest.json"),
    )
    write_json(root / "private/comprehensive_features/verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(Path.cwd()), indent=2))
