"""Resumable immutable jobs; no raw extraction or final-test loading."""

import argparse
import importlib.metadata
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from garmin_comprehensive_regression.run import data as validated_data
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle, write_json
from .models import TASKS, blocks, cohort, nested_fit, specifications, targets


@lru_cache(maxsize=1)
def data(root):
    root = Path(root)
    cfg = json.loads((root / "config/binary_6hour.json").read_text())
    frame, _ = validated_data(root, cfg)
    entries = json.loads((root / cfg["feature_bundle"] / "dictionary.json").read_text())
    for task, threshold in TASKS.items():
        frame[f"target_{task}"] = targets(frame.pss_combo, threshold)
        assert frame[f"target_{task}"].equals(frame[f"target_at_least_{threshold}"].astype(float))
    return frame, blocks(entries), cfg


def job(root, run, task, spec, fold):
    with threadpool_limits(limits=1):
        return _job(root, run, task, spec, fold)


def _job(root, run, task, spec, fold):
    dest = run / "jobs" / f"{task}__{spec['name']}__fold{fold}"
    if dest.exists():
        verify_bundle(dest)
        return "cached"
    frame, feature_blocks, cfg = data(root)
    f = cohort(frame, feature_blocks, spec["cohort"])
    train, valid = f[f.n_fold.ne(fold)], f[f.n_fold.eq(fold)]
    assert not set(train.n_participant) & set(valid.n_participant)
    target = f"target_{task}"
    info = dict(
        task=task,
        spec=spec,
        fold=fold,
        train_responses=len(train),
        validation_responses=len(valid),
        train_participants=train.n_participant.nunique(),
        validation_participants=valid.n_participant.nunique(),
    )
    for role, rows in [("train", train), ("validation", valid)]:
        info[f"{role}_support"] = np.bincount(rows[target].to_numpy(int), minlength=2).tolist()
        info[f"{role}_participants_by_class"] = [
            int(rows.loc[rows[target].eq(c), "n_participant"].nunique()) for c in [0, 1]
        ]
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=run / "jobs"))
    try:
        if len(valid) == 0 or train[target].nunique() != 2:
            raise ValueError("Infeasible outer fold.")
        fitted, best, trials, splits = nested_fit(
            train, task, spec, feature_blocks[spec["block"]], cfg
        )
        valid_x = valid.drop(
            columns=[c for c in valid if c == "pss_combo" or c.startswith("target_")]
        )
        majority = int(train[target].mean() > 0.5)
        for role, rows, probs in [
            ("validation", valid, fitted.probability(valid_x)),
            ("train", train, fitted.probability(train)),
        ]:
            result = rows[["n_ema_id", "n_participant", "n_fold", "n_survey_category"]].copy()
            result["target"] = rows[target].to_numpy(int)
            result["probability"] = probs
            result["dummy_probability"] = majority
            result.to_parquet(build / f"{role}_predictions.parquet", index=False)
        memberships = []
        for inner, (a, b) in enumerate(splits):
            for role, ix in [("train", a), ("validation", b)]:
                memberships.append(
                    train.iloc[ix][["n_ema_id", "n_participant"]].assign(
                        inner_fold=inner, role=role
                    )
                )
        pd.concat(memberships).to_parquet(build / "inner_membership.parquet", index=False)
        joblib.dump(fitted, build / "estimator.joblib", compress=3)
        write_json(build / "trials.json", trials)
        write_json(
            build / "features.json",
            dict(
                names=fitted.transform.names,
                coefficients=fitted.model.coef_[0].tolist(),
                dropped=fitted.transform.dropped,
            ),
        )
        info.update(
            status="complete",
            best=best,
            n_iter=int(fitted.model.n_iter_.max()),
            transformed_features=len(fitted.transform.names),
            nonzero_features=int((np.abs(fitted.model.coef_[0]) > 1e-8).sum()),
        )
    except Exception as exc:
        info.update(status="failed", error_type=type(exc).__name__)
        (build / "error_private.txt").write_text(str(exc))
    write_json(build / "summary.json", info)
    publish_bundle(build, dest, dict(task=task, spec=spec, fold=fold))
    return info["status"]


def run(root, workers=4):
    root = Path(root).resolve()
    frame, feature_blocks, cfg = data(root)
    identity = dict(
        config=cfg,
        specification=sha256(root / "reports/binary_6hour/specification.md"),
        features=sha256(root / cfg["feature_bundle"] / "manifest.json"),
        assignments=sha256(root / "private/frozen_participant_assignments.parquet"),
        code={
            p.name: sha256(p)
            for p in Path(__file__).parent.glob("*.py")
            if p.name in ["models.py", "run.py"]
        },
        reused_source={
            p: sha256(root / p)
            for p in [
                "src/garmin_comprehensive/models.py",
                "src/garmin_comprehensive/run.py",
                "src/garmin_comprehensive_regression/run.py",
            ]
        },
        packages={
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]
        },
    )
    dest = root / "private/binary_6hour/runs" / digest_object(identity)[:16]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "jobs").mkdir(exist_ok=True)
    write_json(root / "private/binary_6hour/current.json", dict(run=str(dest.relative_to(root))))
    if (dest / "manifest.json").exists():
        verify_bundle(dest)
        return dest
    if not (dest / "identity.json").exists():
        write_json(dest / "identity.json", identity)
        write_json(dest / "feature_blocks.json", feature_blocks)
        write_json(dest / "specifications.json", specifications())
        source = dest / "executed_source"
        source.mkdir()
        for name in identity["code"]:
            (source / name).write_bytes((Path(__file__).parent / name).read_bytes())
        (source / "specification.md").write_bytes(
            (root / "reports/binary_6hour/specification.md").read_bytes()
        )
        accounting = frame[
            [
                "n_ema_id",
                "n_participant",
                "n_fold",
                "n_survey_category",
                "pss_combo",
                "target_A",
                "target_B",
                "context_available",
            ]
        ].copy()
        accounting["included"] = frame.pss_combo.notna() & frame.context_available
        accounting["complete_common"] = frame.n_ema_id.isin(
            cohort(frame, feature_blocks, "complete").n_ema_id
        )
        accounting["reason"] = np.where(
            frame.pss_combo.isna(),
            "missing_outcome",
            np.where(~frame.context_available, "missing_context", "included"),
        )
        accounting.to_parquet(dest / "ema_accounting.parquet", index=False)
    jobs = [(task, spec, fold) for task in TASKS for spec in specifications() for fold in range(5)]
    print(f"Frozen {len(jobs)} jobs, version {dest.name}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(job, root, dest, t, s, f): (t, s["name"], f) for t, s, f in jobs}
        for i, result in enumerate(as_completed(pending), 1):
            print(i, len(jobs), *pending[result], result.result(), flush=True)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    print(run(Path.cwd(), parser.parse_args().workers))
