"""Immutable and resumable participant-grouped experiment jobs."""

import argparse
import importlib.metadata
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle, write_json
from .data import data, cohort
from .design import specifications
from .models import nested_fit


def job(root, run, task, spec, fold):
    with threadpool_limits(limits=1):
        return _job(root, run, task, spec, fold)


def _job(root, run, task, spec, fold):
    dest = run / "jobs" / f"{task}__{spec['name']}__fold{fold}"
    if dest.exists():
        verify_bundle(dest)
        return "cached"
    frame, blocks, cfg = data(root)
    f = cohort(frame, spec)
    train, valid = f[f.n_fold.ne(fold)], f[f.n_fold.eq(fold)]
    assert not set(train.n_participant) & set(valid.n_participant)
    target = f"target_{task}"
    info = dict(task=task, spec=spec, fold=fold)
    for label, g in [("train", train), ("validation", valid)]:
        info[label + "_responses"] = len(g)
        info[label + "_participants"] = g.n_participant.nunique()
        info[label + "_support"] = np.bincount(g[target].to_numpy(int), minlength=2).tolist()
        info[label + "_participants_by_class"] = [
            int(g.loc[g[target].eq(c), "n_participant"].nunique()) for c in [0, 1]
        ]
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=run / "jobs"))
    try:
        if not len(valid) or train[target].nunique() != 2:
            raise ValueError("Infeasible fold; no redrawing or support relaxation.")
        previous, reuse = None, None
        if spec["reuse"]:
            previous = root / cfg["previous_run"] / "jobs" / f"{task}__{spec['reuse']}__fold{fold}"
            verify_bundle(previous)
            reuse = json.loads((previous / "summary.json").read_text())
        fitted, best, trials, splits, audits, donors, inner_p, inner_y = nested_fit(
            train, task, spec, blocks[spec["block"]], cfg, reused=reuse
        )
        valid_x = valid.drop(
            columns=[c for c in valid if c.startswith("target_") or c == "pss_combo"]
        )
        for label, rows, probabilities in [
            ("validation", valid, fitted.probability(valid_x)),
            ("train", train, fitted.probability(train)),
        ]:
            pred = rows[["n_ema_id", "n_participant", "n_fold", "n_survey_category"]].copy()
            pred["target"] = rows[target].to_numpy(int)
            pred["probability"] = probabilities
            pred["prior_corrected_probability"] = fitted.corrected(probabilities)
            pred["decision_threshold"] = fitted.threshold
            pred["dummy_probability"] = float(train[target].mean())
            pred.to_parquet(build / f"{label}_predictions.parquet", index=False)
            if previous:
                old = pd.read_parquet(previous / f"{label}_predictions.parquet")
                assert old.n_ema_id.tolist() == pred.n_ema_id.tolist()
                np.testing.assert_allclose(
                    old.probability, pred.probability, rtol=1e-10, atol=1e-10
                )
        members = []
        for i, (a, b) in enumerate(splits):
            for role, ix in [("train", a), ("validation", b)]:
                members.append(
                    train.iloc[ix][["n_ema_id", "n_participant"]].assign(inner_fold=i, role=role)
                )
        pd.concat(members).to_parquet(build / "inner_membership.parquet", index=False)
        donor = pd.concat(donors, ignore_index=True)
        donor["inner_fold"] = donor.inner_fold.astype(str)
        if len(donor):
            donor["base_ema_key"] = train.n_ema_id.to_numpy()[donor.base.to_numpy(int)]
            donor["neighbor_ema_key"] = train.n_ema_id.to_numpy()[donor.neighbor.to_numpy(int)]
        donor.to_parquet(build / "augmentation_lineage.parquet", index=False)
        inner = train[["n_ema_id", "n_participant"]].copy()
        inner["training_target"] = inner_y
        inner["probability"] = inner_p
        inner.to_parquet(build / "inner_oof.parquet", index=False)
        joblib.dump(fitted, build / "estimator.joblib", compress=3)
        write_json(build / "trials.json", trials)
        write_json(build / "resampling.json", audits)
        write_json(
            build / "features.json",
            dict(
                names=fitted.transform.names,
                coefficients=fitted.model.coef_[0].tolist(),
                dropped=fitted.transform.base.dropped,
            ),
        )
        info.update(
            status="complete",
            best=best,
            threshold=fitted.threshold,
            transformed_features=len(fitted.transform.names),
            n_iter=int(fitted.model.n_iter_.max()),
            previous_prediction_replay=bool(previous),
            previous_manifest=sha256(previous / "manifest.json") if previous else None,
            original_prevalence=fitted.prevalence,
            effective_prevalence=fitted.effective_prevalence,
        )
    except Exception as exc:
        info.update(status="failed", error_type=type(exc).__name__)
        (build / "error_private.txt").write_text(str(exc))
    write_json(build / "summary.json", info)
    publish_bundle(build, dest, dict(task=task, spec=spec, fold=fold))
    return info["status"]


def run(root, workers=4):
    root = Path(root).resolve()
    audit = root / "reports/binary_improvement/audit/checks.json"
    assert audit.exists(), "Complete the source audit before fitting."
    frame, blocks, cfg = data(root)
    identity = dict(
        config=cfg,
        audit_sha256=sha256(audit),
        specification_sha256=sha256(root / "reports/binary_improvement/specification.md"),
        features=sha256(root / cfg["feature_bundle"] / "manifest.json"),
        previous=sha256(root / cfg["previous_run"] / "manifest.json"),
        morning=sha256(root / cfg["morning_bundle"] / "manifest.json"),
        assignments=sha256(root / "private/frozen_participant_assignments.parquet"),
        code={
            p.name: sha256(p)
            for p in Path(__file__).parent.glob("*.py")
            if p.stem in ["models", "sampling", "design", "data", "run"]
        },
        reused_source={
            str(p.relative_to(root)): sha256(p)
            for package in [
                "garmin_binary_6hour",
                "garmin_comprehensive",
                "garmin_comprehensive_regression",
            ]
            for p in (root / "src" / package).glob("*.py")
        },
        packages={
            n: importlib.metadata.version(n)
            for n in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]
        },
    )
    run = root / "private/binary_improvement/runs" / digest_object(identity)[:16]
    run.mkdir(parents=True, exist_ok=True)
    (run / "jobs").mkdir(exist_ok=True)
    write_json(
        root / "private/binary_improvement/current.json", dict(run=str(run.relative_to(root)))
    )
    if (run / "manifest.json").exists():
        verify_bundle(run)
        return run
    if not (run / "identity.json").exists():
        write_json(run / "identity.json", identity)
        write_json(run / "blocks.json", blocks)
        write_json(run / "specifications.json", specifications())
        source = run / "executed_source"
        source.mkdir()
        for name in identity["code"]:
            (source / name).write_bytes((Path(__file__).parent / name).read_bytes())
        frame[
            [
                "n_ema_id",
                "n_participant",
                "n_fold",
                "n_survey_category",
                "pss_combo",
                "target_A",
                "target_B",
                "context_available",
                "supported",
                "retro_nominal",
                "retro_later",
            ]
        ].to_parquet(run / "accounting.parquet", index=False)
    jobs = [(t, s, k) for t in ["A", "B"] for s in specifications() for k in range(5)]
    print(f"Frozen {len(jobs)} jobs, version {run.name}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(job, root, run, t, s, k): (t, s["name"], k) for t, s, k in jobs}
        for i, result in enumerate(as_completed(pending), 1):
            print(i, len(jobs), *pending[result], result.result(), flush=True)
    return run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    print(run(Path.cwd(), p.parse_args().workers))
