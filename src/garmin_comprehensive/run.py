"""Resumable immutable outer-fold jobs; private estimators and OOF predictions."""

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from garmin_ema_baseline.linkage import check_development
from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (
    digest_object,
    frozen_assignments,
    publish_bundle,
    verify_bundle,
    write_json,
)
from garmin_pss_classification.data import TASKS
from garmin_pss_classification.metrics import score
from .extract import REFERENCE
from .models import feature_blocks, specs, nested_fit


def metrics(y, prob, groups, k):
    result, cm, per = score(y, prob.argmax(axis=1), prob, groups, k)
    result["log_loss"] = float(log_loss(y, prob, labels=np.arange(k)))
    result["brier_sum"] = float(np.mean(np.sum((prob - np.eye(k)[y]) ** 2, axis=1)))
    return result, cm, per


def get_data(feature_path):
    data = pd.read_parquet(feature_path / "development_features.parquet")
    entries = json.loads((feature_path / "dictionary.json").read_text())
    blocks = feature_blocks(entries)
    quality = pd.read_parquet(feature_path / "quality.parquet")
    qcols = [
        "accepted_readings",
        "accepted_fraction",
        "occupied_fraction",
        "span_seconds",
        "latest_age_seconds",
        "maximum_gap_seconds",
        "supported_pairs",
        "supported_seconds",
    ]
    q = quality.pivot(index="n_ema_id", columns=["sensor", "lookback_minutes"], values=qcols)
    q.columns = [f"quality__{s}__{v}__{m}m" for v, s, m in q.columns]
    # Counter-only / reading-only columns intentionally all-missing for other modalities.
    # Keep semantics explicit; training-only transformer discards empty columns.
    blocks["quality"] = list(q.columns)
    data = data.merge(q.reset_index(), on="n_ema_id", validate="one_to_one")
    return data, blocks


def one_job(root, feature_path, run_path, task, spec, fold, cfg):
    with threadpool_limits(limits=1):
        return _one_job(root, feature_path, run_path, task, spec, fold, cfg)


def _one_job(root, feature_path, run_path, task, spec, fold, cfg):
    destination = run_path / "jobs" / f"{task}__{spec['name']}__fold{fold}"
    if destination.exists():
        verify_bundle(destination)
        return "cached"
    data, blocks = get_data(feature_path)
    target = "target_" + task
    k = TASKS[task]
    mask = data[target].notna() & data.context_available
    if spec["handling"] == "complete":
        mask &= data[blocks["original60"]].notna().all(axis=1)
    data = data[mask].copy()
    train = data[data.n_fold.ne(fold)].copy()
    valid = data[data.n_fold.eq(fold)].copy()
    if set(train.n_participant) & set(valid.n_participant):
        raise AuditError("Participant fold leakage.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=destination.parent))
    info = dict(
        task=task,
        spec=spec,
        fold=int(fold),
        training_responses=len(train),
        validation_responses=len(valid),
        training_participants=train.n_participant.nunique(),
        validation_participants=valid.n_participant.nunique(),
        training_support=np.bincount(train[target].to_numpy(int), minlength=k).tolist(),
        validation_support=np.bincount(valid[target].to_numpy(int), minlength=k).tolist(),
    )
    try:
        if not len(valid) or len(np.unique(train[target])) != k:
            raise AuditError("Infeasible outer-fold class support.")
        trans, model, p, pt, best, trials, splits = nested_fit(
            train, valid.drop(columns=[target]), target, k, spec, blocks, cfg
        )
        info.update(
            best=best,
            transformed_features=len(trans.names),
            retained_numeric=len(trans.kept),
            selected_nonzero=int(np.any(np.abs(model.coef_) > 1e-8, axis=0).sum()),
            n_iter=int(model.n_iter_.max()),
            status="complete",
        )
        joblib.dump(dict(transform=trans, model=model), build / "estimator.joblib")
        pred = valid[["n_ema_id", "n_participant", "n_fold", "n_survey_category"]].copy()
        pred["target"] = valid[target].to_numpy(int)
        for j in range(k):
            pred[f"p{j}"] = p[:, j]
        pred.to_parquet(build / "predictions.parquet", index=False)
        train[["n_ema_id", "n_participant", "n_fold"]].to_parquet(
            build / "training_membership.parquet", index=False
        )
        inner = []
        for i, (a, b) in enumerate(splits):
            for role, idx in [("train", a), ("validation", b)]:
                f = train.iloc[idx][["n_ema_id", "n_participant"]].copy()
                f["inner_fold"] = i
                f["role"] = role
                inner.append(f)
        pd.concat(inner).to_parquet(build / "inner_membership.parquet", index=False)
        train_metrics = metrics(train[target].to_numpy(int), pt, train.n_participant.to_numpy(), k)[
            0
        ]
        valid_metrics = metrics(valid[target].to_numpy(int), p, valid.n_participant.to_numpy(), k)[
            0
        ]
        write_json(build / "training_metrics.json", train_metrics)
        write_json(build / "validation_metrics.json", valid_metrics)
        write_json(build / "trials.json", trials)
        write_json(
            build / "transform.json",
            dict(
                kept=trans.kept,
                dropped=trans.dropped,
                names=trans.names,
                medians=trans.medians.to_dict(),
                means=trans.scaler.mean_.tolist() if trans.scaler else [],
                scales=trans.scaler.scale_.tolist() if trans.scaler else [],
            ),
        )
        coefs = [
            dict(
                feature=name,
                coefficients=model.coef_[:, j].tolist(),
                selected=bool(np.any(np.abs(model.coef_[:, j]) > 1e-8)),
            )
            for j, name in enumerate(trans.names)
        ]
        write_json(build / "coefficients.json", coefs)
    except AuditError as e:
        info.update(status="skipped", reason=str(e))
    write_json(build / "summary.json", info)
    publish_bundle(
        build,
        destination,
        dict(configuration=cfg, feature_manifest=sha256(feature_path / "manifest.json")),
    )
    return info["status"]


def run(root, feature_path, workers=None):
    root, feature_path = Path(root).resolve(), Path(feature_path).resolve()
    cfg = json.loads((root / "config/comprehensive_features.json").read_text())
    verify_bundle(feature_path)
    assignments, _ = frozen_assignments(root, REFERENCE)
    data, blocks = get_data(feature_path)
    check_development(data, assignments, 3706)
    code = {
        p.name: sha256(p)
        for p in Path(__file__).parent.glob("*.py")
        if p.name in ["run.py", "models.py"]
    }
    identity = dict(
        configuration=cfg,
        features=sha256(feature_path / "manifest.json"),
        code=code,
        specification=sha256(root / "reports/comprehensive_features/specification.md"),
    )
    version = digest_object(identity)[:16]
    path = root / "private/comprehensive_features/models" / version
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "run_identity.json", identity)
    write_json(
        root / "private/comprehensive_features/current.json",
        dict(features=str(feature_path.relative_to(root)), models=str(path.relative_to(root))),
    )
    jobs = [(task, spec, fold) for task in TASKS for spec in specs(blocks) for fold in range(5)]
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    print(f"Running {len(jobs)} outer jobs with nested participant validation", flush=True)
    with ProcessPoolExecutor(max_workers=workers or cfg["workers"]) as pool:
        futures = [
            pool.submit(one_job, root, feature_path, path, task, spec, fold, cfg)
            for task, spec, fold in jobs
        ]
        for i, f in enumerate(as_completed(futures)):
            status = f.result()
            if (i + 1) % 5 == 0 or status == "skipped":
                print(f"Finished {i + 1}/{len(jobs)} jobs; latest status {status}", flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    print(run(Path.cwd(), Path(args.features), args.workers))
