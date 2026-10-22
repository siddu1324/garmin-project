"""Resumable immutable data audit and small fold jobs; private predictions only."""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle, write_json
from . import audit
from .data import ITEMS, TASKS, load, build_probe
from .models import specs, fit_model


def build(root):
    root = Path(root)
    f, cfg = load(root)
    identity = dict(config=cfg, specification=sha256(root / "reports/root_cause_analysis/specification.md"),
                    code={p.name: sha256(p) for p in [Path(__file__).with_name(n) for n in ["data.py", "audit.py"]]},
                    sources={n: sha256(root / cfg[n] / "manifest.json") for n in ["development_bundle", "normalized_bundle", "ledger_bundle"]},
                    assignments=sha256(root / "private/frozen_participant_assignments.parquet"))
    dest = root / "private/root_cause_analysis/data" / digest_object(identity)[:16]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        verify_bundle(dest)
        return dest, cfg
    tmp = Path(tempfile.mkdtemp(prefix=".building_", dir=dest.parent))
    audit.run(root, f, cfg, tmp)
    f.to_parquet(tmp / "development.parquet", index=False)
    probe = build_probe(root, f, cfg)
    probe.to_parquet(tmp / "activity_probe.parquet", index=False)
    write_json(tmp / "identity.json", identity)
    publish_bundle(tmp, dest, identity)
    return dest, cfg


def job(folder, frame, target, kind, spec, fold, cfg):
    name = f"{kind}__{target}__{spec['name']}__fold{fold}"
    dest = folder / "jobs" / name
    if dest.exists():
        verify_bundle(dest)
        return
    tmp = Path(tempfile.mkdtemp(prefix=".building_", dir=dest.parent))
    train, valid = frame[frame.n_fold.ne(fold)], frame[frame.n_fold.eq(fold)]
    assert not set(train.n_participant) & set(valid.n_participant)
    info = dict(target=target, kind=kind, spec=spec, fold=fold)
    for role, z in [("train", train), ("validation", valid)]:
        info[role] = dict(responses=len(z), participants=z.n_participant.nunique(),
                          support={str(k): int(v) for k, v in z[target].value_counts().sort_index().items()})
    try:
        if len(valid) == 0 or len(train) == 0:
            raise ValueError("Empty frozen fold")
        fitted, trials, members = fit_model(train, target, spec, kind, cfg)
        for role, z in [("train", train), ("validation", valid)]:
            p = z[["n_ema_id", "n_participant", "n_fold", "n_survey_category"]].copy()
            p["target"] = z[target].to_numpy(float)
            predictions = fitted.predict(z)
            if kind == "ordinal":
                counts = np.bincount(train[target].to_numpy(int) - 1, minlength=5)
                for k in range(5):
                    p[f"p{k+1}"] = predictions[:, k]
                    p[f"dummy{k+1}"] = counts[k] / len(train)
            else:
                p["probability" if kind == "binary" else "prediction"] = predictions
                p["dummy_mean"] = float(train[target].mean())
                p["dummy_median"] = float(train[target].median())
            p.to_parquet(tmp / f"{role}.parquet", index=False)
        joblib.dump(fitted, tmp / "estimator.joblib", compress=3)
        if members:
            pd.concat(members).to_parquet(tmp / "inner_membership.parquet", index=False)
        write_json(tmp / "trials.json", trials)
        info.update(status="complete", feature_names=fitted.transform.names,
                    optimizer_gradient_max=getattr(fitted.model, "gradient_max", None))
    except ValueError as exc:
        info.update(status="blocked", error=str(exc))
    write_json(tmp / "summary.json", info)
    publish_bundle(tmp, dest, info)


def run(root):
    root = Path(root)
    data_path, cfg = build(root)
    identity = dict(config=cfg, data_manifest=sha256(data_path / "manifest.json"),
                    code={p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("models.py")]},
                    reused_transform=sha256(root / "src/garmin_comprehensive/models.py"))
    dest = root / "private/root_cause_analysis/runs" / digest_object(identity)[:16]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "jobs").mkdir(exist_ok=True)
    write_json(root / "private/root_cause_analysis/current.json", dict(data=str(data_path.relative_to(root)), run=str(dest.relative_to(root))))
    if (dest / "manifest.json").exists():
        verify_bundle(dest)
        return dest
    write_json(dest / "identity.json", identity)
    f = pd.read_parquet(data_path / "development.parquet")
    matched = f[f.matched].copy()
    tasks = [("target_" + t, "binary", matched) for t in TASKS]
    tasks += [("ordinal_" + t, "ordinal", matched) for t in ITEMS]
    tasks += [("pss_combo", "continuous", matched)]
    probe = pd.read_parquet(data_path / "activity_probe.parquet")
    tasks += [("target", "binary", probe[probe.eligible].copy())]
    with threadpool_limits(limits=1):
        for target, kind, frame in tasks:
            models = specs() if target != "target" else [s for s in specs() if s["name"] == "hr"]
            for spec in models:
                for fold in range(5):
                    job(dest, frame, target, kind, spec, fold, cfg)
                print("Completed", kind, target, spec["name"], flush=True)
    return dest


if __name__ == "__main__":
    print(run(Path.cwd()))
