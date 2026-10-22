"""Immutable fold jobs and fixed feature provenance; no EMA-history fallback."""

import argparse
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
from .data import build
from .models import cohort, nested, specifications


def job(root, folder, features, task, spec, fold, cfg):
    with threadpool_limits(limits=1):
        dest = folder / "jobs" / f"{task}__{spec['name']}__fold{fold}"
        if dest.exists():
            verify_bundle(dest)
            return "cached"
        f = cohort(pd.read_parquet(features / "development.parquet"), spec["cohort"])
        train, valid = f[f.n_fold.ne(fold)], f[f.n_fold.eq(fold)]
        info = dict(task=task, spec=spec, fold=fold)
        for role, z in [("train", train), ("validation", valid)]:
            info[role] = dict(
                responses=len(z),
                participants=z.n_participant.nunique(),
                class_support=np.bincount(z[f"target_{task}"].to_numpy(int), minlength=2).tolist(),
            )
        tmp = Path(tempfile.mkdtemp(prefix=".building_", dir=folder / "jobs"))
        try:
            if not len(valid) or train[f"target_{task}"].nunique() != 2:
                raise ValueError("Infeasible outer cohort.")
            fitted, best, trials, splits = nested(train, task, spec, cfg)
            for role, z in [("train", train), ("validation", valid)]:
                predictors = z.drop(
                    columns=[c for c in z if c.startswith(("target_", "item_")) or c == "pss_combo"]
                )
                p = z[
                    ["n_ema_id", "n_participant", "n_fold", "n_survey_category", "supported60"]
                ].copy()
                p["target"] = z[f"target_{task}"].to_numpy(int)
                p["probability"] = fitted.probability(predictors)
                p["dummy_probability"] = float(train[f"target_{task}"].mean())
                p.to_parquet(tmp / f"{role}.parquet", index=False)
            members = []
            for k, (a, b) in enumerate(splits):
                for role, ix in [("train", a), ("validation", b)]:
                    members.append(
                        train.iloc[ix][["n_ema_id", "n_participant"]].assign(
                            inner_fold=k, role=role
                        )
                    )
            pd.concat(members).to_parquet(tmp / "inner_membership.parquet", index=False)
            joblib.dump(fitted, tmp / "estimator.joblib", compress=3)
            write_json(tmp / "trials.json", trials)
            write_json(
                tmp / "features.json",
                {
                    "names": fitted.transform.names,
                    "coefficients": fitted.model.coef_[0].tolist()
                    if hasattr(fitted.model, "coef_")
                    else None,
                },
            )
            info.update(status="complete", best=best)
        except Exception as exc:
            info.update(status="failed", error=type(exc).__name__)
            (tmp / "private_error.txt").write_text(str(exc))
        write_json(tmp / "summary.json", info)
        publish_bundle(tmp, dest, info)
        return info["status"]


def run(root, workers=4):
    root = Path(root)
    assert (root / "reports/temporal_personalization/audit/longitudinal.json").exists()
    features = build(root)
    cfg = json.loads((root / "config/temporal_personalization.json").read_text())
    identity = dict(
        config=cfg,
        feature_manifest=sha256(features / "manifest.json"),
        specification=sha256(root / "reports/temporal_personalization/specification.md"),
        code={p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("models.py")]},
        reused_transform=sha256(root / "src/garmin_comprehensive/models.py"),
    )
    folder = root / "private/temporal_personalization/runs" / digest_object(identity)[:16]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "jobs").mkdir(exist_ok=True)
    pointer = {"run": str(folder.relative_to(root)), "features": str(features.relative_to(root))}
    write_json(root / "private/temporal_personalization/current.json", pointer)
    if (folder / "manifest.json").exists():
        verify_bundle(folder)
        return folder
    if not (folder / "identity.json").exists():
        write_json(folder / "identity.json", identity)
        write_json(folder / "specifications.json", specifications())
    tasks = [(t, s, k) for t in ["A", "B"] for s in specifications() for k in range(5)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(job, root, folder, features, t, s, k, cfg): (t, s["name"], k)
            for t, s, k in tasks
        }
        for i, r in enumerate(as_completed(futures), 1):
            print(i, len(tasks), *futures[r], r.result(), flush=True)
    return folder


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    print(run(Path.cwd(), p.parse_args().workers))
