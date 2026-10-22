"""Matched OOF evaluation with participant clusters and distinct cohort denominators."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from garmin_binary_6hour.metrics import score, bootstrap, interval_fields
from garmin_comprehensive_regression.evaluate import finite
from garmin_preprocessing.common import write_json, verify_bundle


def evaluate(root):
    root = Path(root)
    pointer = json.loads((root / "private/temporal_personalization/current.json").read_text())
    run = root / pointer["run"]
    out = root / "reports/temporal_personalization" / run.name
    if (out / "manifest.json").exists():
        verify_bundle(out)
        return out
    out.mkdir(exist_ok=True)
    cfg = json.loads((run / "identity.json").read_text())["config"]
    paths = sorted((run / "jobs").glob("*/summary.json"))
    assert len(paths) == 210
    predictions = []
    supports = []
    selected = []
    failures = []
    coefficients = []
    training = []
    for p in paths:
        verify_bundle(p.parent)
        info = json.loads(p.read_text())
        t, s, k = info["task"], info["spec"], info["fold"]
        for role in ["train", "validation"]:
            supports.append(
                dict(task=t, model=s["name"], cohort=s["cohort"], fold=k, role=role, **info[role])
            )
        if info["status"] != "complete":
            failures.append(dict(task=t, model=s["name"], fold=k, error=info["error"]))
            continue
        z = pd.read_parquet(p.parent / "validation.parquet").assign(
            task=t, model=s["name"], cohort=s["cohort"]
        )
        predictions.append(z)
        tr = pd.read_parquet(p.parent / "train.parquet")
        training.append(dict(task=t, model=s["name"], fold=k, **score(tr.target, tr.probability)))
        trials = json.loads((p.parent / "trials.json").read_text())
        selected.append(
            dict(task=t, model=s["name"], fold=k, **info["best"], candidates=len(trials))
        )
        fs = json.loads((p.parent / "features.json").read_text())
        if fs["coefficients"] is not None:
            coefficients.extend(
                dict(task=t, model=s["name"], fold=k, feature=n, coefficient=v)
                for n, v in zip(fs["names"], fs["coefficients"], strict=True)
            )
    predictions = pd.concat(predictions, ignore_index=True)
    for bad in {(x["task"], x["model"]) for x in failures}:
        predictions = predictions[~(predictions.task.eq(bad[0]) & predictions.model.eq(bad[1]))]
    constants = []
    for (task, cohort), g in predictions.groupby(["task", "cohort"]):
        z = g[g.model.eq(sorted(g.model.unique())[0])].copy()
        z["model"] = cohort + "_constant"
        z["probability"] = z.dummy_probability
        constants.append(z)
    predictions = pd.concat([predictions, *constants], ignore_index=True)
    predictions.to_parquet(run / "predictions.parquet", index=False)
    metrics = []
    folds = []
    participants = []
    participant_summary = []
    calibration = []
    strata = []
    draws = {}
    indexed = {}
    for (task, model), g in predictions.groupby(["task", "model"]):
        g = g.sort_values("n_ema_id").reset_index(drop=True)
        assert g.n_ema_id.is_unique
        indexed[(task, model)] = g
        d = bootstrap(g, cfg)
        draws[(task, model)] = d
        fs = []
        pairs = []
        for k, z in g.groupby("n_fold"):
            s = score(z.target, z.probability)
            folds.append(
                dict(
                    task=task,
                    model=model,
                    fold=int(k),
                    responses=len(z),
                    participants=z.n_participant.nunique(),
                    prevalence=float(z.target.mean()),
                    mean_probability=float(z.probability.mean()),
                    **s,
                )
            )
            if s["auroc"] is not None:
                fs.append(s["auroc"])
                pairs.append(int(z.target.sum() * (len(z) - z.target.sum())))
        ps = []
        for person, z in g.groupby("n_participant"):
            s = score(z.target, z.probability)
            participants.append(
                dict(task=task, model=model, n_participant=person, responses=len(z), **s)
            )
            if s["auroc"] is not None:
                ps.append(s["auroc"])
        participant_summary.append(
            dict(
                task=task,
                model=model,
                participants=g.n_participant.nunique(),
                estimable_participants=len(ps),
                mean_person_auroc=float(np.mean(ps)) if ps else None,
                median_person_auroc=float(np.median(ps)) if ps else None,
            )
        )
        metrics.append(
            dict(
                task=task,
                model=model,
                cohort=g.cohort.iloc[0],
                responses=len(g),
                participants=g.n_participant.nunique(),
                prevalence=float(g.target.mean()),
                **score(g.target, g.probability),
                **interval_fields(d),
                equal_fold_auroc=float(np.mean(fs)),
                within_fold_pair_weighted_auroc=float(np.average(fs, weights=pairs)),
                minimum_fold_auroc=min(fs),
                maximum_fold_auroc=max(fs),
            )
        )
        for b in range(10):
            z = g[(np.minimum((g.probability * 10).astype(int), 9) == b)]
            calibration.append(
                dict(
                    task=task,
                    model=model,
                    bin=b,
                    responses=len(z),
                    mean_probability=float(z.probability.mean()) if len(z) else None,
                    prevalence=float(z.target.mean()) if len(z) else None,
                )
            )
        for field in ["n_survey_category", "supported60"]:
            for value, z in g.groupby(field):
                strata.append(
                    dict(
                        task=task,
                        model=model,
                        stratum=field,
                        value=str(value),
                        responses=len(z),
                        participants=z.n_participant.nunique(),
                        prevalence=float(z.target.mean()),
                        **score(z.target, z.probability),
                    )
                )
    paired = []
    for (task, model), g in indexed.items():
        co = g.cohort.iloc[0]
        context = "context" if co == "all" else co + "_context"
        refs = [context, co + "_constant"]
        if model == "raw":
            refs += ["activity"]
        if model == "personal":
            refs += ["raw", "reference", "quality", "missingness"]
        if model == "extended":
            refs += ["personal"]
        if model == "personal_quality":
            refs += ["personal", "quality"]
        if model == "observed_personal":
            refs += ["observed_raw", "observed_reference"]
        if model == "hgb_personal":
            refs += ["hgb_context", "hgb_raw"]
        if model == "hgb_raw":
            refs += ["hgb_context"]
        for ref in dict.fromkeys(refs):
            if model == ref or (task, ref) not in indexed:
                continue
            b = indexed[(task, ref)]
            assert g.n_ema_id.tolist() == b.n_ema_id.tolist() and np.array_equal(g.target, b.target)
            a_score, b_score = score(g.target, g.probability), score(b.target, b.probability)
            paired.append(
                dict(
                    task=task,
                    model=model,
                    reference=ref,
                    cohort=co,
                    **{
                        k: a_score[k] - b_score[k]
                        for k in ["auroc", "ap", "balanced_accuracy", "brier"]
                    },
                    **interval_fields(draws[(task, model)] - draws[(task, ref)]),
                )
            )
    # No online-history metrics are fabricated for the unavailable comparison rows.
    comparison = []
    labels = {
        "context": "Context only",
        "physiology": "Physiology only",
        "reference": "Earlier wearable reference + context",
        "personal": "Earlier wearable reference + current deviations + context",
    }
    for m in metrics:
        if m["model"] in labels:
            comparison.append({**m, "label": labels[m["model"]], "status": "executed_task_A"})
    for task in ["A", "B"]:
        for name in [
            "Previous EMA only",
            "Previous EMA + physiology",
            "Personalized historical EMA baseline",
            "Personalized historical EMA baseline + physiology",
        ]:
            comparison.append(
                dict(
                    task=task,
                    label=name,
                    status="blocked_missing_answer_availability",
                    responses=0,
                    participants=0,
                    auroc=None,
                    ap=None,
                    balanced_accuracy=None,
                    auroc_low=None,
                    auroc_high=None,
                )
            )
    pd.DataFrame(participants).to_parquet(run / "participant_metrics.parquet", index=False)
    np.savez_compressed(
        run / "bootstrap_draws.npz", **{f"{t}__{m}": v for (t, m), v in draws.items()}
    )
    for name, obj in [
        ("metrics", metrics),
        ("paired", paired),
        ("folds", folds),
        ("fold_support", supports),
        ("selected", selected),
        ("failures", failures),
        ("coefficients", coefficients),
        ("training_metrics", training),
        ("participant_summary", participant_summary),
        ("calibration", calibration),
        ("strata", strata),
        ("comparison", comparison),
    ]:
        write_json(out / f"{name}.json", finite(obj))
    return out


if __name__ == "__main__":
    print(evaluate(Path.cwd()))
