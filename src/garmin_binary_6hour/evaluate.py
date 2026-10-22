"""Aggregate paired evaluation of completed logistic experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from garmin_comprehensive_regression.evaluate import finite
from garmin_preprocessing.common import verify_bundle, write_json
from .metrics import bootstrap, interval_fields, score
from .models import WINDOWS


def evaluate(root):
    root = Path(root)
    run = root / json.loads((root / "private/binary_6hour/current.json").read_text())["run"]
    out = root / "reports/binary_6hour" / run.name
    if (out / "manifest.json").exists():
        verify_bundle(out)
        return out
    out.mkdir(exist_ok=True)
    cfg = json.loads((run / "identity.json").read_text())["config"]
    specs = {s["name"]: s for s in json.loads((run / "specifications.json").read_text())}
    paths = sorted((run / "jobs").glob("*/summary.json"))
    if len(paths) != len(specs) * 10:
        raise ValueError("Experiment jobs incomplete.")
    predictions, settings, fold_metrics, support, training, coefficients, trials = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    failures = []
    for path in paths:
        verify_bundle(path.parent)
        info = json.loads(path.read_text())
        task, name, fold = info["task"], info["spec"]["name"], info["fold"]
        tag = dict(task=task, model=name, fold=fold, cohort=info["spec"]["cohort"])
        for role in ["train", "validation"]:
            support.append(
                dict(
                    **tag,
                    role=role,
                    responses=info[f"{role}_responses"],
                    participants=info[f"{role}_participants"],
                    class0=info[f"{role}_support"][0],
                    class1=info[f"{role}_support"][1],
                    class0_participants=info[f"{role}_participants_by_class"][0],
                    class1_participants=info[f"{role}_participants_by_class"][1],
                )
            )
        if info["status"] != "complete":
            failures.append(dict(**tag, error=info["error_type"]))
            continue
        pred = pd.read_parquet(path.parent / "validation_predictions.parquet")
        pred["model"], pred["task"] = name, task
        predictions.append(pred)
        fold_metrics.append(dict(**tag, **score(pred.target, pred.probability)))
        tr = pd.read_parquet(path.parent / "train_predictions.parquet")
        training.append(dict(**tag, **score(tr.target, tr.probability)))
        settings.append(
            dict(
                **tag,
                **info["best"]["parameters"],
                inner_auroc=info["best"]["mean_auroc"],
                n_iter=info["n_iter"],
                transformed=info["transformed_features"],
                nonzero=info["nonzero_features"],
            )
        )
        ts = json.loads((path.parent / "trials.json").read_text())
        for t in ts:
            trials.append(
                dict(
                    **tag,
                    **t["parameters"],
                    mean_inner_auroc=t["mean_auroc"],
                    converged=all(t["converged"]),
                    max_iterations=max(t["iterations"]),
                )
            )
        fs = json.loads((path.parent / "features.json").read_text())
        for feature, coefficient in zip(fs["names"], fs["coefficients"], strict=True):
            coefficients.append(
                dict(
                    **tag,
                    feature=feature,
                    coefficient=coefficient,
                    selected=abs(coefficient) > 1e-8,
                )
            )
    if failures:
        write_json(out / "failures.json", failures)
        raise ValueError("Failed jobs require explicit review; no silent complete-case ranking.")
    all_pred = pd.concat(predictions, ignore_index=True)
    for task in ["A", "B"]:
        for co, source in [("all", "context_u"), ("complete", "complete_context_u")]:
            dummy = all_pred[(all_pred.task == task) & (all_pred.model == source)].copy()
            dummy["probability"] = dummy.dummy_probability
            name = "majority" if co == "all" else "complete_majority"
            dummy["model"] = name
            predictions.append(dummy)
            specs[name] = dict(name=name, cohort=co, weighted=False, mode="dummy", block="context")
            for fold, f in dummy.groupby("n_fold"):
                fold_metrics.append(
                    dict(
                        task=task,
                        model=name,
                        cohort=co,
                        fold=int(fold),
                        **score(f.target, f.probability),
                    )
                )
    all_pred = pd.concat(predictions, ignore_index=True)
    all_pred.to_parquet(run / "all_predictions.parquet", index=False)
    frames, draws, summaries = {}, {}, {}
    metrics, curves, survey, per_class = [], [], [], []
    grid = np.linspace(0, 1, 101)
    for (task, name), f in all_pred.groupby(["task", "model"]):
        f = f.sort_values("n_ema_id").reset_index(drop=True)
        assert not f.n_ema_id.duplicated().any() and f.n_fold.nunique() == 5
        key = (task, name)
        frames[key] = f
        draws[key] = bootstrap(f, cfg)
        summaries[key] = score(f.target, f.probability)
        metrics.append(
            dict(
                task=task,
                model=name,
                cohort=specs[name]["cohort"],
                responses=len(f),
                participants=f.n_participant.nunique(),
                **summaries[key],
                **interval_fields(draws[key]),
            )
        )
        s = summaries[key]
        for class_id in [0, 1]:
            per_class.append(
                dict(
                    task=task,
                    model=name,
                    class_id=class_id,
                    support=s["positive_support"] if class_id else s["negative_support"],
                    precision=s["precision"] if class_id else s["negative_precision"],
                    recall=s["sensitivity"] if class_id else s["specificity"],
                    f1=s["f1"] if class_id else s["negative_f1"],
                )
            )
        fpr, tpr, _ = roc_curve(f.target, f.probability)
        precision, recall, _ = precision_recall_curve(f.target, f.probability)
        curves.append(
            dict(
                task=task,
                model=name,
                grid=grid.tolist(),
                roc_tpr=np.interp(grid, fpr, tpr).tolist(),
                pr_precision=np.interp(grid, recall[::-1], precision[::-1]).tolist(),
            )
        )
        for category, g in f.groupby("n_survey_category"):
            survey.append(
                dict(
                    task=task,
                    model=name,
                    survey=category,
                    responses=len(g),
                    participants=g.n_participant.nunique(),
                    **score(g.target, g.probability),
                )
            )
    differences = []
    seen = set()

    def compare(task, a, b, reason):
        if (task, a, b) in seen or a == b:
            return
        seen.add((task, a, b))
        fa, fb = frames[task, a], frames[task, b]
        for c in ["n_ema_id", "n_participant", "n_fold", "target"]:
            assert fa[c].equals(fb[c]), "Paired membership mismatch."
        delta = draws[task, a] - draws[task, b]
        differences.append(
            dict(
                task=task,
                model=a,
                baseline=b,
                reason=reason,
                responses=len(fa),
                participants=fa.n_participant.nunique(),
                **{
                    metric + "_difference": summaries[task, a][metric] - summaries[task, b][metric]
                    for metric in ["auroc", "ap", "balanced_accuracy"]
                },
                **interval_fields(delta),
            )
        )

    for task in ["A", "B"]:
        for name, spec in specs.items():
            ctx = (
                "complete_context_u"
                if spec["cohort"] == "complete"
                else "context_w"
                if spec["weighted"]
                else "context_u"
            )
            dummy = "complete_majority" if spec["cohort"] == "complete" else "majority"
            compare(task, name, ctx, "physiology_or_control_vs_matching_context")
            compare(task, name, dummy, "vs_matching_majority")
        for weight in ["u", "w"]:
            for block in ["original", "hr", "activity", "full", "sparse"]:
                for m in [120, 360]:
                    compare(task, f"{block}{m}_{weight}", f"{block}60_{weight}", "paired_window")
        for m in WINDOWS:
            for b in ["original", "hr", "activity", "full", "sparse"]:
                compare(task, f"{b}{m}_w", f"{b}{m}_u", "class_weighting")
            for b in ["corr", "availability", "no_flags", "missing_only", "sparse"]:
                compare(task, f"{b}{m}_u", f"full{m}_u", "selection_or_handling")
            for b in ["original", "hr", "activity"]:
                compare(task, f"full{m}_u", f"{b}{m}_u", "feature_expansion")
        compare(task, "complete_full360_u", "complete_full60_u", "common_complete_refit_window")
        compare(task, "context_w", "context_u", "class_weighting")
    coeff = pd.DataFrame(coefficients)
    stability = (
        coeff.groupby(["task", "model", "feature"])
        .agg(
            fitted_folds=("fold", "nunique"),
            selected_folds=("selected", "sum"),
            mean_coefficient=("coefficient", "mean"),
            min_coefficient=("coefficient", "min"),
            max_coefficient=("coefficient", "max"),
        )
        .reset_index()
    )
    # Features absent in a fold have zero coefficient; report magnitudes with that denominator.
    magnitude = (
        coeff.assign(magnitude=coeff.coefficient.abs())
        .groupby(["task", "model", "feature"])
        .magnitude.sum()
        / 5
    )
    stability = stability.merge(
        magnitude.rename("mean_absolute_coefficient").reset_index(), on=["task", "model", "feature"]
    )
    restricted = []
    for task in ["A", "B"]:
        keys = set(frames[task, "complete_context_u"].n_ema_id)
        for name in ["majority", "context_u", "full60_u", "full120_u", "full360_u"]:
            f = frames[task, name]
            f = f[f.n_ema_id.isin(keys)]
            restricted.append(
                dict(
                    task=task,
                    model=name,
                    training_population="all_valid",
                    evaluation_population="common_original_complete",
                    responses=len(f),
                    participants=f.n_participant.nunique(),
                    **score(f.target, f.probability),
                )
            )
    np.savez_compressed(
        run / "bootstrap_draws.npz", **{f"{t}__{n}": v for (t, n), v in draws.items()}
    )
    outputs = dict(
        metrics=metrics,
        paired_differences=differences,
        fold_metrics=fold_metrics,
        training_metrics=training,
        fold_support=support,
        settings=settings,
        trials=trials,
        selection_stability=stability.to_dict("records"),
        survey_metrics=survey,
        per_class=per_class,
        restricted_evaluation=restricted,
        curves=curves,
        failures=failures,
        configuration=cfg,
        feature_blocks=json.loads((run / "feature_blocks.json").read_text()),
    )
    for name, value in outputs.items():
        write_json(out / f"{name}.json", finite(value))
    print(
        f"Evaluated {len(metrics)} model/task views with paired participant intervals.", flush=True
    )
    return out


if __name__ == "__main__":
    print(evaluate(Path.cwd()))
