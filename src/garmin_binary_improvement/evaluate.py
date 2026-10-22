"""Paired, cohort-matched aggregate evaluation of sealed fold predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from garmin_binary_6hour.metrics import score, bootstrap, interval_fields
from garmin_comprehensive_regression.evaluate import finite
from garmin_preprocessing.common import verify_bundle, write_json


def decision_metrics(y, p, thresholds):
    base = score(y, p)
    hard = score(y, (np.asarray(p) >= np.asarray(thresholds)).astype(float))
    for key in [
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "negative_precision",
        "negative_f1",
        "accuracy",
        "predicted_positive",
        "tn",
        "fp",
        "fn",
        "tp",
    ]:
        base[key] = hard[key]
    return base


def evaluate(root):
    root = Path(root)
    run = root / json.loads((root / "private/binary_improvement/current.json").read_text())["run"]
    out = root / "reports/binary_improvement" / run.name
    if (out / "manifest.json").exists():
        verify_bundle(out)
        return out
    out.mkdir(exist_ok=True)
    cfg = json.loads((run / "identity.json").read_text())["config"]
    paths = sorted((run / "jobs").glob("*/summary.json"))
    assert len(paths) == 600, "Do not publish partial experiment results."
    controls = run / json.loads((run / "context_controls.json").read_text())["path"]
    verify_bundle(controls)
    control_paths = sorted((controls / "jobs").glob("*/summary.json"))
    assert len(control_paths) == 40
    paths += control_paths
    all_p, fold_support, selected, resampling, features, failures, train_metrics = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    specs = {}
    for path in paths:
        verify_bundle(path.parent)
        info = json.loads(path.read_text())
        task, model, fold = info["task"], info["spec"]["name"], info["fold"]
        specs[model] = info["spec"]
        for role in ["train", "validation"]:
            fold_support.append(
                dict(
                    task=task,
                    model=model,
                    fold=fold,
                    role=role,
                    cohort=info["spec"]["cohort"],
                    responses=info[role + "_responses"],
                    participants=info[role + "_participants"],
                    class0=info[role + "_support"][0],
                    class1=info[role + "_support"][1],
                    class0_participants=info[role + "_participants_by_class"][0],
                    class1_participants=info[role + "_participants_by_class"][1],
                )
            )
        if info["status"] != "complete":
            failures.append(dict(task=task, model=model, fold=fold, error=info["error_type"]))
            continue
        p = pd.read_parquet(path.parent / "validation_predictions.parquet").assign(
            task=task, model=model, cohort=info["spec"]["cohort"]
        )
        all_p.append(p)
        tr = pd.read_parquet(path.parent / "train_predictions.parquet")
        train_metrics.append(
            dict(task=task, model=model, fold=fold, **score(tr.target, tr.probability))
        )
        selected.append(
            dict(
                task=task,
                model=model,
                fold=fold,
                threshold=info["threshold"],
                n_iter=info["n_iter"],
                features=info["transformed_features"],
                reused=info["previous_prediction_replay"],
                **info["best"]["parameters"],
            )
        )
        for item in json.loads((path.parent / "resampling.json").read_text()):
            resampling.append(dict(task=task, model=model, fold=fold, **item))
        coef = json.loads((path.parent / "features.json").read_text())
        for name, value in zip(coef["names"], coef["coefficients"], strict=True):
            features.append(
                dict(
                    task=task,
                    model=model,
                    fold=fold,
                    feature=name,
                    coefficient=value,
                    nonzero=abs(value) > 1e-8,
                )
            )
    predictions = pd.concat(all_p, ignore_index=True)
    # Entirely report/skip an incomplete model rather than silently dropping difficult folds.
    incomplete = {(x["task"], x["model"]) for x in failures}
    if incomplete:
        predictions = predictions[
            [(t, m) not in incomplete for t, m in zip(predictions.task, predictions.model)]
        ]
    baseline_views = []
    for (task, cohort_name), g in predictions.groupby(["task", "cohort"]):
        model = sorted(g.model.unique())[0]
        p = g[g.model.eq(model)].copy()
        for kind in ["constant", "majority"]:
            view = p.copy()
            view["model"] = f"{cohort_name}_{kind}"
            view["probability"] = (
                view.dummy_probability
                if kind == "constant"
                else (view.dummy_probability > 0.5).astype(float)
            )
            view["decision_threshold"] = 0.5
            view["prior_corrected_probability"] = view.probability
            baseline_views.append(view)
    predictions = pd.concat([predictions, *baseline_views], ignore_index=True)
    predictions.to_parquet(run / "all_predictions.parquet", index=False)
    metrics, folds, thresholds, calibration, curves, perclass, surveys, influence = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    draws, indexed = {}, {}
    for (task, model), g in predictions.groupby(["task", "model"], sort=True):
        g = g.sort_values("n_ema_id").reset_index(drop=True)
        assert g.n_ema_id.is_unique and g.n_fold.nunique() == 5
        indexed[(task, model)] = g
        meta = dict(
            task=task,
            model=model,
            cohort=g.cohort.iloc[0],
            responses=len(g),
            participants=g.n_participant.nunique(),
            prevalence=float(g.target.mean()),
        )
        values = bootstrap(g, cfg)
        draws[(task, model)] = values
        fm = []
        for fold, part in g.groupby("n_fold"):
            s = score(part.target, part.probability)
            folds.append(dict(task=task, model=model, fold=int(fold), **s))
            fm.append(s["auroc"])
        metrics.append(
            dict(
                **meta,
                **score(g.target, g.probability),
                **interval_fields(values),
                fold_auroc_mean=float(np.mean(fm)),
                fold_auroc_min=min(fm),
                fold_auroc_max=max(fm),
            )
        )
        for mode, p, thresholds_used in [
            ("default", g.probability, 0.5),
            ("inner_BA_threshold", g.probability, g.decision_threshold),
            ("prior_corrected_default", g.prior_corrected_probability, 0.5),
        ]:
            s = decision_metrics(g.target, p, thresholds_used)
            thresholds.append(dict(**meta, mode=mode, **s))
            for cls in [0, 1]:
                perclass.append(
                    dict(
                        task=task,
                        model=model,
                        mode=mode,
                        class_id=cls,
                        support=s["positive_support" if cls else "negative_support"],
                        precision=s["precision" if cls else "negative_precision"],
                        recall=s["sensitivity" if cls else "specificity"],
                        f1=s["f1" if cls else "negative_f1"],
                    )
                )
        for mode, col in [
            ("raw", "probability"),
            ("prior_corrected", "prior_corrected_probability"),
        ]:
            bins = np.minimum((g[col].to_numpy() * 10).astype(int), 9)
            for b in range(10):
                z = g[bins == b]
                calibration.append(
                    dict(
                        task=task,
                        model=model,
                        mode=mode,
                        bin=b,
                        count=len(z),
                        mean_probability=float(z[col].mean()) if len(z) else None,
                        observed_fraction=float(z.target.mean()) if len(z) else None,
                    )
                )
        fp, tp, _ = roc_curve(g.target, g.probability)
        precision, recall, _ = precision_recall_curve(g.target, g.probability)
        grid = np.linspace(0, 1, 101)
        curves.append(
            dict(
                task=task,
                model=model,
                grid=grid.tolist(),
                tpr=np.interp(grid, fp, tp).tolist(),
                precision=np.interp(grid, recall[::-1], precision[::-1]).tolist(),
            )
        )
        for category, z in g.groupby("n_survey_category"):
            surveys.append(
                dict(
                    task=task,
                    model=model,
                    survey=category,
                    responses=len(z),
                    participants=z.n_participant.nunique(),
                    prevalence=float(z.target.mean()),
                    **score(z.target, z.probability),
                )
            )
        if model in ["context", "full360", "full720", "full1440", "compact_none", "compact_smote"]:
            from sklearn.metrics import roc_auc_score

            loo = [
                roc_auc_score(z.target, z.probability)
                for person in g.n_participant.unique()
                if (z := g[g.n_participant.ne(person)]).target.nunique() == 2
            ]
            influence.append(
                dict(
                    task=task,
                    model=model,
                    leave_one_participant_out_min=min(loo),
                    leave_one_participant_out_max=max(loo),
                )
            )
        print(task, model, "evaluated", flush=True)
    paired = []
    for (task, model), g in indexed.items():
        cohort_name = g.cohort.iloc[0]
        context = (
            "context"
            if cohort_name == "all"
            else "support_context"
            if cohort_name == "supported"
            else f"{cohort_name}_context"
        )
        references = [context, f"{cohort_name}_constant", f"{cohort_name}_majority"]
        if cohort_name == "all":
            references += ["original360", "full360"]
        if model.startswith("compact_"):
            references += ["compact_none"]
        if model.startswith("context_compact_"):
            references += ["context_compact_none"]
            references += [model.replace("context_compact_", "context_control_")]
        if model == "minute360":
            references += ["original360"]
        if model.endswith("_wake"):
            references += [f"{cohort_name}_original60"]
        for reference in dict.fromkeys(references):
            if reference == model or (task, reference) not in indexed:
                continue
            b = indexed[(task, reference)]
            assert g.n_ema_id.tolist() == b.n_ema_id.tolist()
            assert np.array_equal(g.target, b.target)
            a_score, b_score = score(g.target, g.probability), score(b.target, b.probability)
            paired.append(
                dict(
                    task=task,
                    model=model,
                    reference=reference,
                    cohort=cohort_name,
                    **{k: a_score[k] - b_score[k] for k in ["auroc", "ap", "balanced_accuracy"]},
                    **interval_fields(draws[(task, model)] - draws[(task, reference)]),
                )
            )
    # Restrict original all-row fits to support cohort without conflating evaluation restriction/refitting.
    restricted = []
    for task in ["A", "B"]:
        if (task, "support_context") not in indexed:
            continue
        ids = set(indexed[(task, "support_context")].n_ema_id)
        for model in ["context", "original360", "full360"]:
            g = indexed[(task, model)]
            g = g[g.n_ema_id.isin(ids)]
            restricted.append(
                dict(
                    task=task,
                    model=model,
                    responses=len(g),
                    participants=g.n_participant.nunique(),
                    **score(g.target, g.probability),
                    **interval_fields(bootstrap(g, cfg)),
                )
            )
    stability = []
    for (task, model, name), g in pd.DataFrame(features).groupby(["task", "model", "feature"]):
        stability.append(
            dict(
                task=task,
                model=model,
                feature=name,
                folds_present=len(g),
                nonzero_folds=int(g.nonzero.sum()),
                positive_folds=int(g.coefficient.gt(1e-8).sum()),
                negative_folds=int(g.coefficient.lt(-1e-8).sum()),
                mean_absolute_coefficient=float(g.coefficient.abs().mean()),
            )
        )
    sampling_summary = (
        pd.DataFrame(resampling)
        .query("inner_fold == 'outer_refit'")
        .groupby(["task", "model"])[
            [
                "augmented_rows",
                "synthetic_rows",
                "cross_person_rows",
                "cross_category_rows",
                "donor_missing_rows",
            ]
        ]
        .sum()
        .reset_index()
        .to_dict("records")
    )
    for name, value in [
        ("resampling_summary", sampling_summary),
        ("metrics", metrics),
        ("paired", paired),
        ("fold_metrics", folds),
        ("fold_support", fold_support),
        ("thresholds", thresholds),
        ("per_class", perclass),
        ("calibration", calibration),
        ("curves", curves),
        ("survey_metrics", surveys),
        ("influence", influence),
        ("selected", selected),
        ("resampling", resampling),
        ("selection_stability", stability),
        ("failures", failures),
        ("restricted_evaluation", restricted),
        ("training_metrics", train_metrics),
    ]:
        write_json(out / f"{name}.json", finite(value))
    np.savez_compressed(
        run / "bootstrap_draws.npz", **{f"{t}__{m}": v for (t, m), v in draws.items()}
    )
    return out


if __name__ == "__main__":
    print(evaluate(Path.cwd()))
