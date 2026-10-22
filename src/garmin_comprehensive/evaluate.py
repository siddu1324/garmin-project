"""Aggregate matched OOF metrics, cluster uncertainty, calibration and selection stability."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from garmin_preprocessing.common import verify_bundle, write_json
from garmin_pss_classification.data import TASKS
from garmin_pss_classification.metrics import confusion, from_confusion
from .models import specs
from .run import get_data, metrics


def auc(y, p, k, weight=None):
    if k == 2:
        return roc_auc_score(y, p[:, 1], sample_weight=weight)
    return np.mean([roc_auc_score(y == j, p[:, j], sample_weight=weight) for j in range(k)])


def paired(a, b, k, draws, seed):
    x = a.sort_values("n_ema_id")
    y = b.sort_values("n_ema_id")
    assert x.n_ema_id.equals(y.n_ema_id) and np.array_equal(x.target, y.target)
    truth = x.target.to_numpy(int)
    pa = x[[f"p{i}" for i in range(k)]].to_numpy()
    pb = y[[f"p{i}" for i in range(k)]].to_numpy()
    people, inverse = np.unique(x.n_participant, return_inverse=True)
    ca = np.stack(
        [
            confusion(truth[inverse == i], pa[inverse == i].argmax(axis=1), k)
            for i in range(len(people))
        ]
    )
    cb = np.stack(
        [
            confusion(truth[inverse == i], pb[inverse == i].argmax(axis=1), k)
            for i in range(len(people))
        ]
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(people), size=(draws, len(people)))
    sa = from_confusion(ca[samples].sum(axis=1))
    sb = from_confusion(cb[samples].sum(axis=1))
    pointa = from_confusion(ca.sum(axis=0))
    pointb = from_confusion(cb.sum(axis=0))
    out = dict(responses=len(x), participants=len(people), draws=draws)
    for name in ["macro_f1", "balanced_accuracy"]:
        lo, hi = np.quantile(sa[name] - sb[name], [0.025, 0.975])
        out[name + "_difference"] = float(pointa[name] - pointb[name])
        out[name + "_low"] = float(lo)
        out[name + "_high"] = float(hi)
    differences = []
    for sample in samples:
        weight = np.bincount(sample, minlength=len(people))[inverse]
        if any(weight[truth == j].sum() == 0 for j in range(k)):
            continue
        differences.append(auc(truth, pa, k, weight) - auc(truth, pb, k, weight))
    lo, hi = np.quantile(differences, [0.025, 0.975])
    out.update(
        roc_auc_difference=float(auc(truth, pa, k) - auc(truth, pb, k)),
        roc_auc_low=float(lo),
        roc_auc_high=float(hi),
    )
    return out


def evaluate(root):
    root = Path(root)
    current = json.loads((root / "private/comprehensive_features/current.json").read_text())
    feature_path = root / current["features"]
    run = root / current["models"]
    public = root / "reports/comprehensive_features" / run.name
    if (public / "manifest.json").exists():
        verify_bundle(public)
        verify_bundle(run)
        return public
    cfg = json.loads((root / "config/comprehensive_features.json").read_text())
    data, blocks = get_data(feature_path)
    expected = len(TASKS) * len({s["name"] for s in specs(blocks)}) * 5
    jobs = sorted((run / "jobs").glob("*/summary.json"))
    if len(jobs) != expected:
        raise RuntimeError(
            f"Incomplete nested run: {len(jobs)}/{expected} jobs; no partial result publication."
        )
    predictions = []
    folds = []
    settings = []
    coefficients = []
    skipped = []
    for p in jobs:
        folder = p.parent
        verify_bundle(folder)
        info = json.loads(p.read_text())
        spec = info["spec"]
        task = info["task"]
        k = TASKS[task]
        flat = {key: value for key, value in info.items() if key not in ["spec", "best"]}
        flat.update(
            view=spec["name"],
            cohort="complete60" if spec["handling"] == "complete" else "all_valid",
        )
        if info["status"] != "complete":
            skipped.append(flat)
            continue
        pred = pd.read_parquet(folder / "predictions.parquet")
        pred["task"] = task
        pred["view"] = spec["name"]
        pred["cohort"] = flat["cohort"]
        predictions.append(pred)
        flat.update(
            **json.loads((folder / "validation_metrics.json").read_text()),
            training_metrics=json.loads((folder / "training_metrics.json").read_text()),
        )
        folds.append(flat)
        settings.append(
            dict(
                task=task,
                view=spec["name"],
                fold=info["fold"],
                **info["best"],
                transformed_features=info["transformed_features"],
                selected_nonzero=info["selected_nonzero"],
                n_iter=info["n_iter"],
            )
        )
        for c in json.loads((folder / "coefficients.json").read_text()):
            coefficients.append(dict(task=task, view=spec["name"], fold=info["fold"], **c))
    pred = pd.concat(predictions, ignore_index=True)
    # Dummies refit to each exact cohort/outer training membership, not full development prevalence.
    for task, k in TASKS.items():
        target = "target_" + task
        for cohort in ["all_valid", "complete60"]:
            f = data[data[target].notna() & data.context_available].copy()
            if cohort == "complete60":
                f = f[f[blocks["original60"]].notna().all(axis=1)]
            for fold in range(5):
                tr, va = f[f.n_fold != fold], f[f.n_fold == fold]
                counts = np.bincount(tr[target].to_numpy(int), minlength=k)
                for style in ["majority", "frequency"]:
                    p = np.tile(
                        counts / counts.sum()
                        if style == "frequency"
                        else np.eye(k)[counts.argmax()],
                        (len(va), 1),
                    )
                    v = va[["n_ema_id", "n_participant", "n_fold", "n_survey_category"]].copy()
                    v["target"] = va[target].to_numpy(int)
                    v["task"] = task
                    v["view"] = "dummy_" + style + ("_complete" if cohort == "complete60" else "")
                    v["cohort"] = cohort
                    for j in range(k):
                        v[f"p{j}"] = p[:, j]
                    predictions.append(v)
                    vals = metrics(v.target.to_numpy(int), p, v.n_participant.to_numpy(), k)[0]
                    folds.append(
                        dict(
                            task=task,
                            view=v.view.iloc[0],
                            fold=fold,
                            cohort=cohort,
                            status="complete",
                            training_responses=len(tr),
                            validation_responses=len(va),
                            training_participants=tr.n_participant.nunique(),
                            validation_participants=va.n_participant.nunique(),
                            training_support=counts.tolist(),
                            validation_support=np.bincount(v.target, minlength=k).tolist(),
                            **vals,
                        )
                    )
    pred = pd.concat(predictions, ignore_index=True)
    pooled = []
    per_class = []
    matrices = []
    survey_metrics = []
    calibration = []
    composition = []
    for (task, view), f in pred.groupby(["task", "view"], sort=True):
        k = TASKS[task]
        if f.n_fold.nunique() != 5:
            skipped.append(dict(task=task, view=view, reason="Incomplete OOF folds"))
            continue
        assert not f.n_ema_id.duplicated().any()
        prob = f[[f"p{j}" for j in range(k)]].to_numpy(float)
        truth = f.target.to_numpy(int)
        result, cm, per = metrics(truth, prob, f.n_participant.to_numpy(), k)
        pooled.append(
            dict(
                task=task,
                view=view,
                cohort=f.cohort.iloc[0],
                responses=len(f),
                participants=f.n_participant.nunique(),
                **result,
            )
        )
        matrices.append(dict(task=task, view=view, matrix=cm.tolist()))
        for j in range(k):
            per_class.append(
                dict(
                    task=task,
                    view=view,
                    target_class=j,
                    precision=float(per["precision"][j]),
                    recall=float(per["recall"][j]),
                    f1=float(per["f1"][j]),
                    support=int(per["support"][j]),
                )
            )
            edges = np.linspace(0, 1, 11)
            for n in range(10):
                choose = (prob[:, j] >= edges[n]) & (
                    prob[:, j] < (edges[n + 1] if n < 9 else 1.0000001)
                )
                if choose.any():
                    calibration.append(
                        dict(
                            task=task,
                            view=view,
                            target_class=j,
                            bin=n,
                            n=int(choose.sum()),
                            mean_probability=float(prob[choose, j].mean()),
                            observed_fraction=float((truth[choose] == j).mean()),
                        )
                    )
        for category, g in f.groupby("n_survey_category"):
            vals = metrics(
                g.target.to_numpy(int),
                g[[f"p{j}" for j in range(k)]].to_numpy(float),
                g.n_participant.to_numpy(),
                k,
            )[0]
            survey_metrics.append(
                dict(
                    task=task,
                    view=view,
                    survey=category,
                    responses=len(g),
                    participants=g.n_participant.nunique(),
                    **vals,
                )
            )
    # Cohort/accounting support independent of model outputs; keep all original rows accounted.
    valid = data.pss_combo.notna() & data.context_available
    complete = valid & data[blocks["original60"]].notna().all(axis=1)
    for cohort, mask in [("all_valid", valid), ("complete60", complete)]:
        for retained in [True, False]:
            for category in ["All", "Morning", "Midday", "Endday"]:
                f = data[mask if retained else ~mask]
                if category != "All":
                    f = f[f.n_survey_category.eq(category)]
                composition.append(
                    dict(
                        cohort=cohort,
                        included=retained,
                        survey=category,
                        responses=len(f),
                        participants=f.n_participant.nunique(),
                        scores_available=int(f.pss_combo.notna().sum()),
                        score_mean=float(f.pss_combo.mean()) if f.pss_combo.notna().any() else None,
                        score_median=float(f.pss_combo.median())
                        if f.pss_combo.notna().any()
                        else None,
                        class_support=[int(f.target_three_class.eq(j).sum()) for j in range(3)],
                    )
                )
    contrasts = []
    pairs = [
        ("original60", "context"),
        ("hr60", "original60"),
        ("hr_all", "hr60"),
        ("hr_activity60", "hr60"),
        ("full", "context"),
        ("full", "original60"),
        ("full", "missingness"),
        ("full", "quality"),
        ("full_l1", "full"),
        ("full_elastic", "full"),
        ("full_corr", "full"),
        ("group_select", "full"),
        ("pulseox_full", "full"),
        ("full_balanced", "full"),
        ("original60_balanced", "original60"),
        ("full_no_flags", "full"),
    ]
    for task, k in TASKS.items():
        for a, b in pairs:
            x = pred[pred.task.eq(task) & pred.view.eq(a)]
            y = pred[pred.task.eq(task) & pred.view.eq(b)]
            if x.n_fold.nunique() != 5 or y.n_fold.nunique() != 5:
                continue
            x = x.sort_values("n_ema_id").reset_index(drop=True)
            y = y.sort_values("n_ema_id").reset_index(drop=True)
            contrasts.append(
                dict(
                    task=task,
                    model=a,
                    baseline=b,
                    cohort="all_valid",
                    **paired(x, y, k, cfg["bootstrap_draws"], cfg["seed"]),
                )
            )
        x = pred[pred.task.eq(task) & pred.view.eq("original60")]
        y = pred[pred.task.eq(task) & pred.view.eq("original60_complete")]
        x = x[x.n_ema_id.isin(y.n_ema_id)].sort_values("n_ema_id").reset_index(drop=True)
        y = y.sort_values("n_ema_id").reset_index(drop=True)
        contrasts.append(
            dict(
                task=task,
                model="original60",
                baseline="original60_complete",
                cohort="matched_validation_different_training",
                **paired(x, y, k, cfg["bootstrap_draws"], cfg["seed"]),
            )
        )
        print(f"Completed aggregate uncertainty for {task}", flush=True)
    c = pd.DataFrame(coefficients)
    stability = []
    for (task, view, name), g in c.groupby(["task", "view", "feature"]):
        stability.append(
            dict(
                task=task,
                view=view,
                feature=name,
                available_folds=len(g),
                selected_folds=int(g.selected.sum()),
                mean_absolute_standardized_coefficient=float(
                    np.mean([np.abs(x).max() for x in g.coefficients])
                ),
                interpretation="numeric standardized; categorical/missingness indicators unscaled; not causal",
            )
        )
    public = root / "reports/comprehensive_features" / run.name
    public.mkdir(parents=True, exist_ok=True)
    outputs = dict(
        metrics=pooled,
        fold_metrics=folds,
        per_class=per_class,
        confusion_matrices=matrices,
        calibration=calibration,
        settings=settings,
        selection_stability=stability,
        survey_metrics=survey_metrics,
        cohort_composition=composition,
        paired_differences=contrasts,
        skipped=skipped,
        configuration=cfg,
    )
    for name, value in outputs.items():
        write_json(public / (name + ".json"), value)
    if not (run / "all_predictions.parquet").exists():
        pred.to_parquet(run / "all_predictions.parquet", index=False)
    # Source estimators and membership remain private; public only aggregate coefficients/stability.
    write_json(
        root / "private/comprehensive_features/evaluation_pointer.json",
        dict(public=str(public.relative_to(root))),
    )
    return public


if __name__ == "__main__":
    print(evaluate(Path.cwd()))
