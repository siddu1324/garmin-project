"""Matched OOF comparisons, fixed-prediction participant bootstrap, calibration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from garmin_binary_6hour.metrics import score, bootstrap, interval_fields
from garmin_comprehensive_regression.evaluate import finite
from garmin_preprocessing.common import verify_bundle, write_json
from .preservation import completed


def ordinal_metrics(z):
    p = z[[f"p{k}" for k in range(1, 6)]].to_numpy(float)
    y = z.target.to_numpy(int)
    cumulative = p.cumsum(axis=1)[:, :-1]
    observed = y[:, None] <= np.arange(1, 5)
    rps = ((cumulative - observed) ** 2).mean(axis=1)
    loss = -np.log(np.maximum(p[np.arange(len(y)), y - 1], 1e-15))
    error = np.abs(p @ np.arange(1, 6) - y)
    return dict(
        ranked_probability_score=float(rps.mean()),
        log_loss=float(loss.mean()),
        expected_category_mae=float(error.mean()),
        expected_category_sd=float((p @ np.arange(1, 6)).std()),
    ), rps


def continuous_metrics(z):
    e = z.prediction - z.target
    return dict(
        mae=float(e.abs().mean()),
        rmse=float(np.sqrt((e**2).mean())),
        spearman=float(spearmanr(z.target, z.prediction).statistic)
        if z.prediction.nunique() > 1
        else None,
        prediction_sd=float(z.prediction.std(ddof=0)),
    ), e.abs().to_numpy()


def error_bootstrap(z, error, cfg):
    _, inv = np.unique(z.n_participant, return_inverse=True)
    counts, sums = np.bincount(inv), np.bincount(inv, weights=error)
    rng = np.random.default_rng(cfg["seed"])
    w = rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), cfg["bootstrap_draws"])
    return w @ sums / (w @ counts)


def evaluate(root):
    root = Path(root)
    ptr = json.loads((root / "private/root_cause_analysis/current.json").read_text())
    run = root / ptr["run"]
    out = root / "reports/root_cause_analysis" / run.name
    if completed(root):
        return out
    out.mkdir(exist_ok=True)
    cfg = json.loads((run / "identity.json").read_text())["config"]
    predictions = []
    supports = []
    failures = []
    training = []
    settings = []
    for path in sorted((run / "jobs").glob("*/summary.json")):
        verify_bundle(path.parent)
        info = json.loads(path.read_text())
        target, kind, model = info["target"], info["kind"], info["spec"]["name"]
        for role in ["train", "validation"]:
            supports.append(
                dict(
                    task=target, kind=kind, model=model, fold=info["fold"], role=role, **info[role]
                )
            )
        if info["status"] != "complete":
            failures.append(info)
            continue
        z = pd.read_parquet(path.parent / "validation.parquet").assign(
            task=target, kind=kind, model=model
        )
        predictions.append(z)
        tr = pd.read_parquet(path.parent / "train.parquet")
        metric = (
            score(tr.target, tr.probability)
            if kind == "binary"
            else ordinal_metrics(tr)[0]
            if kind == "ordinal"
            else continuous_metrics(tr)[0]
        )
        training.append(dict(task=target, kind=kind, model=model, fold=info["fold"], **metric))
        settings.append(
            dict(
                task=target,
                kind=kind,
                model=model,
                fold=info["fold"],
                trials=json.loads((path.parent / "trials.json").read_text()),
                gradient=info.get("optimizer_gradient_max"),
            )
        )
    predictions = pd.concat(predictions, ignore_index=True)
    for info in failures:
        predictions = predictions[
            ~(predictions.task.eq(info["target"]) & predictions.model.eq(info["spec"]["name"]))
        ]
    constants = []
    for (task, kind), g in predictions.groupby(["task", "kind"]):
        z = g[g.model.eq(sorted(g.model.unique())[0])].copy()
        z["model"] = "constant"
        if kind == "binary":
            z["probability"] = z.dummy_mean
        elif kind == "ordinal":
            for k in range(1, 6):
                z[f"p{k}"] = z[f"dummy{k}"]
        else:
            z["prediction"] = z.dummy_mean
            med = z.copy()
            med["model"] = "median"
            med["prediction"] = med.dummy_median
            constants.append(med)
        constants.append(z)
    predictions = pd.concat([predictions, *constants], ignore_index=True)
    predictions.to_parquet(run / "predictions.parquet", index=False)
    metrics = []
    folds = []
    calibration = []
    by_survey = []
    paired = []
    draws = {}
    indexed = {}
    for (task, kind, model), g in predictions.groupby(["task", "kind", "model"]):
        g = g.sort_values("n_ema_id").reset_index(drop=True)
        assert g.n_ema_id.is_unique and g.n_fold.nunique() == 5
        key = (task, model)
        indexed[key] = g
        row = dict(
            task=task,
            kind=kind,
            model=model,
            responses=len(g),
            participants=g.n_participant.nunique(),
        )
        if kind == "binary":
            d = bootstrap(g, cfg)
            draws[key] = d
            row.update(
                score(g.target, g.probability),
                **interval_fields(d),
                prevalence=float(g.target.mean()),
            )
            numerator = denominator = 0
            for fold, z in g.groupby("n_fold"):
                s = score(z.target, z.probability)
                folds.append(
                    dict(
                        task=task,
                        kind=kind,
                        model=model,
                        fold=int(fold),
                        responses=len(z),
                        participants=z.n_participant.nunique(),
                        prevalence=float(z.target.mean()),
                        **s,
                    )
                )
                n = z.target.sum() * (len(z) - z.target.sum())
                if s["auroc"] is not None:
                    numerator += n * s["auroc"]
                    denominator += n
            row["within_fold_pair_weighted_auroc"] = (
                numerator / denominator if denominator else None
            )
            row["mean_probability"] = float(g.probability.mean())
            for bin_, z in g.groupby(np.minimum((g.probability * 10).astype(int), 9)):
                calibration.append(
                    dict(
                        task=task,
                        model=model,
                        bin=int(bin_),
                        responses=len(z),
                        mean_probability=float(z.probability.mean()),
                        positive_fraction=float(z.target.mean()),
                    )
                )
            for cat, z in g.groupby("n_survey_category"):
                by_survey.append(
                    dict(
                        task=task,
                        model=model,
                        survey=cat,
                        responses=len(z),
                        participants=z.n_participant.nunique(),
                        **score(z.target, z.probability),
                    )
                )
        else:
            measure, error = ordinal_metrics(g) if kind == "ordinal" else continuous_metrics(g)
            d = error_bootstrap(g, error, cfg)
            draws[key] = d
            row.update(
                measure,
                primary_error_lower=float(np.quantile(d, 0.025)),
                primary_error_upper=float(np.quantile(d, 0.975)),
            )
            for fold, z in g.groupby("n_fold"):
                s, _ = ordinal_metrics(z) if kind == "ordinal" else continuous_metrics(z)
                folds.append(
                    dict(
                        task=task,
                        kind=kind,
                        model=model,
                        fold=int(fold),
                        responses=len(z),
                        participants=z.n_participant.nunique(),
                        **s,
                    )
                )
        metrics.append(row)
    for task in predictions.task.unique():
        for left, right in [
            ("context", "constant"),
            ("hr", "constant"),
            ("context_hr", "context"),
            ("context_hr_steps", "context_hr"),
            ("context_hr_steps", "context"),
        ]:
            if (task, left) not in draws or (task, right) not in draws:
                continue
            a, b = indexed[task, left], indexed[task, right]
            assert a.n_ema_id.tolist() == b.n_ema_id.tolist()
            np.testing.assert_array_equal(a.target, b.target)
            d = draws[task, left] - draws[task, right]
            record = dict(
                task=task,
                left=left,
                right=right,
                responses=len(a),
                participants=a.n_participant.nunique(),
            )
            if a.kind.iloc[0] == "binary":
                record.update(interval_fields(d, "difference_"))
                for name in ["auroc", "ap", "balanced_accuracy"]:
                    record["difference_" + name] = (
                        score(a.target, a.probability)[name] - score(b.target, b.probability)[name]
                    )
            else:
                lo, hi = np.quantile(d, [0.025, 0.975])
                record.update(difference_lower=lo, difference_upper=hi)
                fun = ordinal_metrics if a.kind.iloc[0] == "ordinal" else continuous_metrics
                record["difference_error"] = float(fun(a)[1].mean() - fun(b)[1].mean())
            paired.append(record)
    contrasts = []
    for other in [
        "target_composite5",
        "target_composite10",
        "target_cope_difficulty",
        "target_control_difficulty",
    ]:
        a = draws["target_nervous", "context_hr"] - draws["target_nervous", "context"]
        b = draws[other, "context_hr"] - draws[other, "context"]
        contrasts.append(
            dict(
                comparison="nervous HR increment minus " + other + " HR increment",
                **interval_fields(a - b, "difference_"),
            )
        )
    for name, obj in dict(
        metrics=metrics,
        folds=folds,
        supports=supports,
        training=training,
        settings=settings,
        failures=failures,
        calibration=calibration,
        survey_metrics=by_survey,
        paired=paired,
        target_increment_comparison=contrasts,
    ).items():
        write_json(out / f"{name}.json", finite(obj))
    return out


if __name__ == "__main__":
    print(evaluate(Path.cwd()))
