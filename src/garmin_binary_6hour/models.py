"""Frozen single-window allowlists and training-only nested logistic fits."""

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from garmin_comprehensive.models import Transform

TASKS = {"A": 5, "B": 10}
WINDOWS = [60, 120, 360]


def blocks(entries):
    out = {"context": []}
    for m in WINDOWS:
        local = [d for d in entries if d["lookback_minutes"] == m]
        out[f"original{m}"] = [f"hr__{x}__{m}m" for x in ["mean", "sd", "latest", "slope"]]
        for name, modalities in [
            ("hr", ["hr"]),
            ("activity", ["hr", "steps", "floors"]),
            ("full", ["hr", "steps", "floors", "interaction"]),
        ]:
            out[f"{name}{m}"] = [d["name"] for d in local if d["modality"] in modalities]
    out["multiscale"] = [d["name"] for d in entries if d["modality"] != "pulseox"]
    out["quality360"] = [
        f"quality__hr__{x}__360m"
        for x in [
            "accepted_readings",
            "accepted_fraction",
            "occupied_fraction",
            "span_seconds",
            "maximum_gap_seconds",
            "latest_age_seconds",
        ]
    ]
    for m in WINDOWS:
        assert len(out[f"full{m}"]) == 47
        assert all(c.endswith(f"__{m}m") for c in out[f"full{m}"])
    assert len(out["multiscale"]) == 438
    return out


def specifications():
    result = []

    def add(name, block, weighted=False, cohort="all", mode="main"):
        result.append(dict(name=name, block=block, weighted=weighted, cohort=cohort, mode=mode))

    for weighted, suffix in [(False, "u"), (True, "w")]:
        add(f"context_{suffix}", "context", weighted)
        for m in WINDOWS:
            for b in ["original", "hr", "activity", "full"]:
                add(f"{b}{m}_{suffix}", f"{b}{m}", weighted)
            add(f"sparse{m}_{suffix}", f"full{m}", weighted, mode="sparse")
    for m in WINDOWS:
        for mode in ["corr", "availability", "no_flags", "missing_only"]:
            add(f"{mode}{m}_u", f"full{m}", mode=mode)
        add(f"complete_full{m}_u", f"full{m}", cohort="complete")
    add("complete_context_u", "context", cohort="complete")
    add("multiscale_u", "multiscale")
    add("quality360_u", "quality360")
    return result


def targets(scores, threshold):
    return (scores >= threshold).astype(float).where(scores.notna())


def cohort(frame, feature_blocks, name):
    mask = frame.pss_combo.notna() & frame.context_available
    if name == "complete":
        cols = sum([feature_blocks[f"original{m}"] for m in WINDOWS], [])
        mask &= frame[cols].notna().all(axis=1)
    return frame.loc[mask].copy()


def make_transform(train, columns, spec, cfg):
    excluded = []
    if spec["mode"] == "availability":
        excluded = [
            c
            for c in columns
            if train[c].notna().mean() < cfg["minimum_training_observed_fraction"]
        ]
    trans = Transform(
        [c for c in columns if c not in excluded],
        flags=spec["mode"] != "no_flags",
        missing_only=spec["mode"] == "missing_only",
        correlation=cfg["correlation_threshold"] if spec["mode"] == "corr" else None,
    ).fit(train)
    trans.dropped.update({c: "training_availability_below_50pct" for c in excluded})
    return trans


def estimator(parameters, weighted, cfg):
    return LogisticRegression(
        C=parameters["C"],
        l1_ratio=parameters["ratio"],
        solver="saga" if parameters["ratio"] else "lbfgs",
        class_weight="balanced" if weighted else None,
        random_state=cfg["seed"],
        max_iter=cfg["max_iter"],
        tol=cfg["tol"],
    )


def checked_fit(model, x, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    return not any(issubclass(w.category, ConvergenceWarning) for w in caught)


@dataclass
class Fitted:
    transform: object
    model: object

    def probability(self, f):
        assert self.model.classes_.tolist() == [0, 1]
        return self.model.predict_proba(self.transform.transform(f))[:, 1]


def nested_fit(train, task, spec, columns, cfg):
    y = train[f"target_{task}"].to_numpy(int)
    splits = list(
        StratifiedGroupKFold(
            n_splits=cfg["inner_splits"], shuffle=True, random_state=cfg["seed"]
        ).split(train, y, train.n_participant)
    )
    transformed = []
    for a, b in splits:
        assert not set(train.iloc[a].n_participant) & set(train.iloc[b].n_participant)
        if any(len(np.unique(y[ix])) != 2 for ix in [a, b]):
            raise ValueError("Insufficient inner class support; do not redraw folds.")
        trans = make_transform(train.iloc[a], columns, spec, cfg)
        transformed.append((trans.transform(train.iloc[a]), trans.transform(train.iloc[b])))
    ratios = [r for r in cfg["ratios"] if spec["mode"] != "sparse" or r > 0]
    trials = []
    for ratio in ratios:
        for c in cfg["C_grid"]:
            params = dict(C=c, ratio=ratio)
            scores, converged, iterations = [], [], []
            for (a, b), (x, xv) in zip(splits, transformed, strict=True):
                model = estimator(params, spec["weighted"], cfg)
                ok = checked_fit(model, x, y[a])
                scores.append(float(roc_auc_score(y[b], model.predict_proba(xv)[:, 1])))
                converged.append(ok)
                iterations.append(int(model.n_iter_.max()))
            trials.append(
                dict(
                    parameters=params,
                    inner_auroc=scores,
                    converged=converged,
                    iterations=iterations,
                    mean_auroc=float(np.mean(scores)),
                )
            )
    valid = [t for t in trials if all(t["converged"])]
    if not valid:
        raise ValueError("No converged candidate.")
    best = max(valid, key=lambda x: x["mean_auroc"])
    trans = make_transform(train, columns, spec, cfg)
    model = estimator(best["parameters"], spec["weighted"], cfg)
    if not checked_fit(model, trans.transform(train), y):
        raise ValueError("Selected outer refit did not converge.")
    return Fitted(trans, model), best, trials, splits
