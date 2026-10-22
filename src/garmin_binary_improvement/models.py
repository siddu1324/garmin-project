"""Nested grouped logistic selection; augmentation never sees held-out rows."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from garmin_binary_6hour.models import checked_fit, estimator
from garmin_comprehensive.models import Transform
from .sampling import prior_correct, resample


class Prepared:
    def __init__(self, columns, spec, cfg):
        self.columns, self.spec, self.cfg = list(columns), spec, cfg

    def fit(self, f):
        columns = self.columns
        if self.spec["compact"]:
            columns = [
                c
                for c in columns
                if f[c].notna().mean() >= self.cfg["compact_min_availability"]
                and f[c].nunique() > 1
            ]
            chosen = []
            for c in columns:
                if not any(
                    abs(f[c].fillna(f[c].median()).corr(f[d].fillna(f[d].median())))
                    > self.cfg["compact_max_correlation"]
                    for d in chosen
                ):
                    chosen.append(c)
                if len(chosen) == self.cfg["compact_max_features"]:
                    break
            columns = chosen
        self.base = Transform(
            columns, flags=self.spec["flags"], missing_only=self.spec["mode"] == "missing_only"
        ).fit(f)
        self.indices = [
            i for i, n in enumerate(self.base.names) if self.spec["context"] or n in columns
        ]
        self.names = [self.base.names[i] for i in self.indices]
        self.numeric_indices = [i for i, n in enumerate(self.names) if n in columns]
        self.physiological = [self.names[i] for i in self.numeric_indices]
        if not self.names:
            raise ValueError("No nonconstant permitted training features.")
        return self

    def transform(self, f):
        return self.base.transform(f)[:, self.indices]


@dataclass
class Fitted:
    transform: object
    model: object
    threshold: float
    prevalence: float
    effective_prevalence: float

    def probability(self, f):
        return self.model.predict_proba(self.transform.transform(f))[:, 1]

    def corrected(self, p):
        return prior_correct(p, self.prevalence, self.effective_prevalence)


def threshold(y, p):
    fp, tp, th = roc_curve(y, p)
    good = np.isfinite(th) & (th >= 0) & (th <= 1)
    values = (tp + 1 - fp)[good] / 2
    candidates = th[good][np.isclose(values, values.max(), atol=1e-12, rtol=0)]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def training_labels(train, task, spec, salt=0):
    y = train[f"target_{task}"].to_numpy(int).copy()
    if spec["permutation"] is not None:
        y = np.random.default_rng(spec["permutation"] + salt).permutation(y)
    return y


def prepare(train, columns, spec, cfg, y, salt):
    t = Prepared(columns, spec, cfg).fit(train)
    x = t.transform(train)
    xr, yr, audit, lineage = resample(
        x,
        y,
        train.n_participant.to_numpy(),
        train.n_survey_category.to_numpy(),
        train[t.physiological].isna().to_numpy(),
        spec["sampler"],
        t.numeric_indices,
        cfg["seed"] + salt,
        cfg["smote_neighbors"],
    )
    return t, xr, yr, audit, lineage


def nested_fit(train, task, spec, columns, cfg, reused=None):
    y = training_labels(train, task, spec)
    splits = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg["seed"]).split(
            train, y, train.n_participant
        )
    )
    prepared, audits, donors = [], [], []
    for i, (a, b) in enumerate(splits):
        assert not set(train.iloc[a].n_participant) & set(train.iloc[b].n_participant)
        if any(len(np.unique(y[ix])) != 2 for ix in [a, b]):
            raise ValueError("Infeasible grouped class support; folds not redrawn.")
        t, x, yt, audit, lineage = prepare(train.iloc[a], columns, spec, cfg, y[a], i)
        audits.append(dict(inner_fold=i, **audit))
        lineage["inner_fold"] = i
        if len(lineage):
            lineage["base"] = a[lineage.base.to_numpy(int)]
            lineage["neighbor"] = a[lineage.neighbor.to_numpy(int)]
        donors.append(lineage)
        prepared.append((x, yt, t.transform(train.iloc[b])))
    if reused:
        params = [reused["best"]["parameters"]]
    elif spec["permutation"] is not None:
        params = [dict(C=0.01, ratio=0.0)]
    else:
        params = [
            dict(C=c, ratio=r)
            for r in cfg["ratios"]
            for c in cfg["C_grid"]
            if spec["mode"] != "sparse" or r > 0
        ]
    trials, probabilities = [], []
    for param in params:
        oof = np.full(len(train), np.nan)
        scores, convergence = [], []
        for (a, b), (x, yt, xv) in zip(splits, prepared, strict=True):
            model = estimator(param, spec["sampler"] == "weighted", cfg)
            ok = checked_fit(model, x, yt)
            oof[b] = model.predict_proba(xv)[:, 1]
            scores.append(float(roc_auc_score(y[b], oof[b])))
            convergence.append(ok)
        trials.append(
            dict(
                parameters=param,
                inner_auroc=scores,
                converged=convergence,
                mean_auroc=float(np.mean(scores)),
            )
        )
        probabilities.append(oof)
    eligible = [i for i, tr in enumerate(trials) if all(tr["converged"])]
    if not eligible:
        raise ValueError("No converged candidate; no budget expansion.")
    best_i = max(eligible, key=lambda i: trials[i]["mean_auroc"])
    best = trials[best_i]
    oof = probabilities[best_i]
    cutoff = threshold(y, oof)
    t, x, yt, audit, lineage = prepare(train, columns, spec, cfg, y, 100)
    audits.append(dict(inner_fold="outer_refit", **audit))
    lineage["inner_fold"] = "outer_refit"
    donors.append(lineage)
    model = estimator(best["parameters"], spec["sampler"] == "weighted", cfg)
    if not checked_fit(model, x, yt):
        raise ValueError("Selected outer fit did not converge.")
    fitted = Fitted(
        t,
        model,
        cutoff,
        float(y.mean()),
        0.5 if spec["sampler"] == "weighted" else float(yt.mean()),
    )
    return fitted, best, trials, splits, audits, donors, oof, y
