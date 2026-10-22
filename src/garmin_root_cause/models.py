"""Small grouped logistic benchmark and independently tested ordinal likelihood."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from garmin_temporal_personalization.models import Prepared
from .data import HR, STEP


def specs():
    return [dict(name=name, columns=cols, context=context, missing=False)
            for name, cols, context in [
                ("context", [], True), ("hr", HR, False),
                ("context_hr", HR, True), ("context_hr_steps", HR + [STEP], True),
            ]]


def ordinal_parameters(w, d):
    beta, raw = w[:d], w[d:]
    theta = np.r_[raw[0], raw[0] + np.cumsum(np.exp(raw[1:]))]
    return beta, theta


def ordinal_objective(w, x, y, C=1.0):
    """Sum negative log likelihood + beta L2/(2C); y integers 0..4."""
    d = x.shape[1]
    beta, theta = ordinal_parameters(w, d)
    eta = x @ beta
    cdf = expit(theta[None, :] - eta[:, None])
    cdf = np.column_stack([np.zeros(len(x)), cdf, np.ones(len(x))])
    n = np.arange(len(x))
    p = np.maximum(cdf[n, y + 1] - cdf[n, y], 1e-14)
    density = cdf * (1 - cdf)
    hi, lo = density[n, y + 1], density[n, y]
    gb = x.T @ ((hi - lo) / p) + beta / C
    gt = np.zeros(4)
    for k in range(4):
        gt[k] = np.sum(np.where(y == k, -hi / p, 0) + np.where(y == k + 1, lo / p, 0))
    gr = np.r_[gt.sum(), np.exp(w[d + 1:]) * np.cumsum(gt[::-1])[::-1][1:]]
    return -np.log(p).sum() + beta @ beta / (2 * C), np.r_[gb, gr]


class OrdinalLogit:
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, x, y):
        y = np.asarray(y, int) - 1
        counts = np.bincount(y, minlength=5) + 0.5
        theta = logit(np.cumsum(counts)[:-1] / counts.sum())
        w = np.r_[np.zeros(x.shape[1]), theta[0], np.log(np.diff(theta))]
        result = minimize(ordinal_objective, w, args=(x, y, self.C), jac=True,
                          method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-6})
        self.converged = bool(result.success)
        self.gradient_max = float(np.max(np.abs(result.jac)))
        self.beta, self.theta = ordinal_parameters(result.x, x.shape[1])
        self.objective = float(result.fun)
        if not self.converged:
            raise ValueError("Ordinal optimization did not converge")
        return self

    def predict_proba(self, x):
        f = expit(self.theta[None, :] - (x @ self.beta)[:, None])
        return np.diff(np.column_stack([np.zeros(len(x)), f, np.ones(len(x))]), axis=1)


class PredictorTransform(Prepared):
    def fit(self, f):
        super().fit(f)
        # Remove one survey indicator: thresholds/intercept supply the location term.
        survey = [i for i, n in enumerate(self.names) if n.startswith("n_survey_category_")]
        if survey:
            j = survey[0]
            self.names.pop(j)
            self.indices.pop(j)
        return self


@dataclass
class Fitted:
    transform: PredictorTransform
    model: object
    kind: str

    def predict(self, f):
        x = self.transform.transform(f)
        if self.kind == "continuous":
            return self.model.predict(x)
        p = self.model.predict_proba(x)
        return p[:, 1] if self.kind == "binary" else p


def logistic(C):
    return LogisticRegression(C=C, l1_ratio=0, max_iter=10000, tol=1e-7, solver="lbfgs")


def checked_fit(model, x, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ValueError("Model nonconvergence")
    return model


def fit_model(train, column, spec, kind, cfg):
    y = train[column].to_numpy(float)
    trials, membership = [], []
    if kind == "binary":
        if len(np.unique(y)) != 2:
            raise ValueError("Single-class training cohort")
        splits = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg["seed"])
                      .split(train, y, train.n_participant))
        parts = []
        for k, (a, b) in enumerate(splits):
            assert not set(train.iloc[a].n_participant) & set(train.iloc[b].n_participant)
            if any(np.unique(y[ix]).size != 2 for ix in [a, b]):
                raise ValueError("Infeasible inner classes; do not redraw")
            t = PredictorTransform(spec).fit(train.iloc[a])
            parts.append((t.transform(train.iloc[a]), t.transform(train.iloc[b]), y[a], y[b]))
            for role, ix in [("train", a), ("validation", b)]:
                membership.append(train.iloc[ix][["n_ema_id", "n_participant"]].assign(inner=k, role=role))
        for C in cfg["C_grid"]:
            values = []
            for x, xv, yt, yv in parts:
                model = checked_fit(logistic(C), x, yt)
                values.append(float(roc_auc_score(yv, model.predict_proba(xv)[:, 1])))
            trials.append(dict(C=C, inner_auroc=values, mean_auroc=float(np.mean(values))))
        best = max(trials, key=lambda v: v["mean_auroc"])["C"]
        model = logistic(best)
    elif kind == "ordinal":
        model = OrdinalLogit(cfg["ordinal_C"])
    else:
        model = Ridge(alpha=cfg["ridge_alpha"])
    transform = PredictorTransform(spec).fit(train)
    model = checked_fit(model, transform.transform(train), y)
    return Fitted(transform, model, kind), trials, membership
