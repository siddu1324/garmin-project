"""Small nested benchmark. Every transform fits on the corresponding training split."""

from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from garmin_comprehensive.models import Transform
from .data import RAW, REFERENCE, PERSONAL, EXTRA, QUALITY


def specifications():
    main = [
        ("context", [], True, False),
        ("physiology", RAW, False, False),
        ("raw", RAW, True, False),
        ("reference", REFERENCE, True, False),
        ("personal", PERSONAL, True, False),
        ("extended", PERSONAL + EXTRA, True, False),
        ("activity", ["steps__supported_rate__60m"], True, False),
        ("missingness", PERSONAL + EXTRA, True, True),
        ("quality", QUALITY, True, False),
        ("personal_quality", PERSONAL + QUALITY, True, False),
    ]
    result = []
    for name, cols, context, missing in main:
        result.append(
            dict(
                name=name,
                columns=cols,
                context=context,
                missing=missing,
                cohort="all",
                family="logistic",
                permutation=None,
            )
        )
    for name in ["context", "raw", "reference", "personal"]:
        base = next(x for x in result if x["name"] == name)
        result.append({**base, "name": "observed_" + name, "cohort": "observed"})
    for name in ["context", "raw"]:
        base = next(x for x in result if x["name"] == name)
        result.append({**base, "name": "supported_" + name, "cohort": "supported"})
    for name in ["context", "raw", "personal"]:
        base = next(x for x in result if x["name"] == name)
        result.append({**base, "name": "hgb_" + name, "family": "hgb"})
    for seed in [7, 19]:
        base = next(x for x in result if x["name"] == "personal")
        result.append({**base, "name": f"permuted_{seed}", "permutation": seed})
    return result


def cohort(f, name):
    ok = f.pss_combo.notna() & f.context_available
    if name == "observed":
        ok &= (
            f[RAW + REFERENCE + ["hr_mean_deviation_bpm", "hr_latest_deviation_bpm"]]
            .notna()
            .all(axis=1)
        )
    if name == "supported":
        ok &= f.supported60
    return f[ok].copy()


class Prepared:
    def __init__(self, spec):
        self.spec = spec

    def fit(self, f):
        self.base = Transform(
            self.spec["columns"], flags=self.spec["missing"], missing_only=self.spec["missing"]
        ).fit(f)
        self.indices = [
            i
            for i, n in enumerate(self.base.names)
            if self.spec["context"] or n in self.spec["columns"]
        ]
        self.names = [self.base.names[i] for i in self.indices]
        if not self.names:
            raise ValueError("No variable training predictors.")
        return self

    def transform(self, f):
        return self.base.transform(f)[:, self.indices]


def parameters(spec, cfg):
    if spec["permutation"] is not None:
        return [{"C": 0.1}]
    if spec["family"] == "logistic":
        return [{"C": c} for c in cfg["C_grid"]]
    return [dict(max_depth=d, max_iter=i) for d in cfg["hgb_depth"] for i in cfg["hgb_iterations"]]


def estimator(spec, p, cfg):
    if spec["family"] == "logistic":
        return LogisticRegression(
            C=p["C"], l1_ratio=0, solver="lbfgs", max_iter=10000, tol=1e-4, random_state=cfg["seed"]
        )
    return HistGradientBoostingClassifier(
        **p,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1,
        early_stopping=False,
        random_state=cfg["seed"],
    )


def fit_checked(model, x, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    return not any(issubclass(w.category, ConvergenceWarning) for w in caught)


@dataclass
class Fitted:
    transform: Prepared
    model: object

    def probability(self, f):
        return self.model.predict_proba(self.transform.transform(f))[:, 1]


def nested(train, task, spec, cfg):
    y = train[f"target_{task}"].to_numpy(int)
    # Split membership is identical across all feature/model comparisons.
    splits = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg["seed"]).split(
            train, y, train.n_participant
        )
    )
    parts = []
    for k, (a, b) in enumerate(splits):
        assert not set(train.iloc[a].n_participant) & set(train.iloc[b].n_participant)
        if any(np.unique(y[ix]).size != 2 for ix in [a, b]):
            raise ValueError("Infeasible inner classes; folds unchanged.")
        t = Prepared(spec).fit(train.iloc[a])
        yt = y[a].copy()
        if spec["permutation"] is not None:
            yt = np.random.default_rng(spec["permutation"] + k).permutation(yt)
        parts.append((t.transform(train.iloc[a]), t.transform(train.iloc[b]), yt, y[b]))
    trials = []
    for p in parameters(spec, cfg):
        scores = []
        converged = []
        for x, xv, yt, yv in parts:
            model = estimator(spec, p, cfg)
            converged.append(fit_checked(model, x, yt))
            scores.append(float(roc_auc_score(yv, model.predict_proba(xv)[:, 1])))
        trials.append(
            dict(
                parameters=p, scores=scores, converged=converged, mean_auroc=float(np.mean(scores))
            )
        )
    valid = [v for v in trials if all(v["converged"])]
    if not valid:
        raise ValueError("No converged candidate.")
    best = max(valid, key=lambda v: v["mean_auroc"])
    trans = Prepared(spec).fit(train)
    yt = y.copy()
    if spec["permutation"] is not None:
        yt = np.random.default_rng(spec["permutation"] + 100).permutation(yt)
    model = estimator(spec, best["parameters"], cfg)
    if not fit_checked(model, trans.transform(train), yt):
        raise ValueError("Nonconverged refit.")
    return Fitted(trans, model), best, trials, splits
