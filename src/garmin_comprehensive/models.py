"""Training-only transforms and bounded nested logistic selection."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from garmin_inventory.privacy import AuditError
from garmin_pss_classification.metrics import macro_f1

CONTEXT = ["local_time_sin", "local_time_cos"]


def feature_blocks(entries):
    names = [d["name"] for d in entries]
    hr = [d["name"] for d in entries if d["modality"] == "hr"]
    activity = [d["name"] for d in entries if d["modality"] in ["steps", "floors", "interaction"]]
    full = [d["name"] for d in entries if d["modality"] != "pulseox"]
    return dict(
        context=[],
        original60=[f"hr__{m}__60m" for m in ["mean", "sd", "latest", "slope"]]
        + ["steps__added_supported__60m"],
        hr60=[c for c in hr if c.endswith("__60m")],
        hr_all=hr,
        hr_activity60=[c for c in hr + activity if c.endswith("__60m")],
        hr_activity_all=hr + activity,
        full=full,
        pulseox_full=names,
        **{
            f"hr_{m}m": [c for c in hr if c.endswith(f"__{m}m")]
            for m in [5, 15, 30, 120, 360, 720, 1440]
        },
    )


class Transform:
    def __init__(
        self, columns, handling="imputed", flags=True, missing_only=False, correlation=None
    ):
        self.columns = list(columns)
        self.handling = handling
        self.flags = flags
        self.missing_only = missing_only
        self.correlation = correlation
        if any(c.startswith(("n_", "target_", "PSS_")) or c == "pss_combo" for c in self.columns):
            raise AuditError("Forbidden identifier/outcome in predictors.")

    def fit(self, f):
        numeric = CONTEXT + ([] if self.missing_only else self.columns)
        x = f[numeric].to_numpy(float)
        if np.isinf(x).any():
            raise AuditError("Infinite predictor.")
        self.kept = [c for c in numeric if f[c].notna().any() and f[c].nunique() > 1]
        self.dropped = {
            c: "all_missing" if not f[c].notna().any() else "constant"
            for c in numeric
            if c not in self.kept
        }
        self.medians = f[self.kept].median()
        filled = f[self.kept].fillna(self.medians).to_numpy(float)
        if self.handling == "complete" and f[numeric].isna().any().any():
            raise AuditError("Missing complete-case value.")
        if self.correlation and len(self.kept) > 1:
            corr = np.abs(np.corrcoef(filled, rowvar=False))
            selected = []
            for j, c in enumerate(self.kept):
                if c in CONTEXT or not any(
                    corr[j, i] > self.correlation for i in selected if self.kept[i] not in CONTEXT
                ):
                    selected.append(j)
                else:
                    self.dropped[c] = "training_correlation"
            self.kept = [self.kept[j] for j in selected]
        self.medians = self.medians[self.kept]
        self.scaler = (
            StandardScaler().fit(f[self.kept].fillna(self.medians).to_numpy(float))
            if self.kept
            else None
        )
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
            f[["n_survey_category"]]
        )
        self.flag_columns = (
            [c for c in self.columns if f[c].isna().nunique() > 1] if self.flags else []
        )
        self.names = (
            self.kept
            + list(self.encoder.get_feature_names_out())
            + ["missing__" + c for c in self.flag_columns]
        )
        return self

    def transform(self, f):
        if self.handling == "complete" and f[self.kept].isna().any().any():
            raise AuditError("Missing complete-case validation value.")
        blocks = []
        if self.kept:
            blocks.append(self.scaler.transform(f[self.kept].fillna(self.medians).to_numpy(float)))
        blocks.append(self.encoder.transform(f[["n_survey_category"]]))
        if self.flag_columns:
            blocks.append(f[self.flag_columns].isna().to_numpy(float))
        x = np.column_stack(blocks)
        if not np.isfinite(x).all():
            raise AuditError("Nonfinite transformed input.")
        return x


def split_inner(f, target, k, cfg):
    pairs = list(
        StratifiedGroupKFold(
            n_splits=cfg["inner_splits"], shuffle=True, random_state=cfg["seed"]
        ).split(f, f[target], f.n_participant)
    )
    for a, b in pairs:
        assert not set(f.iloc[a].n_participant) & set(f.iloc[b].n_participant)
        if any(len(np.unique(f.iloc[ix][target])) != k for ix in [a, b]):
            raise AuditError("Infeasible inner class support; folds unchanged.")
    return pairs


def fit(x, y, parameters, cfg, balanced=False):
    ratio = parameters.get("l1_ratio", 0.0)
    m = LogisticRegression(
        C=parameters["C"],
        l1_ratio=ratio,
        solver="saga" if ratio else "lbfgs",
        max_iter=cfg["max_iter"],
        tol=cfg["tol"],
        random_state=cfg["seed"],
        class_weight="balanced" if balanced else None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(x, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return m, converged


def specs(blocks):
    out = [
        dict(name=n, block=n, method="l2", handling="imputed", flags=True)
        for n in blocks
        if n != "hr_activity_all"
    ]
    for name, method in [
        ("full_corr", "corr"),
        ("full_l1", "l1"),
        ("full_elastic", "elastic"),
        ("group_select", "groups"),
    ]:
        out.append(dict(name=name, block="full", method=method, handling="imputed", flags=True))
    out += [
        dict(name="full_no_flags", block="full", method="l2", handling="imputed", flags=False),
        dict(
            name="missingness",
            block="full",
            method="l2",
            handling="imputed",
            flags=True,
            missing_only=True,
        ),
        dict(name="quality", block="quality", method="l2", handling="imputed", flags=True),
    ]
    out += [
        dict(
            name=n + "_balanced",
            block=n,
            method="l2",
            handling="imputed",
            flags=True,
            balanced=True,
        )
        for n in ["original60", "full"]
    ]
    out += [
        dict(name=n + "_complete", block=n, method="l2", handling="complete", flags=False)
        for n in ["context", "original60"]
    ]
    # get_data adds the quality block; its explicitly named diagnostic appears below.
    # Identical specifications must schedule one job, not two cached views of one fit.
    return list({spec["name"]: spec for spec in out}.values())


def nested_fit(train, valid, target, k, spec, blocks, cfg):
    """Validation targets are not accepted by the tuning functions."""
    splits = split_inner(train, target, k, cfg)
    method = spec["method"]
    choices = ["hr_all", "hr_activity_all", "full"] if method == "groups" else [spec["block"]]
    ratio = {"l1": 1.0, "elastic": 0.5}.get(method, 0.0)
    grid = cfg["selection_C_grid"] if ratio else cfg["C_grid"]
    results = []
    cache = {}
    for block in choices:
        columns = blocks[block]
        kw = dict(
            handling=spec["handling"],
            flags=spec["flags"],
            missing_only=spec.get("missing_only", False),
            correlation=cfg["correlation_threshold"] if method == "corr" else None,
        )
        prepared = []
        for a, b in splits:
            tr, va = train.iloc[a], train.iloc[b]
            trans = Transform(columns, **kw).fit(tr)
            prepared.append(
                (
                    trans.transform(tr),
                    tr[target].to_numpy(int),
                    trans.transform(va),
                    va[target].to_numpy(int),
                )
            )
        cache[block] = kw
        for C in grid:
            scores = []
            iterations = []
            ok = True
            for x, y, xv, yv in prepared:
                model, conv = fit(x, y, dict(C=C, l1_ratio=ratio), cfg, spec.get("balanced", False))
                iterations.append(int(model.n_iter_.max()))
                if not conv:
                    ok = False
                    break
                scores.append(macro_f1(yv, model.predict(xv), k))
            results.append(
                dict(
                    block=block,
                    C=C,
                    l1_ratio=ratio,
                    converged=ok,
                    inner_macro_f1=float(np.mean(scores)) if ok else None,
                    iterations=iterations,
                )
            )
    candidates = [r for r in results if r["converged"]]
    if not candidates:
        raise AuditError("All fixed candidates failed convergence.")
    best = max(candidates, key=lambda r: r["inner_macro_f1"])
    trans = Transform(blocks[best["block"]], **cache[best["block"]]).fit(train)
    model, conv = fit(
        trans.transform(train), train[target].to_numpy(int), best, cfg, spec.get("balanced", False)
    )
    if not conv:
        raise AuditError("Selected outer fit did not converge.")
    assert np.array_equal(model.classes_, np.arange(k))
    probability = model.predict_proba(trans.transform(valid))
    training_probability = model.predict_proba(trans.transform(train))
    return trans, model, probability, training_probability, best, results, splits
