"""Retrospective associations, clustered uncertainty and random-intercept REML."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata

from garmin_comprehensive_regression.evaluate import finite
from garmin_preprocessing.common import write_json
from .data import HR, STEP, ITEMS, TASKS
from .preservation import completed


def random_intercept_reml(x, y, groups):
    """Profile REML with V=sigma²(I+r ZZ'); no subject effects used predictively."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    _, g = np.unique(groups, return_inverse=True)
    counts = np.bincount(g)
    n, p = x.shape
    if np.linalg.matrix_rank(x) != p:
        raise ValueError("Rank-deficient descriptive design")
    sums_x = np.vstack([np.bincount(g, weights=x[:, j]) for j in range(p)]).T
    sums_y = np.bincount(g, weights=y)

    def profile(r, full=False):
        factor = r / (1 + r * counts)
        vx = x - factor[g, None] * sums_x[g]
        vy = y - factor[g] * sums_y[g]
        information = x.T @ vx
        beta = np.linalg.solve(information, x.T @ vy)
        residual = y - x @ beta
        vr = residual - factor[g] * np.bincount(g, weights=residual)[g]
        sigma2 = float(residual @ vr / (n - p))
        objective = float(
            np.log1p(r * counts).sum()
            + np.linalg.slogdet(information)[1]
            + (n - p) * np.log(sigma2)
        )
        return (beta, sigma2, sigma2 * np.linalg.inv(information), objective) if full else objective

    opt = minimize_scalar(lambda logr: profile(np.exp(logr)), bounds=(-20, 10), method="bounded")
    r = float(np.exp(opt.x)) if opt.fun < profile(0) else 0.0
    beta, sigma2, covariance, objective = profile(r, True)
    return dict(
        beta=beta,
        covariance=covariance,
        residual_variance=sigma2,
        participant_variance=r * sigma2,
        icc=r / (1 + r),
        converged=bool(opt.success),
        boundary_solution=bool(r == 0 or opt.x > 9.99),
        objective=objective,
    )


def association(z, outcome):
    h, y = rankdata(z[HR[0]]), rankdata(z[outcome])
    _, ids = np.unique(z.n_participant, return_inverse=True)
    n = np.bincount(ids)
    hm, ym = np.bincount(ids, weights=h) / n, np.bincount(ids, weights=y) / n
    ctx = pd.get_dummies(z[["n_survey_category"]], drop_first=True, dtype=float).to_numpy()
    cols = [
        "local_time_sin",
        "local_time_cos",
        STEP,
        "quality__hr__occupied_fraction__60m",
        "quality__hr__latest_age_seconds__60m",
        "quality__hr__maximum_gap_seconds__60m",
    ]
    nuisance = np.column_stack([np.ones(len(z)), ctx, z[cols].to_numpy(float)])
    rh = h - nuisance @ np.linalg.lstsq(nuisance, h, rcond=None)[0]
    ry = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-10 and np.std(b) > 1e-10 else np.nan

    centered_x, centered_y = h - hm[ids], y - ym[ids]
    centered_nuisance = nuisance.copy()
    for j in range(nuisance.shape[1]):
        centered_nuisance[:, j] -= (np.bincount(ids, weights=nuisance[:, j]) / n)[ids]
    ax = (
        centered_x
        - centered_nuisance @ np.linalg.lstsq(centered_nuisance, centered_x, rcond=None)[0]
    )
    ay = (
        centered_y
        - centered_nuisance @ np.linalg.lstsq(centered_nuisance, centered_y, rcond=None)[0]
    )
    return np.array(
        [corr(h, y), corr(centered_x, centered_y), corr(hm, ym), corr(rh, ry), corr(ax, ay)]
    )


def describe(root):
    root = Path(root)
    ptr = json.loads((root / "private/root_cause_analysis/current.json").read_text())
    data, run = root / ptr["data"], root / ptr["run"]
    out = root / "reports/root_cause_analysis" / run.name
    if completed(root):
        return out
    out.mkdir(exist_ok=True)
    f = pd.read_parquet(data / "development.parquet")
    cfg = json.loads((run / "identity.json").read_text())["config"]
    distributions, variation, mix, survey = [], [], [], []
    measurements = ["pss_combo", *["ordinal_" + t for t in ITEMS], "item_cope", "item_control"]
    for cohort, z in [("all_development", f), ("matched_observed", f[f.matched])]:
        for col in measurements:
            g = z.dropna(subset=[col])
            mean = g.groupby("n_participant")[col].transform("mean")
            total = float(((g[col] - g[col].mean()) ** 2).sum())
            within = float(((g[col] - mean) ** 2).sum())
            variation.append(
                dict(
                    cohort=cohort,
                    measure=col,
                    responses=len(g),
                    participants=g.n_participant.nunique(),
                    variance=float(g[col].var()),
                    within_fraction=within / total,
                    between_fraction=1 - within / total,
                    within_variance_contribution=within / (len(g) - 1),
                    between_variance_contribution=(total - within) / (len(g) - 1),
                )
            )
            for value, count in g[col].value_counts().sort_index().items():
                distributions.append(
                    dict(cohort=cohort, measure=col, value=float(value), responses=int(count))
                )
            for category, part in g.groupby("n_survey_category"):
                survey.append(
                    dict(
                        cohort=cohort,
                        measure=col,
                        survey=category,
                        responses=len(part),
                        participants=part.n_participant.nunique(),
                        mean=float(part[col].mean()),
                        sd=float(part[col].std()),
                    )
                )
        for task in TASKS:
            g = z.dropna(subset=["target_" + task])
            counts = g.groupby("n_participant")["target_" + task].agg(["mean", "size", "nunique"])
            counts.to_parquet(run / f"private_class_mix_{cohort}_{task}.parquet")
            mix.append(
                dict(
                    cohort=cohort,
                    task=task,
                    responses=len(g),
                    positive=int(g["target_" + task].sum()),
                    participants=len(counts),
                    both_classes=int(counts["nunique"].gt(1).sum()),
                    positive_fraction_q25=float(counts["mean"].quantile(0.25)),
                    positive_fraction_median=float(counts["mean"].median()),
                    positive_fraction_q75=float(counts["mean"].quantile(0.75)),
                    responses_per_person_median=float(counts["size"].median()),
                )
            )
    composition = []
    for included, z in f.groupby("matched"):
        for cat, g in z.groupby("n_survey_category"):
            composition.append(
                dict(
                    included=bool(included),
                    survey=cat,
                    responses=len(g),
                    participants=g.n_participant.nunique(),
                    valid_scores=int(g.pss_combo.notna().sum()),
                    mean_score=float(g.pss_combo.mean()),
                    high_fraction=float(g.pss_combo.ge(10).sum() / g.pss_combo.notna().sum()),
                )
            )
    matched = f[f.matched].copy()
    associations = []
    groups = [g for _, g in matched.groupby("n_participant")]
    # Resample whole persons and use independent replicate IDs for duplicated clusters.
    for outcome in ["pss_combo", *["ordinal_" + t for t in ITEMS]]:
        point = association(matched, outcome)
        rng = np.random.default_rng(cfg["seed"])
        draws = []
        for _ in range(cfg["descriptive_bootstrap_draws"]):
            ix = rng.integers(len(groups), size=len(groups))
            z = pd.concat(
                [groups[j].assign(n_participant=f"replicate{k}") for k, j in enumerate(ix)],
                ignore_index=True,
            )
            draws.append(association(z, outcome))
        low, high = np.nanquantile(draws, [0.025, 0.975], axis=0)
        for j, mode in enumerate(
            [
                "pooled_rank",
                "within_person_rank",
                "between_person_mean_rank",
                "context_activity_support_adjusted_rank",
                "within_person_adjusted_rank",
            ]
        ):
            associations.append(
                dict(
                    outcome=outcome,
                    association=mode,
                    correlation=point[j],
                    lower=low[j],
                    upper=high[j],
                    responses=len(matched),
                    participants=len(groups),
                )
            )
    # Continuous composite random intercept with a within/between (Mundlak) HR decomposition.
    means = matched.groupby("n_participant")[HR[0]].transform("mean")
    x = pd.DataFrame(
        dict(
            intercept=np.ones(len(matched)),
            hr_within_per10bpm=(matched[HR[0]] - means) / 10,
            hr_between_per10bpm=means / 10,
        ),
        index=matched.index,
    )
    for col in [
        "local_time_sin",
        "local_time_cos",
        STEP,
        "quality__hr__occupied_fraction__60m",
        "quality__hr__latest_age_seconds__60m",
        "quality__hr__maximum_gap_seconds__60m",
    ]:
        x[col] = (matched[col] - matched[col].mean()) / matched[col].std(ddof=0)
    x = pd.concat(
        [
            x,
            pd.get_dummies(
                matched.n_survey_category, prefix="survey", drop_first=True, dtype=float
            ),
        ],
        axis=1,
    )
    fit = random_intercept_reml(x, matched.pss_combo, matched.n_participant)
    mixed = {k: v for k, v in fit.items() if k not in ["beta", "covariance"]}
    mixed["coefficients"] = [
        dict(term=c, estimate=b, model_se=float(np.sqrt(fit["covariance"][j, j])))
        for j, (c, b) in enumerate(zip(x.columns, fit["beta"]))
    ]
    mixed["interpretation"] = (
        "Retrospective Gaussian random-intercept REML on observed cohort; model-based SE, no causal or unseen-person predictive interpretation"
    )
    # Independent reproduction of lag-one correlations, retaining no predictive person means.
    pairs = []
    for person, z in f.dropna(subset=["pss_combo"]).groupby("n_participant"):
        z = z.sort_values("n_actual_epoch")
        mean = z.pss_combo.mean()
        for a, b in zip(z.pss_combo.iloc[:-1], z.pss_combo.iloc[1:]):
            pairs.append([a, b, a - mean, b - mean])
    a = np.array(pairs)
    lag = dict(
        pairs=len(a),
        pooled=float(np.corrcoef(a[:, :2].T)[0, 1]),
        person_centered=float(np.corrcoef(a[:, 2:].T)[0, 1]),
        verified_online_history_rows=0,
    )
    boundary = pd.read_parquet(
        root / "private/recall_diagnostics/runs/0b217d08cb1c3dce/boundaries.parquet"
    )
    recall = (
        boundary.groupby(["scenario", "n_survey_category", "boundary_reason"], observed=True)
        .agg(responses=("n_ema_id", "size"), participants=("n_participant", "nunique"))
        .reset_index()
        .to_dict("records")
    )
    probe = pd.read_parquet(data / "activity_probe.parquet")
    probe_counts = (
        probe.groupby(["eligible", "reasons"], dropna=False)
        .agg(intervals=("n_ema_id", "size"), participants=("n_participant", "nunique"))
        .reset_index()
        .to_dict("records")
    )
    probe_dist = []
    for label, g in probe[probe.eligible].groupby("target"):
        for col in HR + [
            "supported_seconds",
            "step_gap_seconds",
            "hr_gap_seconds",
            "hr_latest_age_seconds",
            "occupied_minutes",
        ]:
            probe_dist.append(
                dict(
                    label=int(label),
                    measure=col,
                    intervals=len(g),
                    participants=g.n_participant.nunique(),
                    q25=float(g[col].quantile(0.25)),
                    median=float(g[col].median()),
                    q75=float(g[col].quantile(0.75)),
                )
            )
    by_person = probe[probe.eligible].groupby("n_participant").target.nunique()
    probe_overview = dict(
        candidate_unique_intervals=len(probe),
        eligible_intervals=int(probe.eligible.sum()),
        eligible_participants=len(by_person),
        participants_with_both=int(by_person.gt(1).sum()),
        candidate_unique_hours=len(probe) * 5 / 60,
        eligible_observation_hours=int(probe.eligible.sum()) * 5 / 60,
        warning="Observation hours are not continuous supported physiological time; activity semantics provisional",
    )
    for name, obj in dict(
        distributions=distributions,
        variation=variation,
        class_mix=mix,
        survey=survey,
        composition=composition,
        associations=associations,
        mixed_effects=mixed,
        lag=lag,
        recall_availability=recall,
        probe_exclusions=probe_counts,
        probe_distributions=probe_dist,
        probe_overview=probe_overview,
    ).items():
        write_json(out / f"{name}.json", finite(obj))
    return out


if __name__ == "__main__":
    print(describe(Path.cwd()))
