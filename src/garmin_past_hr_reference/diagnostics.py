"""Strictly retrospective aggregate diagnostics; no row-level person means are exported."""

import numpy as np
import pandas as pd
from garmin_ema_baseline.models import cohort_masks, feature_columns

ITEMS = ["PSS_NERVOUS_value", "PSS_COPE_value", "PSS_CONTROL_value"]


def decomposition(frame, column):
    f = frame[["n_participant", column]].dropna()
    y = f[column]
    if not len(f):
        return dict(valid_responses=0, participants=0)
    g = f.groupby("n_participant")[column]
    means = g.mean()
    counts = g.size()
    centered = y - g.transform("mean")
    within = float(np.square(centered).sum())
    between = float((counts * np.square(means - y.mean())).sum())
    total = float(np.square(y - y.mean()).sum())
    assert np.isclose(total, within + between, atol=1e-9)
    return dict(
        valid_responses=len(f),
        participants=len(means),
        mean=float(y.mean()),
        sd=float(y.std(ddof=1)),
        within_ss=within,
        between_ss=between,
        total_ss=total,
        within_ss_fraction=within / total if total else None,
        between_ss_fraction=between / total if total else None,
        pooled_within_sd=np.sqrt(within / (len(f) - len(means))) if len(f) > len(means) else None,
        person_mean_sd=float(means.std(ddof=1)),
        median_person_sd=float(g.std(ddof=1).median()),
        singleton_participants=int(counts.eq(1).sum()),
        zero_valid_score_participants=int(frame.n_participant.nunique() - len(means)),
    )


def association_summary(frame, feature, draws, seed):
    arrays = []
    for _, f in frame.groupby("n_participant", sort=True):
        x = f[feature].to_numpy(float)
        y = f.pss_combo.to_numpy(float)
        arrays.append(
            (
                len(x),
                x.mean(),
                y.mean(),
                np.dot(x - x.mean(), x - x.mean()),
                np.dot(y - y.mean(), y - y.mean()),
                np.dot(x - x.mean(), y - y.mean()),
            )
        )
    a = np.array(arrays)
    n = len(a)

    def stat(indices, mode):
        z = a[indices]
        if mode == "between_person_means":
            x = z[:, 1] - z[:, 1].mean()
            y = z[:, 2] - z[:, 2].mean()
            xx = x @ x
            yy = y @ y
            xy = x @ y
        else:
            z = z[z[:, 0] >= 2]
            xx, yy, xy = z[:, 3:].sum(axis=0)
        return (
            xy / np.sqrt(xx * yy) if xx > 0 and yy > 0 else np.nan,
            xy / xx if xx > 0 else np.nan,
        )

    samples = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    result = []
    for mode in ["between_person_means", "within_person_centered"]:
        corr, slope = stat(np.arange(n), mode)
        boot = np.array([stat(ix, mode) for ix in samples])
        result.append(
            dict(
                feature=feature,
                mode=mode,
                responses=len(frame)
                if mode == "between_person_means"
                else int(a[a[:, 0] >= 2, 0].sum()),
                participants=n if mode == "between_person_means" else int((a[:, 0] >= 2).sum()),
                correlation=corr,
                slope=slope,
                correlation_low=float(np.nanquantile(boot[:, 0], 0.025)),
                correlation_high=float(np.nanquantile(boot[:, 0], 0.975)),
                slope_low=float(np.nanquantile(boot[:, 1], 0.025)),
                slope_high=float(np.nanquantile(boot[:, 1], 0.975)),
                defined_bootstrap_draws=int(np.isfinite(boot).all(axis=1).sum()),
            )
        )
    return result


def retrospective_diagnostics(linked, config):
    masks = {
        "all_development": pd.Series(True, index=linked.index),
        "native_60m_hr": cohort_masks(linked)[("native", "hr", 60)],
    }
    variance, counts, items, count_frequencies = [], [], [], []
    for scope, mask in masks.items():
        frame = linked[mask]
        for column in ["pss_combo"] + ITEMS:
            variance.append(dict(scope=scope, measure=column, **decomposition(frame, column)))
        all_counts = frame.groupby("n_participant").size()
        valid = (
            frame.dropna(subset=["pss_combo"])
            .groupby("n_participant")
            .size()
            .reindex(all_counts.index, fill_value=0)
        )
        for kind, c in [("all_responses", all_counts), ("valid_composite", valid)]:
            counts.append(
                dict(
                    scope=scope,
                    count_kind=kind,
                    participants=len(c),
                    responses=int(c.sum()),
                    minimum=int(c.min()),
                    q25=float(c.quantile(0.25)),
                    median=float(c.median()),
                    q75=float(c.quantile(0.75)),
                    maximum=int(c.max()),
                    at_least_two=int(c.ge(2).sum()),
                )
            )
            for number, people in c.value_counts().sort_index().items():
                count_frequencies.append(
                    dict(
                        scope=scope,
                        count_kind=kind,
                        responses_per_person=int(number),
                        participants=int(people),
                    )
                )
        for category, f in [("All", frame)] + list(frame.groupby("n_survey_category")):
            for item in ITEMS:
                value = f[item]
                row = dict(
                    scope=scope,
                    survey_category=category,
                    item=item,
                    responses=len(f),
                    valid=int(value.notna().sum()),
                    missing=int(value.isna().sum()),
                    stored_mean=float(value.mean()),
                    stored_sd=float(value.std()),
                    composite_contribution_mean=float(
                        value.mean() if item.startswith("PSS_NERVOUS") else 6 - value.mean()
                    ),
                )
                row.update({f"code_{k}": int(value.eq(k).sum()) for k in range(1, 6)})
                items.append(row)
    cohort = linked[masks["native_60m_hr"]]
    associations = [
        row
        for feature in feature_columns(60)
        for row in association_summary(
            cohort, feature, config["bootstrap_draws"], config["bootstrap_seed"]
        )
    ]
    return {
        name: pd.DataFrame(rows)
        for name, rows in [
            ("variation", variance),
            ("repeated_counts", counts),
            ("response_count_distribution", count_frequencies),
            ("component_items", items),
            ("retrospective_associations", associations),
        ]
    }
