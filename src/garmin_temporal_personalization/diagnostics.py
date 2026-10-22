"""Observed-data associations and cohort accounting, never predictive residuals."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from garmin_preprocessing.common import write_json, verify_bundle
from garmin_comprehensive_regression.evaluate import finite
from .data import RAW, PERSONAL, EXTRA, QUALITY
from .models import cohort


def adjusted_correlation(z, col):
    ctx = pd.get_dummies(
        z[["n_survey_category", "local_time_sin", "local_time_cos", "steps__supported_rate__60m"]],
        dtype=float,
    )
    x = np.column_stack([np.ones(len(z)), ctx.to_numpy(float)])
    a = z[col].to_numpy(float)
    b = z.pss_combo.to_numpy(float)
    ra = a - x @ np.linalg.lstsq(x, a, rcond=None)[0]
    rb = b - x @ np.linalg.lstsq(x, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 1e-10 and rb.std() > 1e-10 else np.nan


def describe(root):
    root = Path(root)
    ptr = json.loads((root / "private/temporal_personalization/current.json").read_text())
    run = root / ptr["run"]
    out = root / "reports/temporal_personalization" / run.name
    if (out / "manifest.json").exists():
        verify_bundle(out)
        return out
    out.mkdir(exist_ok=True)
    f = pd.read_parquet(root / ptr["features"] / "development.parquet")
    availability = []
    composition = []
    coverage = []
    for co in ["all", "observed", "supported"]:
        g = cohort(f, co)
        for included, z in [("retained", g), ("excluded", f[~f.n_ema_id.isin(g.n_ema_id)])]:
            for survey, part in z.groupby("n_survey_category"):
                composition.append(
                    dict(
                        cohort=co,
                        group=included,
                        survey=survey,
                        responses=len(part),
                        participants=part.n_participant.nunique(),
                        valid_outcomes=int(part.pss_combo.notna().sum()),
                        mean_score=float(part.pss_combo.mean()),
                        median_score=float(part.pss_combo.median()),
                    )
                )
        for col in dict.fromkeys(RAW + PERSONAL + EXTRA):
            z = g[g[col].notna()]
            availability.append(
                dict(
                    cohort=co,
                    feature=col,
                    responses=len(g),
                    available=len(z),
                    available_participants=z.n_participant.nunique(),
                    missing=len(g) - len(z),
                )
            )
        for col in QUALITY + [
            "q_clock_recorded_minutes",
            "q_clock_utc_dates",
            "q_recent_step_seconds",
            "q_earlier_step_seconds",
        ]:
            z = g[col].dropna()
            coverage.append(
                dict(
                    cohort=co,
                    diagnostic=col,
                    observed=len(z),
                    q25=float(z.quantile(0.25)),
                    median=float(z.median()),
                    q75=float(z.quantile(0.75)),
                )
            )
    associations = []
    main = cohort(f, "all")
    for col in ["hr__mean__60m", "hr__sd__60m", "hr_mean_deviation_bpm", "clock_mean_deviation"]:
        z = main.dropna(subset=[col, "steps__supported_rate__60m"]).copy()
        if len(z) < 30:
            continue
        point = adjusted_correlation(z, col)
        people = z.n_participant.unique()
        rng = np.random.default_rng(20261020)
        draws = []
        for _ in range(500):
            sample = rng.choice(people, len(people), replace=True)
            boot = pd.concat([z[z.n_participant.eq(p)] for p in sample], ignore_index=True)
            draws.append(adjusted_correlation(boot, col))
        low, high = np.nanquantile(draws, [0.025, 0.975])
        associations.append(
            dict(
                feature=col,
                responses=len(z),
                participants=len(people),
                context_activity_adjusted_correlation=point,
                lower=low,
                upper=high,
                raw_correlation=float(z[col].corr(z.pss_combo)),
                bootstrap_draws=500,
                meaning="Retrospective association; refit nuisance adjustment per participant draw; not predictive accuracy",
            )
        )
    for name, obj in [
        ("availability", availability),
        ("composition", composition),
        ("coverage", coverage),
        ("associations", associations),
    ]:
        write_json(out / f"{name}.json", finite(obj))
    return out


if __name__ == "__main__":
    print(describe(Path.cwd()))
