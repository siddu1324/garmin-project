"""Retrospective descriptive diagnostics; never exported as predictive features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from garmin_binary_6hour.diagnostics import quantiles
from garmin_comprehensive_regression.evaluate import finite
from garmin_preprocessing.common import write_json, verify_bundle
from .data import data
from .design import WINDOWS


def describe(root):
    root = Path(root)
    run = root / json.loads((root / "private/binary_improvement/current.json").read_text())["run"]
    out = root / "reports/binary_improvement" / run.name
    out.mkdir(exist_ok=True)
    if (out / "manifest.json").exists():
        verify_bundle(out)
        return out
    frame, blocks, cfg = data(root)
    f = frame[frame.pss_combo.notna() & frame.context_available].copy()
    differences = []
    for m in WINDOWS:
        x = f[f"hr__mean__{m}m"]
        y = f[f"hr__observed_minute_mean__{m}m"]
        ok = x.notna() & y.notna()
        differences.append(
            dict(
                window=m,
                paired_responses=int(ok.sum()),
                paired_participants=f.loc[ok, "n_participant"].nunique(),
                correlation=float(x[ok].corr(y[ok])),
                identical_fraction=float(np.isclose(x[ok], y[ok], atol=1e-8).mean()),
                **{
                    "absolute_difference_" + k: v
                    for k, v in quantiles((x[ok] - y[ok]).abs()).items()
                },
            )
        )
    correlations = []
    for m in WINDOWS:
        x = f[blocks[f"full{m}"]]
        c = x.corr(min_periods=30).to_numpy()
        values = c[np.triu_indices(len(c), 1)]
        correlations.append(
            dict(
                window=m,
                features=x.shape[1],
                all_missing=int(x.isna().all().sum()),
                constant_observed=int((x.nunique() <= 1).sum()),
                defined_pairs=int(np.isfinite(values).sum()),
                pairs_abs_r_over_095=int((np.abs(values) > 0.95).sum()),
                median_abs_r=float(np.nanmedian(np.abs(values))),
            )
        )
    # Class contrasts are descriptive, using only rows with actual observed feature values.
    # Full-data context residuals are never merged into a feature matrix.
    associations = []
    cols = list(dict.fromkeys(blocks["compact360"] + blocks["history360"] + blocks["minute1440"]))
    for task in ["A", "B"]:
        for col in cols:
            g = f[f[col].notna()].copy()
            if len(g) < 30 or g[f"target_{task}"].nunique() != 2 or g[col].std() == 0:
                continue
            context = pd.get_dummies(
                g[["n_survey_category", "local_time_sin", "local_time_cos"]], dtype=float
            )
            x = g[col].to_numpy(float)
            residual = x - LinearRegression().fit(context, x).predict(context)
            g["observed"] = x
            g["context_residual"] = residual
            y = g[f"target_{task}"].to_numpy(int)
            for kind in ["observed", "context_residual"]:
                v = g[kind]
                sd = float(v.std(ddof=1))
                rawdiff = float(v[y == 1].mean() - v[y == 0].mean())
                part = g.groupby(["n_participant", f"target_{task}"])[kind].mean().unstack()
                paired = part.dropna()
                person_diff = (paired[1.0] - paired[0.0]) if len(paired) else pd.Series(dtype=float)
                # Leave-one-participant-out ranges of the response-weighted class contrast.
                totals = g.groupby(["n_participant", f"target_{task}"])[kind].agg(["sum", "count"])
                overall = g.groupby(f"target_{task}")[kind].agg(["sum", "count"])
                loo = []
                for person in g.n_participant.unique():
                    z = overall - totals.loc[person].reindex(overall.index, fill_value=0)
                    if z["count"].gt(0).all():
                        means = z["sum"] / z["count"]
                        loo.append(float(means.loc[1.0] - means.loc[0.0]))
                associations.append(
                    dict(
                        task=task,
                        feature=col,
                        representation=kind,
                        responses=len(g),
                        participants=g.n_participant.nunique(),
                        class0=int((y == 0).sum()),
                        class1=int((y == 1).sum()),
                        class1_minus_class0=rawdiff,
                        standardized_difference=rawdiff / sd if sd else None,
                        within_person_contrast_participants=len(paired),
                        within_person_equal_weight_difference=float(person_diff.mean())
                        if len(paired)
                        else None,
                        leave_one_person_out_difference_min=min(loo),
                        leave_one_person_out_difference_max=max(loo),
                    )
                )
    # Source nestedness gives additional accepted counts, not independent segments or complete coverage.
    q = pd.read_parquet(
        root / cfg["feature_bundle"] / "quality.parquet", filters=[("sensor", "=", "hr")]
    )
    lin = pd.read_parquet(
        root / cfg["feature_bundle"] / "lineage.parquet", filters=[("sensor", "=", "hr")]
    )
    p = lin.merge(
        q[["n_ema_id", "lookback_minutes", "accepted_readings"]],
        on=["n_ema_id", "lookback_minutes"],
        validate="one_to_one",
    )
    p = p[p.n_ema_id.isin(f.n_ema_id)]
    base = p[p.lookback_minutes.eq(360)].set_index("n_ema_id")
    additions = []
    for m in [720, 1440]:
        z = p[p.lookback_minutes.eq(m)].set_index("n_ema_id").loc[base.index]
        assert z.end_epoch.equals(base.end_epoch) and z.group_stop.equals(base.group_stop)
        assert z.directory.equals(base.directory) and (z.group_start <= base.group_start).all()
        d = z.accepted_readings - base.accepted_readings
        assert d.ge(0).all()
        additions.append(
            dict(
                window=m,
                reference=360,
                responses=len(d),
                gain=int(d.gt(0).sum()),
                no_gain=int(d.eq(0).sum()),
                **quantiles(d),
            )
        )
    for name, values in [
        ("weighting_differences", differences),
        ("feature_correlations", correlations),
        ("class_associations", associations),
        ("new_readings", additions),
    ]:
        write_json(out / f"{name}.json", finite(values))
    return out


if __name__ == "__main__":
    print(describe(Path.cwd()))
