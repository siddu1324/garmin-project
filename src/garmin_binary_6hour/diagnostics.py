"""Read-only support and missingness audit from existing feature provenance."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_comprehensive_regression.evaluate import finite
from garmin_preprocessing.common import write_json
from .models import WINDOWS, cohort
from .run import data


def quantiles(x):
    x = pd.Series(x).dropna().astype(float)
    return dict(
        observed=len(x),
        median=float(x.median()) if len(x) else None,
        p25=float(x.quantile(0.25)) if len(x) else None,
        p75=float(x.quantile(0.75)) if len(x) else None,
        p90=float(x.quantile(0.9)) if len(x) else None,
    )


def audit(root):
    root = Path(root)
    frame, blocks, cfg = data(root)
    run = root / json.loads((root / "private/binary_6hour/current.json").read_text())["run"]
    out = root / "reports/binary_6hour" / run.name
    out.mkdir(exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("Sealed diagnostics must not be overwritten.")
    folder = root / cfg["feature_bundle"]
    q = pd.read_parquet(folder / "quality.parquet")
    q = q[q.sensor.eq("hr") & q.lookback_minutes.isin(WINDOWS)]
    lineage = pd.read_parquet(folder / "lineage.parquet")
    lineage = lineage[lineage.sensor.eq("hr") & lineage.lookback_minutes.isin(WINDOWS)]
    assert not q.duplicated(["n_ema_id", "lookback_minutes"]).any()
    f = cohort(frame, blocks, "all")
    accounting = dict(
        all_development_responses=len(frame),
        development_participants=frame.n_participant.nunique(),
        valid_outcome_responses=int(frame.pss_combo.notna().sum()),
        missing_outcome_responses=int(frame.pss_combo.isna().sum()),
        missing_context_among_valid=int((frame.pss_combo.notna() & ~frame.context_available).sum()),
        primary_responses=len(f),
        primary_participants=f.n_participant.nunique(),
    )
    complete = set(cohort(frame, blocks, "complete").n_ema_id)
    quality = f[
        ["n_ema_id", "n_participant", "n_fold", "n_survey_category", "target_A", "target_B"]
    ].merge(q, on="n_ema_id", how="left", validate="one_to_many")
    quality.to_parquet(run / "quality_accounting.parquet", index=False)
    distributions, coverage, missingness, quality_stats, additions, composition = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for task in ["A", "B"]:
        target = f"target_{task}"
        for class_id in [0, 1]:
            selected = f[f[target].eq(class_id)]
            distributions.append(
                dict(
                    task=task,
                    class_id=class_id,
                    responses=len(selected),
                    participants=selected.n_participant.nunique(),
                    fraction=len(selected) / len(f),
                )
            )
        for m in WINDOWS:
            for b in ["original", "hr", "activity", "full"]:
                cols = blocks[f"{b}{m}"]
                available = f[cols].notna().all(axis=1)
                coverage.append(
                    dict(
                        task=task,
                        window=m,
                        block=b,
                        features=len(cols),
                        complete_responses=int(available.sum()),
                        complete_participants=f.loc[available, "n_participant"].nunique(),
                        any_observed_responses=int(f[cols].notna().any(axis=1).sum()),
                        all_valid_responses=len(f),
                    )
                )
            for c in blocks[f"full{m}"]:
                rates = {
                    str(cls): float(f.loc[f[target].eq(cls), c].isna().mean()) for cls in [0, 1]
                }
                missingness.append(
                    dict(
                        task=task,
                        window=m,
                        feature=c,
                        missing_fraction=float(f[c].isna().mean()),
                        class0_missing=rates["0"],
                        class1_missing=rates["1"],
                        class1_minus_class0=rates["1"] - rates["0"],
                    )
                )
            qm = quality[quality.lookback_minutes.eq(m)]
            for category in ["All", "Morning", "Midday", "Endday"]:
                for cls in ["All", 0, 1]:
                    g = qm if category == "All" else qm[qm.n_survey_category.eq(category)]
                    if cls != "All":
                        g = g[g[target].eq(cls)]
                    for metric in [
                        "recorded_groups",
                        "accepted_readings",
                        "occupied_minutes",
                        "occupied_fraction",
                        "span_seconds",
                        "latest_age_seconds",
                        "left_gap_seconds",
                        "maximum_internal_gap_seconds",
                        "maximum_gap_seconds",
                    ]:
                        quality_stats.append(
                            dict(
                                task=task,
                                window=m,
                                survey=category,
                                class_id=cls,
                                metric=metric,
                                responses=len(g),
                                participants=g.n_participant.nunique(),
                                zero_responses=int(g[metric].eq(0).sum()),
                                **quantiles(g[metric]),
                            )
                        )
        for label, keep in [
            ("common_complete_retained", f.n_ema_id.isin(complete)),
            ("common_complete_excluded", ~f.n_ema_id.isin(complete)),
        ]:
            for category in ["All", "Morning", "Midday", "Endday"]:
                g = f[keep]
                if category != "All":
                    g = g[g.n_survey_category.eq(category)]
                composition.append(
                    dict(
                        task=task,
                        group=label,
                        survey=category,
                        responses=len(g),
                        participants=g.n_participant.nunique(),
                        class0=int(g[target].eq(0).sum()),
                        class1=int(g[target].eq(1).sum()),
                        pss_mean=float(g.pss_combo.mean()),
                        pss_median=float(g.pss_combo.median()),
                    )
                )
    # Verify nested source slices before interpreting accepted-count differences.
    bounds = lineage.merge(
        q[["n_ema_id", "lookback_minutes", "accepted_readings"]],
        on=["n_ema_id", "lookback_minutes"],
        validate="one_to_one",
    )
    bounds = bounds[bounds.n_ema_id.isin(f.n_ema_id)]
    checks = []
    for long in [120, 360]:
        a = bounds[bounds.lookback_minutes.eq(60)].set_index("n_ema_id")
        b = bounds[bounds.lookback_minutes.eq(long)].set_index("n_ema_id").loc[a.index]
        assert a.directory.equals(b.directory)
        assert a.end_epoch.equals(b.end_epoch)
        assert a.group_stop.equals(b.group_stop)
        assert (b.group_start <= a.group_start).all()
        assert np.allclose(a.end_epoch - a.start_epoch, 60 * 60)
        assert np.allclose(b.end_epoch - b.start_epoch, long * 60)
        added = b.accepted_readings - a.accepted_readings
        assert (added >= 0).all()
        private = pd.DataFrame(
            dict(
                n_ema_id=a.index,
                additional_readings=added.to_numpy(),
                shared_accepted=a.accepted_readings.to_numpy(),
            )
        )
        private.to_parquet(run / f"new_readings_{long}m.parquet", index=False)
        for category in ["All", "Morning", "Midday", "Endday"]:
            g = private.merge(
                f[["n_ema_id", "n_participant", "n_survey_category"]],
                on="n_ema_id",
                validate="one_to_one",
            )
            if category != "All":
                g = g[g.n_survey_category.eq(category)]
            additions.append(
                dict(
                    window=long,
                    reference_window=60,
                    survey=category,
                    responses=len(g),
                    participants=g.n_participant.nunique(),
                    no_new_readings=int(g.additional_readings.eq(0).sum()),
                    new_readings=int(g.additional_readings.gt(0).sum()),
                    gained_from_zero=int(
                        (g.shared_accepted.eq(0) & g.additional_readings.gt(0)).sum()
                    ),
                    **quantiles(g.additional_readings),
                )
            )
        checks.append(
            dict(window=long, verified_nested_slices=len(a), accepted_count_monotonic=True)
        )
    inventory = []
    for name, cols in blocks.items():
        inventory.append(
            dict(
                block=name,
                physiological_candidates=len(cols),
                columns=cols,
                role="recording_quality_control"
                if name == "quality360"
                else "separate_multiscale"
                if name == "multiscale"
                else "single_window_or_context",
            )
        )
    for name, value in dict(
        distributions=distributions,
        coverage=coverage,
        missingness=missingness,
        quality=quality_stats,
        new_readings=additions,
        composition=composition,
        lineage_checks=checks,
        feature_inventory=inventory,
        accounting_summary=accounting,
    ).items():
        write_json(out / f"{name}.json", finite(value))
    return out


if __name__ == "__main__":
    print(audit(Path.cwd()))
