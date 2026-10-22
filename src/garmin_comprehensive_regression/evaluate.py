"""Matched OOF evaluation and participant-cluster uncertainty; public aggregates only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_preprocessing.common import verify_bundle, write_json
from .run import TARGET, basic, cohort, data


def finite(value):
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def cluster_values(f, draws, seed):
    f = f.sort_values("n_ema_id")
    y, p = f[TARGET].to_numpy(float), f.prediction.to_numpy(float)
    people, inverse = np.unique(f.n_participant, return_inverse=True)
    n = np.bincount(inverse)
    absolute = np.bincount(inverse, weights=np.abs(p - y))
    squared = np.bincount(inverse, weights=(p - y) ** 2)
    sy = np.bincount(inverse, weights=y)
    sy2 = np.bincount(inverse, weights=y * y)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(people), size=(draws, len(people)))
    size = n[sample].sum(axis=1)
    se = squared[sample].sum(axis=1)
    total = sy2[sample].sum(axis=1) - sy[sample].sum(axis=1) ** 2 / size
    values = dict(
        mae=absolute[sample].sum(axis=1) / size,
        rmse=np.sqrt(se / size),
        r2=np.divide(se, total, out=np.full(draws, np.nan), where=total > 0),
        participant_mae=(absolute / n)[sample].mean(axis=1),
    )
    values["r2"] = 1 - values["r2"]
    return values


def metrics(f, cfg=None):
    out = basic(f[TARGET].to_numpy(float), f.prediction.to_numpy(float))
    out.update(
        responses=len(f),
        participants=f.n_participant.nunique(),
        folds=f.n_fold.nunique(),
        participant_mae=float(
            f.assign(error=np.abs(f.prediction - f[TARGET]))
            .groupby("n_participant")
            .error.mean()
            .mean()
        ),
        near_median_fraction=float(np.mean(np.abs(f.prediction - f.dummy_median) <= 0.25))
        if "dummy_median" in f
        else None,
    )
    if cfg:
        for key, values in cluster_values(f, cfg["bootstrap_draws"], cfg["seed"]).items():
            low, high = np.nanquantile(values, [0.025, 0.975])
            out[key + "_low"], out[key + "_high"] = float(low), float(high)
    return out


def paired(a, b, cfg):
    original_a, original_b = len(a), len(b)
    common = sorted(set(a.n_ema_id) & set(b.n_ema_id))
    if not common:
        return dict(status="no_common_rows", responses=0)
    a = a.set_index("n_ema_id").loc[common].reset_index()
    b = b.set_index("n_ema_id").loc[common].reset_index()
    for key in [TARGET, "n_participant", "n_fold"]:
        assert a[key].equals(b[key])
    ma, mb = metrics(a), metrics(b)
    va = cluster_values(a, cfg["bootstrap_draws"], cfg["seed"])
    vb = cluster_values(b, cfg["bootstrap_draws"], cfg["seed"])
    out = dict(
        status="matched",
        responses=len(a),
        participants=a.n_participant.nunique(),
        folds=a.n_fold.nunique(),
        full_membership_match=original_a == original_b == len(common),
    )
    for key in ["mae", "rmse", "participant_mae"]:
        low, high = np.quantile(va[key] - vb[key], [0.025, 0.975])
        out.update(
            {
                key + "_difference": ma[key] - mb[key],
                key + "_low": float(low),
                key + "_high": float(high),
            }
        )
    return out


def score_band(y):
    return np.where(y <= 4, "3–4", np.where(y <= 9, "5–9", "10–15"))


def evaluate(root):
    root = Path(root)
    pointer = json.loads((root / "private/comprehensive_regression/current.json").read_text())
    run = root / pointer["run"]
    public = root / "reports/comprehensive_regression" / run.name
    if (public / "manifest.json").exists():
        verify_bundle(public)
        return public
    identity = json.loads((run / "run_identity.json").read_text())
    cfg = identity["config"]
    expected = len(json.loads((run / "specifications.json").read_text())) * 5
    summaries = sorted((run / "jobs").glob("*/summary.json"))
    if len(summaries) != expected:
        raise RuntimeError(f"Incomplete experiment: {len(summaries)}/{expected} jobs.")
    public.mkdir(parents=True, exist_ok=True)
    prediction_frames, settings, train_scores, failed, feature_rows = [], [], [], [], []
    for path in summaries:
        verify_bundle(path.parent)
        info = json.loads(path.read_text())
        s = info["spec"]
        tags = dict(cohort=s["cohort"], view=s["view"], family=s["family"], model=s["name"])
        if info["status"] != "complete":
            failed.append(dict(**tags, fold=info["fold"], error_type=info["error_type"]))
            continue
        f = pd.read_parquet(path.parent / "predictions.parquet")
        prediction_frames.append(f.assign(**tags))
        settings.append(
            dict(
                **tags,
                fold=info["fold"],
                setting=info["best"]["setting"],
                inner_mae=info["best"]["inner_mae"],
                train_mae=info["train"]["mae"],
                validation_mae=info["validation"]["mae"],
                transformed_features=info["transformed_features"],
                selected_features=info["selected_features"],
                nonzero_coefficients=info["nonzero_coefficients"],
                training_responses=info["training_responses"],
                validation_responses=info["validation_responses"],
                training_participants=info["training_participants"],
                validation_participants=info["validation_participants"],
                nonconverged_candidates=sum(
                    not t["converged"]
                    for t in json.loads((path.parent / "trials.json").read_text())
                ),
            )
        )
        train_scores.append(dict(**tags, fold=info["fold"], **info["train"]))
        features = json.loads((path.parent / "features.json").read_text())
        for i, name in enumerate(features["names"]):
            value = features["coefficient"][i] if features["coefficient"] is not None else None
            importance = features["importance"][i] if features["importance"] is not None else None
            feature_rows.append(
                dict(
                    **tags,
                    fold=info["fold"],
                    feature=name,
                    coefficient=value,
                    importance=importance,
                )
            )
    prediction = pd.concat(prediction_frames, ignore_index=True)
    # One dummy view per cohort/fold; duplicated dummy values must agree across model jobs.
    dummy_frames = []
    for cname, group in prediction.groupby("cohort"):
        for col in ["dummy_mean", "dummy_median"]:
            assert group.groupby("n_ema_id")[col].nunique().eq(1).all()
            d = group.drop_duplicates("n_ema_id").copy()
            d["prediction"], d["family"], d["view"] = d[col], col, "context"
            d["model"] = cname + "__" + col
            dummy_frames.append(d)
    prediction = pd.concat([prediction, *dummy_frames], ignore_index=True)
    assert not prediction.duplicated(["model", "n_ema_id"]).any()
    prediction.to_parquet(run / "all_predictions.parquet", index=False)
    grouped = {key: g.copy() for key, g in prediction.groupby("model")}
    pooled, folds, strata = [], [], []
    for name, f in grouped.items():
        tags = {k: f[k].iloc[0] for k in ["cohort", "view", "family", "model"]}
        pooled.append(dict(**tags, complete_outer_folds=f.n_fold.nunique() == 5, **metrics(f, cfg)))
        for fold, g in f.groupby("n_fold"):
            folds.append(dict(**tags, fold=int(fold), **metrics(g)))
        for dimension, values in [
            ("score_band", score_band(f[TARGET])),
            ("survey", f.n_survey_category.to_numpy()),
        ]:
            for label in np.unique(values):
                g = f.loc[values == label]
                strata.append(
                    dict(
                        **tags,
                        dimension=dimension,
                        stratum=str(label),
                        actual_mean=float(g[TARGET].mean()),
                        prediction_mean=float(g.prediction.mean()),
                        **metrics(g, cfg),
                    )
                )
    comparisons = set()
    for name, f in grouped.items():
        cname, family, view = f.cohort.iloc[0], f.family.iloc[0], f["view"].iloc[0]
        if family.startswith("dummy_"):
            continue
        for base in [
            cname + "__dummy_mean",
            cname + "__dummy_median",
            cname + "__context__" + family,
        ]:
            if base in grouped and base != name:
                comparisons.add((name, base))
        if view in [
            "full",
            "full_corr",
            "full_l1select",
            "full_enetselect",
            "pulseox_full",
            "full_no_flags",
        ]:
            for b in ["original60", "hr_all"] if view == "full" else ["full"]:
                base = f"{cname}__{b}__{family}"
                if base in grouped:
                    comparisons.add((name, base))
        if view == "full" and family != "ridge" and cname + "__full__ridge" in grouped:
            comparisons.add((name, cname + "__full__ridge"))
        if view in ["missingness", "quality"] and cname + "__full__" + family in grouped:
            comparisons.add((cname + "__full__" + family, name))
    differences = [
        dict(model=a, baseline=b, **paired(grouped[a], grouped[b], cfg))
        for a, b in sorted(comparisons)
    ]
    # Compare imputed predictions on complete60 validation responses separately from refits.
    complete_ids = set(prediction[prediction.cohort.eq("complete60")].n_ema_id)
    restricted = []
    for name, f in grouped.items():
        if f.cohort.iloc[0] == "all_valid" and f["view"].iloc[0] in [
            "context",
            "original60",
            "full",
        ]:
            g = f[f.n_ema_id.isin(complete_ids)]
            if len(g):
                restricted.append(
                    dict(
                        model=name,
                        evaluation="complete60_only_training_all_valid",
                        **metrics(g, cfg),
                    )
                )
                counterpart = name.replace("all_valid__", "complete60__")
                if counterpart in grouped:
                    differences.append(
                        dict(
                            model=name + " [restricted validation]",
                            baseline=counterpart,
                            **paired(g, grouped[counterpart], cfg),
                        )
                    )
    legacy, replay = legacy_comparison(root, cfg, grouped, complete_ids)
    frame, blocks = data(root, cfg)
    composition, availability = [], []
    for cname in ["all_valid", "complete60"]:
        ids = set(cohort(frame, blocks, cname).n_ema_id)
        for included in [True, False]:
            f = frame[frame.n_ema_id.isin(ids).eq(included)]
            for survey, part in [("All", f), *list(f.groupby("n_survey_category"))]:
                valid = part[TARGET].dropna()
                composition.append(
                    dict(
                        cohort=cname,
                        included=included,
                        survey=survey,
                        responses=len(part),
                        participants=part.n_participant.nunique(),
                        scores_available=len(valid),
                        score_mean=float(valid.mean()),
                        score_median=float(valid.median()),
                        score_sd=float(valid.std()),
                        band_counts={
                            band: int((score_band(valid) == band).sum())
                            for band in ["3–4", "5–9", "10–15"]
                        },
                    )
                )
    for block in ["original60", "hr60", "hr_all", "hr_activity_all", "full", "pulseox_full"]:
        count = frame[blocks[block]].notna().sum(axis=1)
        for band in ["All", "3–4", "5–9", "10–15"]:
            mask = (
                pd.Series(True, index=frame.index)
                if band == "All"
                else frame[TARGET].notna() & (score_band(frame[TARGET]) == band)
            )
            availability.append(
                dict(
                    block=block,
                    score_band=band,
                    candidate_columns=len(blocks[block]),
                    responses=int(mask.sum()),
                    complete_responses=int(count[mask].eq(len(blocks[block])).sum()),
                    median_observed_columns=float(count[mask].median()),
                    no_observed_columns=int(count[mask].eq(0).sum()),
                )
            )
    importance = []
    ff = pd.DataFrame(feature_rows)
    for (model, feature), g in ff.groupby(["model", "feature"]):
        importance.append(
            dict(
                model=model,
                feature=feature,
                present_folds=len(g),
                nonzero_folds=int(g.coefficient.abs().gt(1e-8).sum())
                if g.coefficient.notna().any()
                else None,
                mean_absolute_coefficient=float(g.coefficient.abs().mean()),
                mean_tree_importance=float(g.importance.mean()),
            )
        )
    outputs = dict(
        metrics=pooled,
        fold_metrics=folds,
        strata=strata,
        paired_differences=differences,
        settings=settings,
        training_metrics=train_scores,
        failures=failed,
        cohort_composition=composition,
        feature_availability=availability,
        feature_stability=importance,
        restricted_evaluation=restricted,
        legacy_comparison=legacy,
        legacy_replay=replay,
        configuration=identity,
        specifications=json.loads((run / "specifications.json").read_text()),
    )
    for name, value in outputs.items():
        write_json(public / (name + ".json"), finite(value))
    return public


def legacy_comparison(root, cfg, grouped, complete_ids):
    rows, replay = [], []
    for notebook, folder in [
        ("05", cfg["legacy_regression_bundle"]),
        ("09", cfg["legacy_benchmark_bundle"]),
    ]:
        verify_bundle(root / folder)
        old = pd.read_parquet(root / folder / "out_of_fold_predictions.parquet")
        if notebook == "05":
            old = old[
                old.scope.eq("native") & old.block.eq("hr_steps") & old.lookback_minutes.eq(60)
            ]
            mapping = {"context": "prediction_context", "original60": "prediction_context_hr_steps"}
        else:
            old = old[old.block.eq("hr_steps")]
            mapping = {
                "context": "prediction_fixed_ridge__context",
                "original60": "prediction_fixed_ridge__context_hr_steps",
            }
        assert set(old.n_ema_id) == complete_ids and not old.n_ema_id.duplicated().any()
        for view, col in mapping.items():
            new = grouped[f"complete60__{view}__fixed_ridge"].set_index("n_ema_id")
            match = old.set_index("n_ema_id").loc[new.index]
            for key in [TARGET, "n_participant", "n_fold"]:
                assert match[key].equals(new[key])
            error = float(np.max(np.abs(match[col] - new.prediction)))
            assert error < 1e-7, "Legacy fixed Ridge replay mismatch; stop publication."
            replay.append(dict(notebook=notebook, view=view, responses=len(new), max_error=error))
        for col in [c for c in old if c.startswith("prediction_") and old[c].notna().all()]:
            g = old.rename(columns={col: "prediction"}).copy()
            rows.append(dict(notebook=notebook, legacy_model=col, **metrics(g, cfg)))
            for family in ["ridge", "hgb"]:
                name = f"complete60__full__{family}"
                if name in grouped:
                    rows.append(
                        dict(
                            notebook=notebook,
                            legacy_model=col,
                            new_model=name,
                            **paired(grouped[name], g, cfg),
                        )
                    )
    return rows, replay


if __name__ == "__main__":
    print(evaluate(Path.cwd()))
