"""Matched outer folds and strictly training-only grouped inner selection."""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from garmin_ema_baseline.models import (
    cohort_masks,
    make_model,
    metrics,
    model_columns,
    paired_bootstrap,
)
from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import digest_object

META = ["n_ema_id", "n_participant", "n_fold", "n_survey_category", "pss_combo"]


def fit_predict(train, valid, view, family, setting, config):
    """Only explicit allowed X columns reach the estimator; never missing-value routing."""
    if set(train.n_participant) & set(valid.n_participant):
        raise AuditError("Participant overlap in model split.")
    numeric, categorical = model_columns(view, 60)
    cols = numeric + categorical
    if (
        not np.isfinite(train[numeric + ["pss_combo"]]).all().all()
        or not np.isfinite(valid[numeric]).all().all()
    ):
        raise AuditError("Complete-case benchmark received missing or nonfinite inputs.")
    model = make_model(view, 60)
    if family == "hgb":
        model.set_params(ridge=HistGradientBoostingRegressor(**config["hgb_fixed"], **setting))
    else:
        model.set_params(ridge__alpha=setting["alpha"])
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model.fit(train[cols], train.pss_combo)
        prediction = model.predict(valid[cols])
        training = model.predict(train[cols])
    transform = model.named_steps["transform"]
    scaler = transform.named_transformers_["numeric"]
    cats = transform.named_transformers_["category"].categories_[0].tolist()
    if not np.allclose(
        scaler.mean_, train[numeric].mean(), atol=1e-12, rtol=1e-12
    ) or cats != sorted(train.n_survey_category.unique()):
        raise AuditError("Training-only preprocessing mismatch.")
    estimator = model.named_steps["ridge"]
    if family == "hgb" and (
        estimator.do_early_stopping_ or estimator.n_iter_ != setting["max_iter"]
    ):
        raise AuditError("Unexpected early stopping.")
    if not np.isfinite(prediction).all() or not np.isfinite(training).all():
        raise AuditError("Nonfinite model prediction.")
    audit = dict(
        input_columns=cols,
        training_mean=scaler.mean_.tolist(),
        training_scale=scaler.scale_.tolist(),
        training_count=int(scaler.n_samples_seen_),
        categories=cats,
        warning_categories=[type(w.message).__name__ for w in caught],
        estimator_parameters=estimator.get_params(),
        coefficient=None if family == "hgb" else estimator.coef_.tolist(),
        intercept=None if family == "hgb" else float(estimator.intercept_),
        iterations=int(estimator.n_iter_) if family == "hgb" else None,
        training_prediction_sd=float(np.std(training)),
        validation_prediction_sd=float(np.std(prediction)),
    )
    return prediction, training, audit


def inner_splits(train):
    if train.n_participant.nunique() < 3:
        return []
    splits = list(GroupKFold(3, shuffle=False).split(train, groups=train.n_participant))
    for a, b in splits:
        if train.iloc[a].n_participant.nunique() < 2 or set(train.iloc[a].n_participant) & set(
            train.iloc[b].n_participant
        ):
            return []
    return splits


def tune(train, view, family, config, splits):
    """No outer validation data are accepted by this interface."""
    grid = (
        [{"alpha": a} for a in config["ridge_alphas"]]
        if family == "tuned_ridge"
        else config["hgb_grid"]
    )
    scores, audits = [], []
    for i, setting in enumerate(grid):
        maes = []
        for inner, (a, b) in enumerate(splits):
            tr, va = train.iloc[a], train.iloc[b]
            p, _, audit = fit_predict(tr, va, view, family, setting, config)
            mae = float(np.mean(np.abs(p - va.pss_combo.to_numpy())))
            maes.append(mae)
            audits.append(
                dict(
                    candidate=i,
                    setting=setting,
                    inner_fold=inner,
                    mae=mae,
                    training_key_sha256=digest_object(tr.n_ema_id.tolist()),
                    validation_key_sha256=digest_object(va.n_ema_id.tolist()),
                    training_responses=len(tr),
                    validation_responses=len(va),
                    training_participants=tr.n_participant.nunique(),
                    validation_participants=va.n_participant.nunique(),
                    **audit,
                )
            )
        scores.append(float(np.mean(maes)))
    selected = int(np.argmin(scores))
    return grid[selected], scores[selected], audits


def run_models(linked, config):
    predictions, training_rows, audits, counts, selections, memberships = [], [], [], [], [], []
    masks = cohort_masks(linked)
    for block in ["hr", "hr_steps"]:
        mask = masks[("native", block, 60)]
        membership = linked[META].copy().assign(block=block, included=mask)
        memberships.append(membership)
        frame = linked.loc[mask].sort_values("n_ema_id").copy()
        views = ["context", "context_hr"] + (["context_hr_steps"] if block == "hr_steps" else [])
        for fold in range(5):
            train, valid = frame[frame.n_fold.ne(fold)], frame[frame.n_fold.eq(fold)]
            splits = inner_splits(train)
            status = "fitted" if len(valid) and len(splits) == 3 else "infeasible_grouped_split"
            tags = dict(block=block, fold=fold)
            counts.append(
                dict(
                    **tags,
                    status=status,
                    training_responses=len(train),
                    validation_responses=len(valid),
                    training_participants=train.n_participant.nunique(),
                    validation_participants=valid.n_participant.nunique(),
                )
            )
            if status != "fitted":
                continue
            vp, tp = valid[META].copy(), train[META].copy()
            for f in [vp, tp]:
                f["block"], f["outer_fold"] = block, fold
            for name in ["mean", "median"]:
                constant = float(getattr(train.pss_combo, name)())
                vp["prediction_" + name] = constant
                tp["prediction_" + name] = constant
            for view in views:
                for family in ["fixed_ridge", "tuned_ridge", "hgb"]:
                    name = family + "__" + view
                    setting, inner_mae, inner_audit = (
                        ({"alpha": 1}, None, [])
                        if family == "fixed_ridge"
                        else tune(train, view, family, config, splits)
                    )
                    pred, trpred, audit = fit_predict(train, valid, view, family, setting, config)
                    vp["prediction_" + name], tp["prediction_" + name] = pred, trpred
                    audits.append(
                        dict(
                            **tags,
                            model=name,
                            setting=setting,
                            inner=inner_audit,
                            training_key_sha256=digest_object(train.n_ema_id.tolist()),
                            validation_key_sha256=digest_object(valid.n_ema_id.tolist()),
                            **audit,
                        )
                    )
                    selections.append(
                        dict(
                            **tags,
                            model=name,
                            setting=json.dumps(setting, sort_keys=True),
                            inner_mae=inner_mae,
                            training_mae=float(np.abs(trpred - train.pss_combo).mean()),
                            validation_mae=float(np.abs(pred - valid.pss_combo).mean()),
                            prediction_sd=float(np.std(pred)),
                            prediction_min=float(pred.min()),
                            prediction_max=float(pred.max()),
                            unique_predictions=int(len(np.unique(np.round(pred, 8)))),
                            warnings=len(audit["warning_categories"]),
                        )
                    )
            predictions.append(vp)
            training_rows.append(tp)
            print(f"Completed {block} outer fold {fold}", flush=True)
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(training_rows, ignore_index=True),
        audits,
        pd.DataFrame(counts),
        pd.DataFrame(selections),
        pd.concat(memberships, ignore_index=True),
    )


def evaluate(predictions, training, config):
    scores, differences = [], []
    for stage, data in [("oof", predictions), ("training_repeated_memberships", training)]:
        for block, f in data.groupby("block"):
            names = [c for c in f if c.startswith("prediction_") and f[c].notna().all()]
            for col in names:
                for label, part in [("pooled", f)] + [
                    (str(k), g) for k, g in f.groupby("outer_fold")
                ]:
                    scores.append(
                        dict(
                            stage=stage,
                            block=block,
                            model=col[11:],
                            fold=label,
                            survey_category="All",
                            prediction_sd=float(part[col].std(ddof=0)),
                            **metrics(part, col),
                        )
                    )
                for category, part in f.groupby("n_survey_category"):
                    scores.append(
                        dict(
                            stage=stage,
                            block=block,
                            model=col[11:],
                            fold="pooled",
                            survey_category=category,
                            prediction_sd=float(part[col].std(ddof=0)),
                            **metrics(part, col),
                        )
                    )
                if stage != "oof" or col[11:] in ["mean", "median"]:
                    continue
                family, view = col[11:].split("__")
                baselines = ["mean", "median", family + "__context", "fixed_ridge__" + view]
                if view == "context_hr_steps":
                    baselines.append(family + "__context_hr")
                for base in dict.fromkeys(baselines):
                    if "prediction_" + base == col:
                        continue
                    differences.append(
                        dict(
                            block=block,
                            model=col[11:],
                            baseline=base,
                            **paired_bootstrap(
                                f,
                                col,
                                "prediction_" + base,
                                config["bootstrap_draws"],
                                config["bootstrap_seed"],
                            ),
                        )
                    )
    return pd.DataFrame(scores), pd.DataFrame(differences)
