"""Notebook 09's bounded nested fitting, with explicit Notebook 10 predictor sets."""

import json
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from threadpoolctl import threadpool_limits
from garmin_ema_baseline.models import CONTEXT, feature_columns, metrics, paired_bootstrap
from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import digest_object
from garmin_window_benchmark.models import inner_splits

META = ["n_ema_id", "n_participant", "n_fold", "n_survey_category", "pss_combo"]


def feature_inputs(view):
    blocks = {
        "B_context": [],
        "C_raw_hr": feature_columns(60),
        "D_reference": ["reference_mean_bpm"],
        "E_reference_deviations": [
            "reference_mean_bpm",
            "hr_mean_deviation_bpm",
            "hr_latest_deviation_bpm",
            "hr_sd_bpm__60m",
            "hr_slope_bpm_per_minute__60m",
        ],
    }
    if view not in blocks:
        raise AuditError("Unapproved predictor block.")
    return CONTEXT + blocks[view], ["n_survey_category"]


def build_model(view):
    numeric, categorical = feature_inputs(view)
    return Pipeline(
        [
            (
                "transform",
                ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), numeric),
                        (
                            "category",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            categorical,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            ("ridge", Ridge(alpha=1, solver="svd", fit_intercept=True)),
        ]
    )


def fit_predict(train, valid, view, family, setting, config):
    """Only explicit allowed X columns reach the estimator; never missing-value routing."""
    if set(train.n_participant) & set(valid.n_participant):
        raise AuditError("Participant overlap in model split.")
    numeric, categorical = feature_inputs(view)
    cols = numeric + categorical
    if (
        not np.isfinite(train[numeric + ["pss_combo"]]).all().all()
        or not np.isfinite(valid[numeric]).all().all()
    ):
        raise AuditError("Complete-case benchmark received missing or nonfinite inputs.")
    model = build_model(view)
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


def run_models(eligible, config):
    frame = eligible.sort_values("n_ema_id").copy()
    predictions, training, audits, counts, settings = [], [], [], [], []
    for fold in range(5):
        train, valid = frame[frame.n_fold.ne(fold)], frame[frame.n_fold.eq(fold)]
        splits = inner_splits(train)
        status = "fitted" if len(valid) and len(splits) == 3 else "infeasible_grouped_split"
        counts.append(
            dict(
                fold=fold,
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
        vp["outer_fold"] = fold
        tp["outer_fold"] = fold
        for name in ["mean", "median"]:
            constant = float(getattr(train.pss_combo, name)())
            vp["prediction_" + name] = constant
            tp["prediction_" + name] = constant
        for view in ["B_context", "C_raw_hr", "D_reference", "E_reference_deviations"]:
            for family in ["tuned_ridge", "hgb"]:
                name = family + "__" + view
                setting, inner_mae, inner_audit = tune(train, view, family, config, splits)
                pred, trpred, audit = fit_predict(train, valid, view, family, setting, config)
                vp["prediction_" + name] = pred
                tp["prediction_" + name] = trpred
                audits.append(
                    dict(
                        fold=fold,
                        model=name,
                        setting=setting,
                        inner=inner_audit,
                        training_key_sha256=digest_object(train.n_ema_id.tolist()),
                        validation_key_sha256=digest_object(valid.n_ema_id.tolist()),
                        **audit,
                    )
                )
                settings.append(
                    dict(
                        fold=fold,
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
        training.append(tp)
        print(f"Completed earlier-reference outer fold {fold}", flush=True)
    empty = pd.DataFrame(columns=META + ["outer_fold"])
    return (
        pd.concat(predictions, ignore_index=True) if predictions else empty,
        pd.concat(training, ignore_index=True) if training else empty,
        audits,
        pd.DataFrame(counts),
        pd.DataFrame(settings),
    )


def evaluate(predictions, training, config):
    scores, differences = [], []
    if not len(predictions):
        return pd.DataFrame(), pd.DataFrame()
    for stage, f in [("oof", predictions), ("training_repeated_memberships", training)]:
        for col in [c for c in f if c.startswith("prediction_")]:
            model = col.removeprefix("prediction_")
            for fold, part in [("pooled", f)] + [(str(k), g) for k, g in f.groupby("outer_fold")]:
                scores.append(
                    dict(
                        stage=stage,
                        model=model,
                        fold=fold,
                        survey_category="All",
                        prediction_sd=float(part[col].std(ddof=0)),
                        **metrics(part, col),
                    )
                )
            for category, part in f.groupby("n_survey_category"):
                scores.append(
                    dict(
                        stage=stage,
                        model=model,
                        fold="pooled",
                        survey_category=category,
                        prediction_sd=float(part[col].std(ddof=0)),
                        **metrics(part, col),
                    )
                )
            if stage != "oof" or model in ["mean", "median"]:
                continue
            family, view = model.split("__")
            baselines = ["mean", "median", family + "__B_context"]
            if view == "E_reference_deviations":
                baselines += [family + "__C_raw_hr", family + "__D_reference"]
            for base in dict.fromkeys(baselines):
                if base != model:
                    differences.append(
                        dict(
                            model=model,
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
