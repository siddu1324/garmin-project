"""Training-only loss diagnostics: no held-out outcomes or predictions accepted."""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor

from garmin_ema_baseline.models import CONTEXT, feature_columns, make_model, model_columns
from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import digest_object


def training_frame(cohort, baseline_path, heldout_fold):
    """Filter keys BEFORE requesting targets; Arrow excludes held-out fold targets."""
    train = cohort.loc[cohort.n_fold.ne(heldout_fold)].sort_values("n_ema_id", kind="stable")
    target = pd.read_parquet(
        baseline_path / "linked_development.parquet",
        columns=["n_ema_id", "pss_combo"],
        filters=[("n_fold", "!=", int(heldout_fold))],
    )
    merged = train.merge(target, on="n_ema_id", how="left", validate="one_to_one")
    if len(merged) != len(train) or merged.pss_combo.isna().any():
        raise AuditError("Training target linkage failed.")
    return merged


def fit_training(train, heldout_fold, name, alpha, config):
    """Only a training frame is accepted; refused if any held-out row leaks in."""
    if (
        train.n_fold.eq(heldout_fold).any()
        or not train.n_partition.eq("development").all()
        or train.n_ema_id.duplicated().any()
        or not train.n_fold.isin(range(5)).all()
        or alpha not in config["alphas"]
        or name not in config["models"]
    ):
        raise AuditError("Training diagnostic scope violation.")
    numeric, categorical = model_columns(name, 5)
    columns = numeric + categorical
    if not np.isfinite(train[numeric]).all().all() or not np.isfinite(train.pss_combo).all():
        raise AuditError("Training diagnostic requires the unchanged complete-case cohort.")
    model = make_model(name, 5).set_params(
        ridge=QuantileRegressor(
            quantile=config["quantile"], alpha=alpha, solver=config["solver"], fit_intercept=True
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train[columns], train.pss_combo)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise AuditError("Quantile diagnostic did not converge; no substitute fit.")
    transform = model.named_steps["transform"]
    regressor = model.named_steps["ridge"]
    matrix = transform.transform(train[columns])
    prediction = model.predict(train[columns])
    replay = matrix @ regressor.coef_ + regressor.intercept_
    scaler = transform.named_transformers_["numeric"]
    if (
        not np.allclose(prediction, replay, atol=1e-12)
        or not np.allclose(scaler.mean_, train[numeric].mean(), atol=1e-12)
        or not np.allclose(scaler.var_, train[numeric].var(ddof=0), atol=1e-12)
    ):
        raise AuditError("Training scaling or arithmetic replay differs.")
    y = train.pss_combo.to_numpy(float)
    median = float(np.median(y))
    mae = float(np.abs(y - prediction).mean())
    dummy_mae = float(np.abs(y - median).mean())
    penalty = float(alpha * np.abs(regressor.coef_).sum())
    objective = 0.5 * mae + penalty
    if not np.isfinite(prediction).all() or objective > 0.5 * dummy_mae + 1e-7:
        raise AuditError("Optimized objective exceeds feasible median constant.")
    tags = {"heldout_fold": int(heldout_fold), "model": name, "alpha": float(alpha)}
    stats = {
        **tags,
        "training_responses": len(train),
        "training_participants": train.n_participant.nunique(),
        "converged": True,
        "solver_iterations": int(regressor.n_iter_),
        "warning_categories": sorted({w.category.__name__ for w in caught}),
        "coefficient_count": len(regressor.coef_),
        "nonzero_coefficients": int(
            np.sum(np.abs(regressor.coef_) > config["coefficient_zero_tolerance"])
        ),
        "design_columns_with_intercept": matrix.shape[1] + 1,
        "design_rank_with_intercept": int(
            np.linalg.matrix_rank(np.column_stack([np.ones(len(matrix)), matrix]))
        ),
        "constant_transformed_columns": int(np.sum(np.ptp(matrix, axis=0) < 1e-12)),
        "training_prediction_sd": float(np.std(prediction)),
        "training_prediction_min": float(prediction.min()),
        "training_prediction_max": float(prediction.max()),
        "training_unique_predictions_8dp": int(len(np.unique(np.round(prediction, 8)))),
        "training_median": median,
        "training_below_median_fraction": float(np.mean(y < median)),
        "training_at_median_fraction": float(np.mean(y == median)),
        "training_above_median_fraction": float(np.mean(y > median)),
        "training_mae": mae,
        "training_median_dummy_mae": dummy_mae,
        "training_mae_difference_vs_median": mae - dummy_mae,
        "training_pinball_loss": 0.5 * mae,
        "l1_penalty": penalty,
        "training_objective": objective,
        "training_median_dummy_objective": 0.5 * dummy_mae,
    }
    parameters = {
        **tags,
        "training_keys_sha256": digest_object(train.n_ema_id.tolist()),
        "input_columns": columns,
        "numeric_columns": numeric,
        "terms": transform.get_feature_names_out().tolist(),
        "training_mean": scaler.mean_.tolist(),
        "training_scale": scaler.scale_.tolist(),
        "training_categories": transform.named_transformers_["category"].categories_[0].tolist()
        if categorical
        else [],
        "coefficients": regressor.coef_.tolist(),
        "intercept": float(regressor.intercept_),
    }
    predictions = train[["n_ema_id", "n_participant", "n_fold", "pss_combo"]].copy()
    for key, value in tags.items():
        predictions[key] = value
    predictions["training_prediction"] = prediction
    return stats, parameters, predictions


def run_training(baseline, followup, config):
    membership = pd.read_parquet(
        followup / "membership.parquet",
        columns=["n_ema_id", "policy", "scope", "lookback_minutes", "included"],
    )
    chosen = membership.loc[
        membership.policy.eq("A")
        & membership.scope.eq("native")
        & membership.lookback_minutes.eq(5)
        & membership.included,
        ["n_ema_id"],
    ]
    columns = [
        "n_ema_id",
        "n_participant",
        "n_partition",
        "n_fold",
        "n_survey_category",
        *CONTEXT,
        *feature_columns(5),
    ]
    covariates = pd.read_parquet(baseline / "linked_development.parquet", columns=columns)
    cohort = chosen.merge(covariates, on="n_ema_id", how="left", validate="one_to_one")
    if len(cohort) != 1395 or cohort.n_participant.nunique() != 117:
        raise AuditError("Existing five-minute Policy A cohort changed.")
    saved = json.loads((followup / "training_audit.json").read_text())
    saved = {
        a["fold"]: a
        for a in saved
        if a["stage"] == "loss_only" and a["scope"] == "native" and a["lookback_minutes"] == 5
    }
    stats, parameters, predictions, inspection = [], [], [], []
    for fold in range(5):
        train = training_frame(cohort, baseline, fold)
        if set(train.n_participant) & set(cohort.loc[cohort.n_fold.eq(fold), "n_participant"]):
            raise AuditError("Frozen participant separation failed.")
        for name in config["models"]:
            old = next(m for m in saved[fold]["models"] if m["model"] == "q_" + name)
            inspection.append(
                {
                    "heldout_fold": fold,
                    "model": name,
                    "saved_nonzero_coefficients": old["nonzero_coefficients_at_1e_minus_8"],
                    "saved_prediction_sd": old["validation_prediction_sd"],
                    "saved_unique_predictions_8dp": old["validation_unique_predictions_8dp"],
                    "saved_training_objective": old["training_pinball_plus_penalty"],
                    "saved_converged": old["converged"],
                }
            )
            for alpha in config["alphas"]:
                stat, audit, pred = fit_training(train, fold, name, alpha, config)
                if alpha == 0.01:
                    for key in ["training_mean", "training_scale", "coefficients", "intercept"]:
                        if not np.allclose(audit[key], old[key], rtol=1e-9, atol=1e-9):
                            raise AuditError("Alpha .01 does not reproduce Notebook 06 parameters.")
                    if (
                        audit["training_keys_sha256"] != saved[fold]["training_keys_sha256"]
                        or audit["training_categories"] != old["training_categories"]
                        or not np.isclose(
                            stat["training_objective"],
                            old["training_pinball_plus_penalty"],
                            atol=1e-10,
                        )
                    ):
                        raise AuditError("Alpha .01 training provenance/objective differs.")
                stats.append(stat)
                parameters.append(audit)
                predictions.append(pred)
    return (
        pd.DataFrame(stats),
        parameters,
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(inspection),
    )
