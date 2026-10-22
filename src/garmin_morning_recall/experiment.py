"""Matched original-fold Ridge comparisons; retrospective wake is never a predictor."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from garmin_ema_baseline.models import (
    CONTEXT,
    feature_columns,
    make_model,
    model_columns,
    paired_bootstrap,
)
from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import digest_object
from .data import WINDOWS, columns_for

KEYS = ["scenario", "cohort"]


def window_set(cohort):
    if cohort == "common":
        return WINDOWS
    interval = cohort.removeprefix("pair_")
    if cohort != "pair_" + interval or interval not in WINDOWS[1:]:
        raise AuditError("Unspecified comparison cohort.")
    return ["wake", interval]


def cohort_membership(features, config):
    rows = []
    for scenario in config["scenarios"]:
        f = features.loc[features.scenario.eq(scenario)]
        for cohort in ["pair_fixed_60", "pair_fixed_5", "pair_fixed_120", "common"]:
            flags = pd.DataFrame(
                {
                    "missing_outcome": ~np.isfinite(f.pss_combo),
                    "context_unavailable": ~f.context_available
                    | ~np.isfinite(f[CONTEXT]).all(axis=1),
                    "wake_boundary_unavailable": ~f.candidate_available,
                },
                index=f.index,
            )
            for interval in window_set(cohort):
                flags[interval + "_features_unavailable"] = ~np.isfinite(
                    f[columns_for(interval)]
                ).all(axis=1)
            m = f[
                [
                    "n_ema_id",
                    "n_participant",
                    "n_partition",
                    "n_fold",
                    "n_survey_category",
                    "pss_combo",
                ]
            ].copy()
            m["scenario"], m["cohort"] = scenario, cohort
            m["included"] = ~flags.any(axis=1)
            m["reason"] = [
                ";".join(flags.columns[np.asarray(row, bool)]) or "included"
                for row in flags.itertuples(index=False, name=None)
            ]
            rows.append(m)
    return pd.concat(rows, ignore_index=True)


def model_view(frame, interval):
    """Explicitly project features before renaming; no other-window/quality/wake-clock columns."""
    cols = [
        "n_ema_id",
        "n_participant",
        "n_partition",
        "n_fold",
        "n_survey_category",
        "pss_combo",
        *CONTEXT,
    ]
    if interval is not None:
        cols += columns_for(interval)
    result = frame[cols].copy()
    if interval is not None:
        result = result.rename(columns=dict(zip(columns_for(interval), feature_columns(5))))
    return result


def fit_matched(train, valid, intervals, config):
    if (
        set(train.n_participant) & set(valid.n_participant)
        or len(set(valid.n_fold)) != 1
        or train.n_fold.isin(valid.n_fold).any()
        or not train.n_partition.eq("development").all()
        or not valid.n_partition.eq("development").all()
        or not train.n_survey_category.eq("Morning").all()
        or not valid.n_survey_category.eq("Morning").all()
    ):
        raise AuditError("Morning participant/fold separation failed.")
    predictions, audits = {}, []
    names = ["mean", "median", "context", *intervals]
    for name in names:
        if name in ["mean", "median"]:
            model = DummyRegressor(strategy=name).fit(np.zeros((len(train), 1)), train.pss_combo)
            predictions[name] = model.predict(np.zeros((len(valid), 1)))
            audits.append({"model": name, "constant": float(model.constant_.item())})
            continue
        kind = "context" if name == "context" else "context_hr"
        t = model_view(train, None if name == "context" else name)
        v = model_view(valid, None if name == "context" else name)
        numeric, category = model_columns(kind, 5)
        cols = numeric + category
        if not np.isfinite(t[numeric]).all().all() or not np.isfinite(v[numeric]).all().all():
            raise AuditError("Incomplete features cannot enter a matched fit.")
        model = make_model(kind, 5)
        if (
            model.named_steps["ridge"].alpha != config["ridge_alpha"]
            or model.named_steps["ridge"].solver != config["ridge_solver"]
        ):
            raise AuditError("Existing Ridge parameters differ from frozen specification.")
        model.fit(t[cols], t.pss_combo)
        predictions[name] = model.predict(v[cols])
        transform = model.named_steps["transform"]
        scaler = transform.named_transformers_["numeric"]
        categories = transform.named_transformers_["category"].categories_[0].tolist()
        if categories != ["Morning"] or not np.allclose(
            scaler.mean_, t[numeric].mean(), atol=1e-12
        ):
            raise AuditError("Training-only transform verification failed.")
        audits.append(
            {
                "model": name,
                "numeric_columns": numeric,
                "input_columns": cols,
                "source_feature_columns": columns_for(name) if name != "context" else [],
                "training_mean": scaler.mean_.tolist(),
                "training_scale": scaler.scale_.tolist(),
                "training_categories": categories,
                "training_count": int(scaler.n_samples_seen_),
                "coefficients": model.named_steps["ridge"].coef_.tolist(),
                "intercept": float(model.named_steps["ridge"].intercept_),
            }
        )
    if not all(np.isfinite(v).all() for v in predictions.values()):
        raise AuditError("Nonfinite OOF prediction.")
    return predictions, audits


def run_models(features, config):
    membership = cohort_membership(features, config)
    predictions, folds, cohorts, audits = [], [], [], []
    for (scenario, cohort), m in membership.groupby(KEYS, sort=True):
        tags = {"scenario": scenario, "cohort": cohort}
        f = features.loc[
            features.scenario.eq(scenario) & features.n_ema_id.isin(m.loc[m.included, "n_ema_id"])
        ].sort_values("n_ema_id", kind="stable")
        people = [f.loc[f.n_fold.eq(k), "n_participant"].nunique() for k in range(5)]
        limited = (
            len(f) < config["small_cohort_responses"]
            or f.n_participant.nunique() < config["small_cohort_participants"]
            or min(people) < config["small_validation_participants_per_fold"]
        )
        cohorts.append(
            {
                **tags,
                "responses": len(f),
                "participants": f.n_participant.nunique(),
                "excluded_responses": len(m) - len(f),
                "limited_interpretation": bool(limited),
            }
        )
        for fold in range(5):
            train, valid = f.loc[f.n_fold.ne(fold)], f.loc[f.n_fold.eq(fold)]
            status = (
                "no_validation_responses"
                if not len(valid)
                else "insufficient_training"
                if len(train) < config["minimum_training_responses"]
                or train.n_participant.nunique() < config["minimum_training_participants"]
                else "fitted"
            )
            folds.append(
                {
                    **tags,
                    "fold": fold,
                    "training_responses": len(train),
                    "training_participants": train.n_participant.nunique(),
                    "validation_responses": len(valid),
                    "validation_participants": valid.n_participant.nunique(),
                    "excluded_validation_responses": int((m.n_fold.eq(fold) & ~m.included).sum()),
                    "small_validation_fold": valid.n_participant.nunique()
                    < config["small_validation_participants_per_fold"],
                    "status": status,
                }
            )
            if status != "fitted":
                continue
            pred, audit = fit_matched(train, valid, window_set(cohort), config)
            out = valid[["n_ema_id", "n_participant", "n_fold", "pss_combo"]].copy()
            for key, value in tags.items():
                out[key] = value
            for name, values in pred.items():
                out["prediction_" + name] = values
            predictions.append(out)
            audits.append(
                {
                    **tags,
                    "fold": fold,
                    "training_keys_sha256": digest_object(train.n_ema_id.tolist()),
                    "validation_keys_sha256": digest_object(valid.n_ema_id.tolist()),
                    "models": audit,
                }
            )
    empty_columns = ["n_ema_id", "n_participant", "n_fold", "pss_combo", *KEYS] + [
        "prediction_" + n for n in ["mean", "median", "context", *WINDOWS]
    ]
    p = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=empty_columns)
    )
    for row in cohorts:
        sub = p.loc[p.scenario.eq(row["scenario"]) & p.cohort.eq(row["cohort"])]
        row["oof_responses"], row["oof_participants"] = len(sub), sub.n_participant.nunique()
        row["eligible_without_oof"] = row["responses"] - len(sub)
    return membership, p, pd.DataFrame(cohorts), pd.DataFrame(folds), audits


def metrics(frame, prediction):
    error = frame[prediction] - frame.pss_combo
    grouped = pd.DataFrame(
        {"person": frame.n_participant, "absolute": error.abs(), "squared": error**2}
    ).groupby("person")
    return {
        "responses": len(frame),
        "participants": frame.n_participant.nunique(),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "participant_mae": float(grouped.absolute.mean().mean()),
        "participant_rmse": float(np.sqrt(grouped.squared.mean()).mean()),
    }


def contrasts(cohort):
    intervals = window_set(cohort)
    return (
        [("context", "mean"), ("context", "median")]
        + [(name, baseline) for name in intervals for baseline in ["mean", "median", "context"]]
        + [("wake", fixed) for fixed in intervals if fixed != "wake"]
    )


def evaluate(predictions, config):
    scores, differences = [], []
    for keys, frame in predictions.groupby(KEYS, sort=True):
        tags = dict(zip(KEYS, keys))
        for name in ["mean", "median", "context", *window_set(tags["cohort"])]:
            for fold, f in [("pooled", frame)] + [(str(k), v) for k, v in frame.groupby("n_fold")]:
                scores.append(
                    {**tags, "fold": fold, "model": name, **metrics(f, "prediction_" + name)}
                )
        for model, baseline in contrasts(tags["cohort"]):
            differences.append(
                {
                    **tags,
                    "model": model,
                    "baseline": baseline,
                    "responses": len(frame),
                    "participants": frame.n_participant.nunique(),
                    **paired_bootstrap(
                        frame,
                        "prediction_" + model,
                        "prediction_" + baseline,
                        config["bootstrap_draws"],
                        config["bootstrap_seed"],
                    ),
                }
            )
    return pd.DataFrame(scores), pd.DataFrame(differences)


def describe_cohorts(membership, quality):
    composition, distributions, reasons, diagnostics = [], [], [], []
    for keys, m in membership.groupby(KEYS, sort=True):
        tags = dict(zip(KEYS, keys))
        for included in [True, False]:
            part = m.loc[m.included.eq(included)]
            y = part.pss_combo.dropna()
            composition.append(
                {
                    **tags,
                    "group": "included" if included else "excluded",
                    "responses": len(part),
                    "participants": part.n_participant.nunique(),
                    "valid_scores": len(y),
                    "missing_scores": int(part.pss_combo.isna().sum()),
                    "mean": y.mean(),
                    "sd": y.std(ddof=1),
                    "median": y.median(),
                    "q25": y.quantile(0.25),
                    "q75": y.quantile(0.75),
                    "minimum": y.min(),
                    "maximum": y.max(),
                }
            )
            for value, count in part.pss_combo.value_counts(dropna=False).sort_index().items():
                distributions.append(
                    {
                        **tags,
                        "group": "included" if included else "excluded",
                        "score": value,
                        "responses": int(count),
                    }
                )
            q = quality.loc[
                quality.scenario.eq(tags["scenario"])
                & quality.n_ema_id.isin(part.n_ema_id)
                & quality.interval.isin(window_set(tags["cohort"]))
            ]
            for interval, data in q.groupby("interval"):
                row = {
                    **tags,
                    "group": "included" if included else "excluded",
                    "interval": interval,
                    "responses": len(data),
                    "participants": data.n_participant.nunique(),
                }
                for col in [
                    "duration_seconds",
                    "valid_readings",
                    "occupied_bin_fraction",
                    "observed_span_seconds",
                    "latest_age_seconds",
                    "left_gap_seconds",
                    "right_gap_seconds",
                    "maximum_internal_gap_seconds",
                    "maximum_gap_including_edges_seconds",
                ]:
                    row[col + "_available"] = int(data[col].notna().sum())
                    for label, quantile in [("q25", 0.25), ("median", 0.5), ("q75", 0.75)]:
                        row[col + "_" + label] = data[col].quantile(quantile)
                diagnostics.append(row)
        for reason, part in m.groupby("reason", sort=True):
            reasons.append(
                {
                    **tags,
                    "reason_combination": reason,
                    "responses": len(part),
                    "participants": part.n_participant.nunique(),
                }
            )
    return {
        name: pd.DataFrame(value)
        for name, value in dict(
            composition=composition,
            score_distribution=distributions,
            exclusion_combinations=reasons,
            quality_diagnostics=diagnostics,
        ).items()
    }
