"""Independent scalar feature/window checks, Ridge replay and cluster-bootstrap verification."""

import json
import math
import statistics
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from garmin_ema_baseline.linkage import check_development
from garmin_ema_baseline.models import CONTEXT, feature_columns
from garmin_ema_baseline.pipeline import verified_inputs
from garmin_ema_baseline.verify import independent_prediction
from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import digest_object, frozen_assignments, verify_bundle
from .data import WINDOWS, columns_for
from .experiment import KEYS, contrasts, window_set


def require(condition, message):
    if not condition:
        raise AuditError(message)


def scalar_features(t, v):
    out = [np.nan, np.nan, v[-1] if len(v) else np.nan, np.nan]
    if len(v) >= 2 and t[-1] - t[0] >= 1:
        out[:2] = [statistics.mean(v), statistics.stdev(v)]
    if len(v) >= 3 and t[-1] - t[0] >= 60:
        x = [(a - t[0]) / 60 for a in t]
        mx, mv = statistics.mean(x), statistics.mean(v)
        out[3] = sum((a - mx) * (b - mv) for a, b in zip(x, v)) / sum((a - mx) ** 2 for a in x)
    return out


def scalar_metrics(frame, name):
    error = frame["prediction_" + name].to_numpy() - frame.pss_combo.to_numpy()
    ae, squared = np.abs(error), error**2
    groups = frame.n_participant.to_numpy()
    people = sorted(set(groups))
    return {
        "mae": float(sum(ae) / len(ae)),
        "rmse": float(math.sqrt(sum(squared) / len(squared))),
        "participant_mae": statistics.mean(float(ae[groups == p].mean()) for p in people),
        "participant_rmse": statistics.mean(
            float(np.sqrt(squared[groups == p].mean())) for p in people
        ),
    }


def bootstrap_replay(frame, model, baseline, config):
    error = np.abs(frame["prediction_" + model].to_numpy() - frame.pss_combo.to_numpy()) - np.abs(
        frame["prediction_" + baseline].to_numpy() - frame.pss_combo.to_numpy()
    )
    labels = frame.n_participant.to_numpy()
    groups = [error[labels == name] for name in sorted(set(labels))]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    means = sums / counts
    rng = np.random.default_rng(config["bootstrap_seed"])
    pooled, person = [], []
    for _ in range(config["bootstrap_draws"]):
        draw = rng.integers(0, len(groups), size=len(groups))
        pooled.append(float(sums[draw].sum() / counts[draw].sum()))
        person.append(float(means[draw].mean()))
    result = {
        "mae_difference": float(error.mean()),
        "participant_mae_difference": float(means.mean()),
    }
    for label, values in [("mae", pooled), ("participant_mae", person)]:
        result[label + "_difference_low"], result[label + "_difference_high"] = np.percentile(
            values, [2.5, 97.5]
        )
    return result


def verify(root, bundle, reference):
    manifest = verify_bundle(bundle)
    config = json.loads((bundle / "frozen_configuration.json").read_text())
    baseline, recall, features = [
        root / config[k + "_bundle"] for k in ["baseline", "recall", "feature"]
    ]
    for label, parent in [("baseline", baseline), ("recall", recall), ("feature", features)]:
        verify_bundle(parent)
        require(
            sha256(parent / "manifest.json") == manifest["inputs"][label + "_manifest_sha256"],
            "Parent manifest changed.",
        )
    normalized, current_features = verified_inputs(
        root, json.loads((baseline / "frozen_specification.json").read_text())
    )
    require(current_features == features, "Unexpected corrected feature parent.")
    assignments, _ = frozen_assignments(root, reference)
    f = pd.read_parquet(bundle / "features.parquet")
    q = pd.read_parquet(bundle / "quality.parquet")
    accounting = pd.read_parquet(bundle / "accounting.parquet")
    membership = pd.read_parquet(bundle / "membership.parquet")
    predictions = pd.read_parquet(bundle / "out_of_fold_predictions.parquet")
    raw = pd.read_parquet(
        normalized / "ema/development_ema.parquet",
        columns=["n_ema_id", "n_actual_local", "n_offset_minutes", "SLEEP_WAKETIME"],
        filters=[("n_survey_category", "=", "Morning")],
    ).set_index("n_ema_id")
    original = pd.read_parquet(
        baseline / "linked_development.parquet",
        columns=[
            "n_ema_id",
            "pss_combo",
            *CONTEXT,
            *[c for w in WINDOWS[1:] for c in columns_for(w)],
        ],
        filters=[("n_survey_category", "=", "Morning")],
    ).set_index("n_ema_id")
    for scenario, part in f.groupby("scenario"):
        check_development(part, assignments, 934)
        require(
            part.n_survey_category.eq("Morning").all() and not part.available_at_prediction.any(),
            "Morning/retrospective scope violated.",
        )
        source = original.loc[part.n_ema_id]
        for col in original.columns:
            require(
                np.allclose(part[col], source[col], equal_nan=True, rtol=0, atol=0),
                "Existing outcome/context/fixed features changed.",
            )
        for row in part.itertuples():
            r = raw.loc[row.n_ema_id]
            actual = r.n_actual_local.to_pydatetime()
            utc = (actual - timedelta(minutes=int(r.n_offset_minutes))).replace(tzinfo=timezone.utc)
            cutoff = utc - timedelta(seconds=60 if row.n_start_precision == "minute" else 0)
            require(row.end_epoch == cutoff.timestamp(), "Independent cutoff mismatch.")
            if str(r.SLEEP_WAKETIME).strip():
                clock = datetime.strptime(str(r.SLEEP_WAKETIME).strip(), "%I:%M %p").time()
                wake = (
                    datetime.combine(actual.date(), clock)
                    - timedelta(minutes=int(r.n_offset_minutes))
                ).replace(tzinfo=timezone.utc)
                if scenario == "reported_minute_end_sensitivity":
                    wake += timedelta(seconds=60)
                require(row.start_epoch == wake.timestamp(), "Wake date/offset changed.")
                require(row.candidate_available == (wake < cutoff), "Wake eligibility changed.")
            else:
                require(not row.candidate_available, "Missing wake accepted.")
        a = accounting.loc[accounting.scenario.eq(scenario)]
        check_development(a, assignments, 934)
        aligned = part.set_index("n_ema_id").loc[a.n_ema_id]
        require(
            np.array_equal(a.score_available, np.isfinite(aligned.pss_combo)),
            "Score missingness changed.",
        )
        for interval in WINDOWS:
            require(
                np.array_equal(
                    a["complete_hr__" + interval],
                    np.isfinite(aligned[columns_for(interval)]).all(axis=1),
                ),
                "Availability accounting mismatch.",
            )
    # Reuse no production window generator: compare ledger Boolean masks and scalar statistics.
    indexed = f.set_index(["scenario", "n_ema_id"])
    qcheck = q.loc[q.interval.eq("wake") | q.scenario.eq("reported_minute_start")].copy()
    qcheck["directory"] = [
        indexed.loc[(r.scenario, r.n_ema_id), "directory"] for r in qcheck.itertuples()
    ]
    checked = 0
    for directory, part in qcheck.groupby("directory"):
        ledger = pd.read_parquet(features / "reading_dispositions" / directory / "hr.parquet")
        accepted = ledger.loc[ledger.a_analysis_keep]
        for row in part.itertuples():
            values = indexed.loc[(row.scenario, row.n_ema_id)]
            if row.interval == "wake" and not values.candidate_available:
                require(
                    pd.isna(row.valid_readings) and row.recording_status == "boundary_unavailable",
                    "Undefined window quality became zero.",
                )
                require(
                    values[columns_for("wake")].isna().all(), "Undefined wake produced features."
                )
                continue
            end = float(values.n_cutoff_epoch)
            start = (
                float(values.start_epoch)
                if row.interval == "wake"
                else end - int(row.interval.split("_")[1]) * 60
            )
            require(
                row.start_epoch == start and row.end_epoch == end and start < end,
                "Independent interval boundary mismatch.",
            )
            selected = accepted.loc[accepted.n_epoch_s.ge(start) & accepted.n_epoch_s.lt(end)]
            t, v = selected.n_epoch_s.tolist(), selected.n_value.tolist()
            expected = scalar_features(t, v)
            observed = values[columns_for(row.interval)].to_numpy(float)
            require(
                np.allclose(expected, observed, rtol=1e-10, atol=1e-8, equal_nan=True),
                "Independent HR statistic mismatch.",
            )
            bins = {int((float(x) - start) // 60) for x in t}
            edges = [start, *t, end]
            gap = max(b - a for a, b in zip(edges, edges[1:]))
            internal = max((b - a for a, b in zip(t, t[1:])), default=np.nan)
            recorded = int((ledger.n_epoch_s.ge(start) & ledger.n_epoch_s.lt(end)).sum())
            checks = {
                "recorded_rows": recorded,
                "valid_readings": len(t),
                "occupied_minute_bins": len(bins),
                "possible_minute_bins": math.ceil((end - start) / 60),
                "occupied_bin_fraction": len(bins) / math.ceil((end - start) / 60),
                "maximum_gap_including_edges_seconds": gap,
                "maximum_internal_gap_seconds": internal,
                "latest_age_seconds": end - t[-1] if t else np.nan,
                "left_gap_seconds": t[0] - start if t else np.nan,
                "right_gap_seconds": end - t[-1] if t else np.nan,
                "observed_span_seconds": t[-1] - t[0] if t else np.nan,
            }
            require(
                all(
                    np.isclose(getattr(row, key), value, equal_nan=True)
                    for key, value in checks.items()
                ),
                "Independent quality diagnostic mismatch.",
            )
            checked += 1
    # Verify fixed quality repeats unchanged between the two wake scenarios.
    for interval in WINDOWS[1:]:
        a = (
            q.loc[q.interval.eq(interval) & q.scenario.eq(config["scenarios"][0])]
            .sort_values("n_ema_id")
            .drop(columns="scenario")
            .reset_index(drop=True)
        )
        b = (
            q.loc[q.interval.eq(interval) & q.scenario.eq(config["scenarios"][1])]
            .sort_values("n_ema_id")
            .drop(columns="scenario")
            .reset_index(drop=True)
        )
        require(a.equals(b), "Fixed quality changed between sensitivities.")
    # Reconstruct cohort masks independently, including every original Morning response.
    audit = json.loads((bundle / "training_audit.json").read_text())
    fitted = largest = 0
    for keys, members in membership.groupby(KEYS, sort=True):
        scenario, cohort = keys
        part = f.loc[f.scenario.eq(scenario)]
        expected = (
            np.isfinite(part.pss_combo)
            & part.context_available
            & np.isfinite(part[CONTEXT]).all(axis=1)
            & part.candidate_available
        )
        for window in window_set(cohort):
            for col in columns_for(window):
                expected &= np.isfinite(part[col])
        require(
            len(members) == 934 and not members.n_ema_id.duplicated().any(),
            "Cohort accounting omitted responses.",
        )
        require(
            set(members.loc[members.included, "n_ema_id"]) == set(part.loc[expected, "n_ema_id"]),
            "Independent pairwise/common intersection differs.",
        )
        selected = part.loc[expected].sort_values("n_ema_id", kind="stable")
        pred = predictions.loc[predictions.scenario.eq(scenario) & predictions.cohort.eq(cohort)]
        require(not pred.n_ema_id.duplicated().any(), "Multiple OOF predictions per comparison.")
        for fold in range(5):
            train, valid = (
                selected.loc[selected.n_fold.ne(fold)],
                selected.loc[selected.n_fold.eq(fold)],
            )
            if (
                not len(valid)
                or len(train) < config["minimum_training_responses"]
                or train.n_participant.nunique() < config["minimum_training_participants"]
            ):
                require(not pred.n_fold.eq(fold).any(), "Infeasible fold has predictions.")
                continue
            observed = pred.set_index("n_ema_id").loc[valid.n_ema_id]
            require(not set(train.n_participant) & set(valid.n_participant), "Participant leakage.")
            require(np.array_equal(observed.pss_combo, valid.pss_combo), "OOF outcomes changed.")
            record = next(
                a
                for a in audit
                if (a["scenario"], a["cohort"], a["fold"]) == (scenario, cohort, fold)
            )
            require(
                record["training_keys_sha256"] == digest_object(train.n_ema_id.tolist())
                and record["validation_keys_sha256"] == digest_object(valid.n_ema_id.tolist()),
                "Model training/validation membership differs.",
            )
            for name in ["mean", "median", "context", *window_set(cohort)]:
                kind = name if name in ["mean", "median", "context"] else "context_hr"

                def project(frame):
                    out = frame[
                        ["n_participant", "n_survey_category", "pss_combo", *CONTEXT]
                    ].copy()
                    if kind == "context_hr":
                        for source, target in zip(columns_for(name), feature_columns(5)):
                            out[target] = frame[source]
                    return out

                t, v = project(train), project(valid)
                replay = independent_prediction(t, v, kind, 5)
                error = float(np.max(np.abs(replay - observed["prediction_" + name].to_numpy())))
                require(error < 1e-8, "Independent Ridge/dummy prediction differs.")
                largest = max(largest, error)
                fitted += 1
                saved = next(a for a in record["models"] if a["model"] == name)
                if name not in ["mean", "median"]:
                    numeric = CONTEXT + (feature_columns(5) if kind == "context_hr" else [])
                    require(
                        saved["input_columns"] == numeric + ["n_survey_category"]
                        and saved["numeric_columns"] == numeric,
                        "Non-allowlisted predictor used.",
                    )
                    require(
                        saved["training_categories"] == ["Morning"], "Unexpected context category."
                    )
                    require(
                        np.allclose(saved["training_mean"], t[numeric].mean()),
                        "Training-only mean differs.",
                    )
                    scale = t[numeric].std(ddof=0).to_numpy()
                    scale[scale == 0] = 1
                    require(
                        np.allclose(saved["training_scale"], scale), "Training-only scale differs."
                    )
    scores = pd.read_json(bundle / "metrics.json")
    for row in scores.itertuples():
        part = predictions.loc[
            predictions.scenario.eq(row.scenario) & predictions.cohort.eq(row.cohort)
        ]
        if str(row.fold) != "pooled":
            part = part.loc[part.n_fold.eq(int(row.fold))]
        require(
            len(part) == row.responses and part.n_participant.nunique() == row.participants,
            "Metric counts differ.",
        )
        require(
            all(
                np.isclose(value, getattr(row, name), atol=1e-10)
                for name, value in scalar_metrics(part, row.model).items()
            ),
            "Independent metric replay differs.",
        )
    intervals = pd.read_json(bundle / "paired_intervals.json")
    for row in intervals.itertuples():
        part = predictions.loc[
            predictions.scenario.eq(row.scenario) & predictions.cohort.eq(row.cohort)
        ]
        require((row.model, row.baseline) in contrasts(row.cohort), "Unspecified contrast.")
        expected = bootstrap_replay(part, row.model, row.baseline, config)
        require(
            all(
                np.isclose(value, getattr(row, name), atol=1e-10)
                for name, value in expected.items()
            ),
            "Independent paired cluster-bootstrap replay differs.",
        )
    return {
        "morning_responses_per_scenario": 934,
        "scenarios": 2,
        "features_and_quality_independently_checked": checked,
        "all_existing_outcomes_context_fixed_features_unchanged": True,
        "independently_verified_models": fitted,
        "maximum_independent_prediction_error": largest,
        "metric_rows_verified": len(scores),
        "paired_bootstrap_contrasts_verified": len(intervals),
        "original_participant_folds_preserved": True,
        "matched_training_and_validation_membership": True,
        "quality_and_wake_clock_excluded_from_predictors": True,
        "final_test_participants_sealed": 32,
    }
