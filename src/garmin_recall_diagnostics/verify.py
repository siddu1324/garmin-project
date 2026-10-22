"""Independent datetime/Boolean-mask coverage and training-parameter verification."""

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from garmin_ema_baseline.linkage import check_development
from garmin_ema_baseline.models import CONTEXT, feature_columns
from garmin_ema_baseline.pipeline import verified_inputs
from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import digest_object, frozen_assignments, verify_bundle
from .recall import EMA_COLUMNS


def verify(root, bundle, reference):
    manifest = verify_bundle(bundle)
    config = json.loads((bundle / "frozen_configuration.json").read_text())
    baseline, followup = root / config["baseline_bundle"], root / config["followup_bundle"]
    for label, parent in [
        ("baseline_manifest_sha256", baseline),
        ("followup_manifest_sha256", followup),
    ]:
        verify_bundle(parent)
        if sha256(parent / "manifest.json") != manifest["inputs"][label]:
            raise AuditError("Parent provenance changed.")
    normalized, features = verified_inputs(
        root, json.loads((baseline / "frozen_specification.json").read_text())
    )
    assignments, _ = frozen_assignments(root, reference)
    ema = pd.read_parquet(normalized / "ema/development_ema.parquet", columns=EMA_COLUMNS)
    check_development(ema, assignments, 3706)
    lookup = ema.set_index("n_ema_id")
    boundary = pd.read_parquet(bundle / "boundaries.parquet")
    candidates = pd.read_parquet(bundle / "candidate_coverage.parquet")
    if len(boundary) != 2 * len(ema) or boundary.available_at_prediction.any():
        raise AuditError("Boundary accounting/availability mismatch.")
    for _, f in boundary.groupby("scenario"):
        check_development(f, assignments, 3706)
    checked = 0
    for row in boundary.itertuples():
        source = lookup.loc[row.n_ema_id]
        actual = source.n_actual_local.to_pydatetime()
        actual_utc = (actual - timedelta(minutes=int(source.n_offset_minutes))).replace(
            tzinfo=timezone.utc
        )
        cutoff = actual_utc - timedelta(seconds=60 if source.n_start_precision == "minute" else 0)
        if row.end_epoch != cutoff.timestamp():
            raise AuditError("Independent cutoff reconstruction failed.")
        if row.n_survey_category != "Morning":
            if row.candidate_available or row.boundary_reason != "previous_prompt_log_unavailable":
                raise AuditError("Fabricated previous-prompt boundary.")
            continue
        wake = str(source.SLEEP_WAKETIME).strip()
        if not wake:
            if row.candidate_available:
                raise AuditError("Missing wake accepted.")
            continue
        clock = datetime.strptime(wake, "%I:%M %p").time()
        local = datetime.combine(actual.date(), clock)
        expected = (local - timedelta(minutes=int(source.n_offset_minutes))).replace(
            tzinfo=timezone.utc
        )
        expected += timedelta(seconds=config["wake_scenarios_seconds"][row.scenario])
        if expected.timestamp() != row.start_epoch:
            raise AuditError("Independent wake arithmetic failed.")
        if row.candidate_available != (expected < cutoff and not source.n_invalid_clock):
            raise AuditError("Nonpositive/invalid wake interval accepted.")
        if row.candidate_available and row.duration_seconds != (cutoff - expected).total_seconds():
            raise AuditError("Independent candidate duration failed.")
        checked += 1
    quality_checked = 0
    for directory, part in candidates.groupby("directory"):
        ledger = pd.read_parquet(features / "reading_dispositions" / directory / "hr.parquet")
        for row in part.itertuples():
            inside = ledger.n_epoch_s.ge(row.start_epoch) & ledger.n_epoch_s.lt(row.end_epoch)
            accepted = ledger.loc[inside & ledger.a_analysis_keep, "n_epoch_s"].to_numpy(float)
            grouped = ledger.loc[inside & ledger.a_first]
            bins = {int((float(t) - float(row.start_epoch)) // 60) for t in accepted}
            gaps = np.diff(np.r_[row.start_epoch, accepted, row.end_epoch])
            expected = {
                "valid_readings": len(accepted),
                "recorded_rows": int(grouped.a_group_size.sum()),
                "occupied_minute_bins": len(bins),
                "possible_minute_bins": math.ceil(row.duration_seconds / 60),
                "maximum_gap_including_edges_seconds": float(max(gaps)),
                "observed_span_seconds": float(accepted[-1] - accepted[0])
                if len(accepted)
                else np.nan,
                "latest_age_seconds": float(row.end_epoch - accepted[-1])
                if len(accepted)
                else np.nan,
            }
            if any(not np.isclose(getattr(row, k), v, equal_nan=True) for k, v in expected.items()):
                raise AuditError("Boolean-mask HR coverage verification failed.")
            quality_checked += 1
    # Reconstruct matrices by hand, with no sklearn transform and no held-out outcomes.
    covariates = pd.read_parquet(
        baseline / "linked_development.parquet",
        columns=[
            "n_ema_id",
            "n_participant",
            "n_partition",
            "n_fold",
            "n_survey_category",
            *CONTEXT,
            *feature_columns(5),
        ],
    ).set_index("n_ema_id")
    predictions = pd.read_parquet(bundle / "training_predictions.parquet")
    parameters = json.loads((bundle / "training_parameters.json").read_text())
    diagnostics = pd.read_json(bundle / "training_diagnostics.json")
    largest = 0.0
    for p in parameters:
        fold = p["heldout_fold"]
        pred = predictions.loc[
            predictions.heldout_fold.eq(fold)
            & predictions.model.eq(p["model"])
            & predictions.alpha.eq(p["alpha"])
        ]
        train = covariates.loc[pred.n_ema_id].reset_index()
        if (
            train.n_fold.eq(fold).any()
            or not train.n_partition.eq("development").all()
            or digest_object(train.n_ema_id.tolist()) != p["training_keys_sha256"]
        ):
            raise AuditError("Training-only membership failed.")
        target = pd.read_parquet(
            baseline / "linked_development.parquet",
            columns=["n_ema_id", "pss_combo"],
            filters=[("n_fold", "!=", fold)],
        ).set_index("n_ema_id")
        y = target.loc[train.n_ema_id, "pss_combo"].to_numpy()
        if not np.array_equal(y, pred.pss_combo):
            raise AuditError("Training targets changed.")
        raw = train[p["numeric_columns"]].to_numpy(float)
        mean, scale = raw.mean(axis=0), raw.std(axis=0)
        scale[scale == 0] = 1
        if not np.allclose(mean, p["training_mean"]) or not np.allclose(scale, p["training_scale"]):
            raise AuditError("Independent training moments differ.")
        matrix = (raw - mean) / scale
        if p["training_categories"]:
            if p["training_categories"] != sorted(train.n_survey_category.unique()):
                raise AuditError("Encoder used categories outside training.")
            matrix = np.column_stack(
                [matrix]
                + [train.n_survey_category.eq(c).to_numpy(float) for c in p["training_categories"]]
            )
        replay = matrix @ p["coefficients"] + p["intercept"]
        error = float(np.max(np.abs(replay - pred.training_prediction)))
        largest = max(largest, error)
        stat = diagnostics.loc[
            diagnostics.heldout_fold.eq(fold)
            & diagnostics.model.eq(p["model"])
            & diagnostics.alpha.eq(p["alpha"])
        ].iloc[0]
        objective = 0.5 * np.abs(y - replay).mean() + p["alpha"] * np.abs(p["coefficients"]).sum()
        if error > 1e-9 or not np.isclose(objective, stat.training_objective, atol=1e-10):
            raise AuditError("Independent training prediction/objective differs.")
    # Inspect saved predictions without reading their outcomes.
    saved = pd.read_parquet(
        followup / "predictions.parquet",
        columns=[
            "stage",
            "policy",
            "scope",
            "lookback_minutes",
            "n_fold",
            "prediction_q_context",
            "prediction_q_hr",
            "prediction_q_context_hr",
        ],
    )
    saved = saved.loc[
        saved.stage.eq("loss_only")
        & saved.policy.eq("A")
        & saved.scope.eq("native")
        & saved.lookback_minutes.eq(5)
    ]
    old_checked = 0
    for _, part in saved.groupby("n_fold"):
        for name in config["models"]:
            values = part["prediction_q_" + name].to_numpy()
            if not np.allclose(values, 6.0, atol=1e-8):
                raise AuditError("Saved constant prediction inspection differs.")
            old_checked += 1
    evidence = json.loads((bundle / "source_evidence.json").read_text())
    for value in evidence["questionnaire_source_hashes"]:
        if sha256(root / "private/recall_diagnostics/questionnaires" / (value + ".pdf")) != value:
            raise AuditError("Questionnaire citation copy changed.")
    return {
        "development_emas_accounted": len(ema),
        "boundary_rows": len(boundary),
        "wake_datetime_checks": checked,
        "candidate_coverage_checks": quality_checked,
        "independently_replayed_training_fits": len(parameters),
        "training_prediction_rows": len(predictions),
        "maximum_prediction_replay_error": largest,
        "saved_five_minute_prediction_groups_checked_without_outcomes": old_checked,
        "new_validation_outcomes_used": False,
        "new_validation_predictions": 0,
        "frozen_assignments_preserved": True,
        "corrected_provenance_verified": True,
        "sealed_final_test_participants": 32,
    }
