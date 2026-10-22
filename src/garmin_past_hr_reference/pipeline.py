"""Immutable development-only reference experiment, frozen before fitting."""

import argparse
import json
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_ema_baseline.linkage import check_development
from garmin_ema_baseline.models import CONTEXT, cohort_masks, feature_columns
from garmin_ema_baseline.pipeline import records, verified_inputs
from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (
    digest_object,
    frozen_assignments,
    input_guard,
    publish_bundle,
    verify_bundle,
    write_json,
)
from garmin_preprocessing.normalize import write_parquet
from .diagnostics import retrospective_diagnostics
from .models import evaluate, run_models
from .reference import META, NEW, build_reference


def link_reference(linked, reference, quality):
    if not reference.n_ema_id.is_unique or set(linked.n_ema_id) != set(reference.n_ema_id):
        raise AuditError("Reference keys are not one-to-one.")
    ref = reference.set_index("n_ema_id").loc[linked.n_ema_id].reset_index()
    f = linked.reset_index(drop=True)
    for col in META:
        if not (ref[col].eq(f[col]) | (ref[col].isna() & f[col].isna())).all():
            raise AuditError("Reference identity/cutoff/fold mismatch.")
    predictors = f[META + ["pss_combo", "context_available"] + CONTEXT + feature_columns(60)].copy()
    for col in NEW:
        predictors[col] = ref[col]
    q = quality.set_index("n_ema_id").loc[f.n_ema_id].reset_index()
    native = cohort_masks(f)[("native", "hr", 60)]
    eligible = native & q.reference_available & np.isfinite(predictors[NEW]).all(axis=1)
    accounting = (
        f[META + ["pss_combo"]]
        .copy()
        .assign(native_60m_hr=native, reference_available=q.reference_available, eligible=eligible)
    )
    reasons = []
    for i, row in f.iterrows():
        why = []
        if pd.isna(row.pss_combo):
            why.append("missing_outcome")
        if not row.context_available:
            why.append("invalid_context")
        if not np.isfinite(row[feature_columns(60)].to_numpy(float)).all():
            why.append("incomplete_current_hr")
        if not q.reference_available.iloc[i]:
            why.extend(str(q.reason.iloc[i]).split(";"))
        reasons.append(";".join(why) or "eligible")
    accounting["reason"] = reasons
    return predictors, accounting


def availability(accounting, quality):
    composition, diagnostics, attrition = [], [], []
    for scope, f in [
        ("all_development", accounting),
        ("native_60m_hr", accounting[accounting.native_60m_hr]),
    ]:
        for label, mask in [("eligible", f.eligible), ("excluded", ~f.eligible)]:
            part = f[mask]
            for category, g in [("All", part)] + list(part.groupby("n_survey_category")):
                y = g.pss_combo.dropna()
                composition.append(
                    dict(
                        scope=scope,
                        group=label,
                        survey_category=category,
                        responses=len(g),
                        participants=g.n_participant.nunique(),
                        valid_scores=len(y),
                        missing_scores=int(g.pss_combo.isna().sum()),
                        score_mean=float(y.mean()),
                        score_median=float(y.median()),
                        score_sd=float(y.std()),
                        score_q25=float(y.quantile(0.25)),
                        score_q75=float(y.quantile(0.75)),
                    )
                )
                q = quality.set_index("n_ema_id").loc[g.n_ema_id]
                row = dict(scope=scope, group=label, survey_category=category, responses=len(q))
                for col in [
                    "occupied_utc_minutes",
                    "occupied_utc_hours",
                    "occupied_minute_fraction",
                    "accepted_readings",
                    "latest_reading_age_seconds",
                    "first_reading_age_seconds",
                    "mean_observed_minute_age_seconds",
                    "observed_span_seconds",
                    "left_gap_seconds",
                    "right_gap_seconds",
                    "maximum_internal_gap_seconds",
                    "maximum_gap_including_edges_seconds",
                ]:
                    for tag, value in [("median", 0.5), ("q25", 0.25), ("q75", 0.75)]:
                        row[col + "_" + tag] = float(q[col].quantile(value))
                diagnostics.append(row)
        for reason, g in f.groupby("reason"):
            attrition.append(
                dict(
                    scope=scope,
                    kind="exclusive_combination",
                    reason=reason,
                    responses=len(g),
                    participants=g.n_participant.nunique(),
                )
            )
        for reason in sorted(set(";".join(f.reason).split(";"))):
            mask = f.reason.str.split(";").map(lambda xs: reason in xs)
            attrition.append(
                dict(
                    scope=scope,
                    kind="overlapping_reason",
                    reason=reason,
                    responses=int(mask.sum()),
                    participants=f.loc[mask, "n_participant"].nunique(),
                )
            )
    return {
        "composition": pd.DataFrame(composition),
        "reference_quality": pd.DataFrame(diagnostics),
        "attrition": pd.DataFrame(attrition),
    }


def run(root, reference_root):
    root, reference_root = Path(root).resolve(), Path(reference_root)
    config = json.loads((root / "config/past_hr_reference.json").read_text())
    freeze = json.loads((root / "private/past_hr_reference/prefit_specification.json").read_text())
    for path, h in freeze["files"].items():
        if sha256(root / path) != h:
            raise AuditError("Frozen specification/input changed.")
    old_config = json.loads((root / "config/window_benchmark.json").read_text())
    for k in [
        "ridge_alphas",
        "hgb_grid",
        "hgb_fixed",
        "inner_splits",
        "bootstrap_draws",
        "bootstrap_seed",
    ]:
        if config[k] != old_config[k]:
            raise AuditError("Notebook 09 model grid changed.")
    input_guard(root, reference_root, root / "data")
    assignments, _ = frozen_assignments(root, reference_root)
    baseline = root / config["baseline_bundle"]
    benchmark = root / config["benchmark_bundle"]
    features = root / config["feature_bundle"]
    for p in [baseline, benchmark, features]:
        verify_bundle(p)
    normalized, current = verified_inputs(
        root, json.loads((baseline / "frozen_specification.json").read_text())
    )
    if current != features:
        raise AuditError("Corrected feature provenance mismatch.")
    inputs = dict(
        frozen=freeze,
        normalized_manifest_sha256=sha256(normalized / "manifest.json"),
        lock_sha256=sha256(root / "uv.lock"),
        code={
            str(p.relative_to(root)): sha256(p)
            for package in [
                "garmin_past_hr_reference",
                "garmin_window_benchmark",
                "garmin_ema_baseline",
                "garmin_hr_activity",
                "garmin_preprocessing",
            ]
            for p in sorted((root / "src" / package).glob("*.py"))
            if p.name not in ["verify.py", "report.py"]
        },
        versions={
            p: version(p)
            for p in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "threadpoolctl"]
        },
    )
    identifier = digest_object(inputs)[:16]
    folder = root / "private/past_hr_reference/runs"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / identifier
    if destination.exists():
        verify_bundle(destination)
        return destination
    linked = pd.read_parquet(baseline / "linked_development.parquet")
    check_development(linked, assignments, 3706)
    print("Input guards passed; building earlier unlabeled HR references.", flush=True)
    refs, quality, lineage = build_reference(
        linked[META + feature_columns(60)].copy(), features, config
    )
    check_development(refs, assignments, 3706)
    predictive, accounting = link_reference(linked, refs, quality)
    tables = retrospective_diagnostics(linked, config)
    tables.update(availability(accounting, quality))
    eligible = predictive[accounting.eligible]
    predictions, training, audits, folds, settings = run_models(eligible, config)
    scores, differences = evaluate(predictions, training, config)
    tables.update(
        metrics=scores, paired_intervals=differences, fold_counts=folds, selected_settings=settings
    )
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=folder))
    for name, f in [
        ("reference_features", refs),
        ("reference_quality", quality),
        ("reference_lineage", lineage),
        ("predictive_inputs", predictive),
        ("accounting", accounting),
        ("out_of_fold_predictions", predictions),
        ("training_predictions", training),
    ]:
        write_parquet(f, build / (name + ".parquet"))
    for name, f in tables.items():
        write_json(build / (name + ".json"), records(f))
    write_json(build / "training_audit.json", audits)
    write_json(build / "frozen_configuration.json", config)
    input_guard(root, reference_root, root / "data")
    for p in [baseline, features, benchmark]:
        verify_bundle(p)
    publish_bundle(
        build,
        destination,
        dict(
            version=identifier,
            inputs=inputs,
            development_only=True,
            final_test_access=False,
            accounted_responses=len(accounting),
            reference_available=int(quality.reference_available.sum()),
            native_hr_responses=int(accounting.native_60m_hr.sum()),
            eligible_responses=len(eligible),
            eligible_participants=eligible.n_participant.nunique(),
            prediction_rows=len(predictions),
            training_prediction_rows=len(training),
            fitted_outer_models=len(audits),
            fitted_inner_models=sum(len(a["inner"]) for a in audits),
            limited_interpretation=bool(
                len(eligible) < 100
                or eligible.n_participant.nunique() < 30
                or folds.validation_participants.min() < 5
            ),
        ),
    )
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--reference", type=Path, required=True)
    a = p.parse_args()
    print("Verified bundle:", run(a.root, a.reference).relative_to(a.root))


if __name__ == "__main__":
    main()
