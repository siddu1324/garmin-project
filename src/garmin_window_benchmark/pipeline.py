"""Guarded immutable development experiment; public tables contain aggregates only."""

import argparse
import json
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_ema_baseline.linkage import check_development
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
from .information import aggregate_information, run_information
from .models import evaluate, run_models
from .synthetic import positive_control
from .trace import trace_sample


def baseline_agreement(predictions, baseline):
    old = pd.read_parquet(baseline / "out_of_fold_predictions.parquet")
    old = old[old.scope.eq("native") & old.lookback_minutes.eq(60)].set_index(["block", "n_ema_id"])
    counts = 0
    for block, f in predictions.groupby("block"):
        prior = old.loc[block].loc[f.n_ema_id]
        for model in ["mean", "median", "context", "context_hr"] + (
            ["context_hr_steps"] if block == "hr_steps" else []
        ):
            col = "prediction_" + (
                model if model in ["mean", "median"] else "fixed_ridge__" + model
            )
            if not np.allclose(f[col], prior["prediction_" + model], atol=1e-10, rtol=1e-10):
                raise AuditError("Fixed reference fails Notebook 05 reproduction.")
            counts += len(f)
    return {
        "fixed_reference_prediction_values_verified": counts,
        "maximum_allowed_absolute_tolerance": 1e-10,
    }


def run(root, reference):
    root, reference = Path(root), Path(reference)
    config = json.loads((root / "config/window_benchmark.json").read_text())
    freeze = json.loads((root / "private/window_benchmark/prefit_specification.json").read_text())
    for f, h in freeze["files"].items():
        if sha256(root / f) != h:
            raise AuditError("Frozen experiment specification or input changed.")
    input_guard(root, reference, root / "data")
    assignments, _ = frozen_assignments(root, reference)
    baseline, morning, features = [
        root / config[k + "_bundle"] for k in ["baseline", "morning", "feature"]
    ]
    for path in [baseline, morning, features]:
        verify_bundle(path)
    normalized, verified_features = verified_inputs(
        root, json.loads((baseline / "frozen_specification.json").read_text())
    )
    if features != verified_features:
        raise AuditError("Corrected feature provenance mismatch.")
    morning_manifest = json.loads((morning / "manifest.json").read_text())
    if morning_manifest["inputs"]["feature_manifest_sha256"] != sha256(features / "manifest.json"):
        raise AuditError("Morning feature provenance mismatch.")
    inputs = dict(
        frozen=freeze,
        normalized_manifest=sha256(normalized / "manifest.json"),
        uv_lock=sha256(root / "uv.lock"),
        code={
            str(p.relative_to(root)): sha256(p)
            for package in [
                "garmin_window_benchmark",
                "garmin_ema_baseline",
                "garmin_hr_activity",
                "garmin_morning_recall",
                "garmin_preprocessing",
            ]
            for p in sorted((root / "src" / package).glob("*.py"))
            if p.name not in ["report.py", "verify.py"]
        },
        versions={
            p: version(p)
            for p in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "threadpoolctl"]
        },
    )
    identifier = digest_object(inputs)[:16]
    folder = root / "private/window_benchmark/runs"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / identifier
    if destination.exists():
        verify_bundle(destination)
        return destination
    linked = pd.read_parquet(baseline / "linked_development.parquet")
    check_development(linked, assignments, 3706)
    print("Guards verified; auditing source information.", flush=True)
    pairs, density = run_information(morning, features, config)
    trace, trace_result = trace_sample(root, normalized, morning, features, config)
    synthetic = positive_control(json.loads((root / "config/hr_activity.json").read_text()), config)
    print("Independent source trace and synthetic positive control passed.", flush=True)
    predictions, training, audits, folds, selections, membership = run_models(linked, config)
    old_members = pd.read_parquet(baseline / "cohort_membership.parquet")
    for block, f in membership.groupby("block"):
        old = (
            old_members[
                old_members.scope.eq("native")
                & old_members.block.eq(block)
                & old_members.lookback_minutes.eq(60)
            ]
            .set_index("n_ema_id")
            .loc[f.n_ema_id]
        )
        if not np.array_equal(old.included, f.included):
            raise AuditError("Existing cohort membership changed.")
    agreement = baseline_agreement(predictions, baseline)
    scores, differences = evaluate(predictions, training, config)
    tables = aggregate_information(pairs, density)
    composition = []
    for (block, included, category), f in membership.groupby(
        ["block", "included", "n_survey_category"]
    ):
        composition.append(
            dict(
                block=block,
                included=bool(included),
                survey_category=category,
                responses=len(f),
                participants=f.n_participant.nunique(),
                valid_scores=int(f.pss_combo.notna().sum()),
                score_mean=float(f.pss_combo.mean()),
                score_median=float(f.pss_combo.median()),
            )
        )
    tables.update(
        metrics=scores,
        paired_intervals=differences,
        fold_counts=folds,
        selected_settings=selections,
        composition=pd.DataFrame(composition),
    )
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=folder))
    for name, f in [
        ("window_pairs", pairs),
        ("window_density", density),
        ("source_trace", trace),
        ("out_of_fold_predictions", predictions),
        ("training_predictions", training),
        ("membership", membership),
    ]:
        write_parquet(f, build / (name + ".parquet"))
    for name, f in tables.items():
        write_json(build / (name + ".json"), records(f))
    write_json(build / "training_audit.json", audits)
    write_json(build / "source_verification.json", trace_result)
    write_json(build / "synthetic_positive_control.json", synthetic)
    write_json(build / "reference_agreement.json", agreement)
    write_json(build / "frozen_configuration.json", config)
    input_guard(root, reference, root / "data")
    for path in [baseline, morning, features]:
        verify_bundle(path)
    publish_bundle(
        build,
        destination,
        dict(
            version=identifier,
            inputs=inputs,
            development_only=True,
            final_test_access=False,
            prediction_rows=len(predictions),
            training_prediction_rows=len(training),
            membership_rows=len(membership),
            window_pair_rows=len(pairs),
            window_density_rows=len(density),
            outer_learned_fits=len(audits),
            inner_learned_fits=sum(len(a["inner"]) for a in audits),
        ),
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    print("Verified bundle:", run(args.root, args.reference).relative_to(args.root))


if __name__ == "__main__":
    main()
