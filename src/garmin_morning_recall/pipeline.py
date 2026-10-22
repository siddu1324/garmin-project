"""Guarded immutable retrospective Morning experiment; no final-test access."""

import argparse
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

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
from .data import availability, build_features
from .experiment import describe_cohorts, evaluate, run_models


def run(root, reference):
    root, reference = Path(root), Path(reference)
    config_path = root / "config/morning_recall.json"
    config = json.loads(config_path.read_text())
    frozen = json.loads((root / "private/morning_recall/prefit_specification.json").read_text())
    baseline, recall, features = [
        root / config[k + "_bundle"] for k in ["baseline", "recall", "feature"]
    ]
    files = {
        "config_sha256": config_path,
        "specification_sha256": root / "reports/morning_recall_specification.md",
        "baseline_manifest_sha256": baseline / "manifest.json",
        "recall_manifest_sha256": recall / "manifest.json",
        "feature_manifest_sha256": features / "manifest.json",
        "hr_rules_sha256": root / "config/hr_activity.json",
    }
    if any(sha256(path) != frozen[key] for key, path in files.items()):
        raise AuditError("Frozen Morning specification or input changed.")
    input_guard(root, reference, root / "data")
    assignments, _ = frozen_assignments(root, reference)
    prior, parent = verify_bundle(baseline), verify_bundle(recall)
    normalized, verified_features = verified_inputs(
        root, json.loads((baseline / "frozen_specification.json").read_text())
    )
    if (
        verified_features != features
        or parent["inputs"]["feature_manifest"] != sha256(features / "manifest.json")
        or parent["inputs"]["baseline_manifest_sha256"] != sha256(baseline / "manifest.json")
    ):
        raise AuditError("Recall and corrected feature provenance differ.")
    for key, path in [
        ("assignments_sha256", root / "private/frozen_participant_assignments.parquet"),
        ("supersession_sha256", root / "reports/window_supersession.json"),
    ]:
        if sha256(path) != prior["inputs"][key] or parent["inputs"][key] != prior["inputs"][key]:
            raise AuditError("Frozen assignment or window provenance changed.")
    inputs = {
        **frozen,
        "normalized_manifest_sha256": sha256(normalized / "manifest.json"),
        "assignments_sha256": prior["inputs"]["assignments_sha256"],
        "supersession_sha256": prior["inputs"]["supersession_sha256"],
        "freeze_record_sha256": sha256(root / "private/morning_recall/prefit_specification.json"),
        "uv_lock_sha256": sha256(root / "uv.lock"),
        "code": {
            str(p.relative_to(root)): sha256(p)
            for folder in [
                "src/garmin_morning_recall",
                "src/garmin_hr_activity",
                "src/garmin_ema_baseline",
            ]
            for p in sorted((root / folder).glob("*.py"))
            if p.name not in ["report.py", "verify.py"]
        },
        "versions": {
            p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "pyarrow"]
        },
    }
    identifier = digest_object(inputs)[:16]
    folder = root / "private/morning_recall/runs"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / identifier
    if destination.exists():
        verify_bundle(destination)
        return destination
    f, quality, accounting, lineage = build_features(
        baseline,
        recall,
        features,
        config,
        json.loads((root / "config/hr_activity.json").read_text()),
    )
    for _, part in f.groupby("scenario"):
        check_development(part, assignments, 934)
        if not part.n_survey_category.eq("Morning").all() or part.available_at_prediction.any():
            raise AuditError("Retrospective Morning scope violated.")
    membership, predictions, cohorts, folds, audits = run_models(f, config)
    scores, differences = evaluate(predictions, config)
    summaries = describe_cohorts(membership, quality)
    summaries.update(
        availability=availability(f, quality),
        cohorts=cohorts,
        fold_counts=folds,
        metrics=scores,
        paired_intervals=differences,
    )
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=folder))
    for name, frame in [
        ("features", f),
        ("quality", quality),
        ("accounting", accounting),
        ("lineage", lineage),
        ("membership", membership),
        ("out_of_fold_predictions", predictions),
    ]:
        write_parquet(frame, build / (name + ".parquet"))
    for name, frame in summaries.items():
        write_json(build / (name + ".json"), records(frame))
    write_json(build / "training_audit.json", audits)
    write_json(build / "frozen_configuration.json", config)
    input_guard(root, reference, root / "data")
    verify_bundle(recall)
    verify_bundle(baseline)
    publish_bundle(
        build,
        destination,
        {
            "version": identifier,
            "inputs": inputs,
            "retrospective_only": True,
            "development_only": True,
            "morning_emas_accounted": 934,
            "feature_rows": len(f),
            "quality_rows": len(quality),
            "membership_rows": len(membership),
            "prediction_rows": len(predictions),
            "fitted_cohort_folds": len(audits),
            "fitted_models": sum(len(a["models"]) for a in audits),
        },
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--reference", type=Path, default=os.environ.get("GARMIN_REFERENCE_PROJECT")
    )
    args = parser.parse_args()
    if args.reference is None:
        parser.error("Pass --reference or set GARMIN_REFERENCE_PROJECT.")
    print(
        "Verified retrospective Morning bundle:",
        run(args.root, args.reference).relative_to(args.root),
    )


if __name__ == "__main__":
    main()
