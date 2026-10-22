"""Immutable Notebook 07 pipeline; scoped to retrospective coverage and training diagnostics."""

import argparse
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

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
from .evidence import inspect_sources
from .recall import EMA_COLUMNS, boundaries, coverage, summaries
from .training import run_training


def run(root, reference):
    root, reference = Path(root), Path(reference)
    config_path = root / "config/recall_diagnostics.json"
    spec_path = root / "reports/recall_alignment_specification.md"
    config = json.loads(config_path.read_text())
    frozen = json.loads((root / "private/recall_diagnostics/prefit_specification.json").read_text())
    baseline, followup = root / config["baseline_bundle"], root / config["followup_bundle"]
    for key, path in [
        ("config_sha256", config_path),
        ("specification_sha256", spec_path),
        ("baseline_manifest_sha256", baseline / "manifest.json"),
        ("followup_manifest_sha256", followup / "manifest.json"),
    ]:
        if frozen[key] != sha256(path):
            raise AuditError("Frozen Notebook 07 specification or parent changed.")
    input_guard(root, reference, root / "data")
    assignments, _ = frozen_assignments(root, reference)
    prior = verify_bundle(baseline)
    verify_bundle(followup)
    normalized, features = verified_inputs(
        root, json.loads((baseline / "frozen_specification.json").read_text())
    )
    if features != root / config["feature_bundle"]:
        raise AuditError("Unexpected feature provenance.")
    for key, path in [
        ("assignments_sha256", root / "private/frozen_participant_assignments.parquet"),
        ("supersession_sha256", root / "reports/window_supersession.json"),
    ]:
        if sha256(path) != prior["inputs"][key]:
            raise AuditError("Assignments or corrected provenance changed.")
    inputs = {
        **frozen,
        "feature_manifest": sha256(features / "manifest.json"),
        "normalized_manifest": sha256(normalized / "manifest.json"),
        "freeze_record_sha256": sha256(
            root / "private/recall_diagnostics/prefit_specification.json"
        ),
        "assignments_sha256": prior["inputs"]["assignments_sha256"],
        "supersession_sha256": prior["inputs"]["supersession_sha256"],
        "uv_lock_sha256": sha256(root / "uv.lock"),
        "code": {
            str(p.relative_to(root)): sha256(p)
            for folder in [
                "src/garmin_recall_diagnostics",
                "src/garmin_hr_activity",
                "src/garmin_ema_baseline",
            ]
            for p in sorted((root / folder).glob("*.py"))
            if p.name not in ["report.py", "verify.py"]
        },
        "versions": {
            p: version(p) for p in ["scikit-learn", "scipy", "numpy", "pandas", "pyarrow", "pypdf"]
        },
    }
    identifier = digest_object(inputs)[:16]
    parent = root / "private/recall_diagnostics/runs"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / identifier
    if destination.exists():
        verify_bundle(destination)
        return destination
    ema = pd.read_parquet(normalized / "ema/development_ema.parquet", columns=EMA_COLUMNS)
    check_development(ema, assignments, 3706)
    evidence = inspect_sources(root, ema)
    boundary = boundaries(ema, config)
    candidates = coverage(
        boundary, features, json.loads((root / "config/hr_activity.json").read_text())
    )
    fixed = pd.read_parquet(followup / "verified_hr_support.parquet")
    aggregate = summaries(boundary, candidates, fixed)
    stats, parameters, predictions, inspection = run_training(baseline, followup, config)
    aggregate.update(training_diagnostics=stats, saved_model_inspection=inspection)
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    for name, frame in [
        ("boundaries", boundary),
        ("candidate_coverage", candidates),
        ("training_predictions", predictions),
    ]:
        write_parquet(frame, build / (name + ".parquet"))
    for name, frame in aggregate.items():
        write_json(build / (name + ".json"), records(frame))
    write_json(build / "source_evidence.json", evidence)
    write_json(build / "training_parameters.json", parameters)
    write_json(build / "frozen_configuration.json", config)
    input_guard(root, reference, root / "data")
    verify_bundle(baseline)
    verify_bundle(followup)
    publish_bundle(
        build,
        destination,
        {
            "version": identifier,
            "inputs": inputs,
            "development_only": True,
            "accounted_development_emas": 3706,
            "boundary_rows": len(boundary),
            "candidate_rows": len(candidates),
            "training_prediction_rows": len(predictions),
            "training_fits": len(stats),
            "new_validation_predictions": 0,
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
    print("Verified Notebook 07 bundle:", run(args.root, args.reference).relative_to(args.root))


if __name__ == "__main__":
    main()
