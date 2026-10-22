"""Exact, documented maintenance exceptions; scientific artifacts remain immutable."""

import json
import os
from pathlib import Path

from garmin_inventory.privacy import sha256
from garmin_comprehensive.extract import REFERENCE
from garmin_measurement_alignment.seal import completed

MAINTENANCE = frozenset({"README.md", ".gitignore", "src/garmin_inventory/privacy.py"})


def preserved(root, name, expected):
    root = Path(root)
    actual = sha256(root / name)
    if actual == expected:
        return True
    receipt = root / "reports/repository_publication/maintenance.json"
    if name not in MAINTENANCE or not receipt.exists():
        return False
    record = json.loads(receipt.read_text())["changes"].get(name, {})
    return record.get("before_sha256") == expected and record.get("after_sha256") == actual


def verify_prior(root):
    root = Path(root)
    before = json.loads((root / "private/pipeline_review/before.json").read_text())
    for group in ["files", "manifests"]:
        assert all(preserved(root, p, h) for p, h in before[group].items()), (
            "Prior artifact changed"
        )
    assert completed(root)
    baseline = json.loads((root / "private/integrity_baseline.json").read_text())
    raw = 0
    reference = Path(os.environ.get("GARMIN_REFERENCE_PROJECT", str(REFERENCE)))
    for key, folder in [("data", root / "data"), ("reference", reference)]:
        for name, expected in baseline[key].items():
            actual = (folder / name).stat()
            assert actual.st_size == expected["size"] and actual.st_mtime_ns == expected["mtime_ns"]
            raw += 1
    public = root / "reports/complete_pipeline_review"
    audit = json.loads((public / "audit.json").read_text())
    results = json.loads((public / "result_verification.json").read_text())
    # The caller checks the entire Notebook 20 seal, including these audit records.
    return dict(
        raw_reference_metadata_checks=raw,
        prior_manifests_preserved=len(before["manifests"]),
        feature_value_checks=audit["feature_value_checks"],
        classifier_replays=audit["classification_model_replays"],
        regressor_replays=results["regression_model_replays"],
        inner_group_separation_checks=audit["inner_separation_checks"],
    )
