"""Preservation, exact-source and aggregate verification for Notebook 20."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from garmin_inventory.privacy import sha256
from garmin_comprehensive.extract import REFERENCE
from garmin_measurement_alignment.seal import completed as prior_completed
from .sources import source_path


def verify(root):
    root = Path(root)
    before = json.loads((root / "private/pipeline_review/before.json").read_text())
    for group in ["files", "manifests"]:
        assert all(sha256(root / p) == h for p, h in before[group].items())
    assert prior_completed(root)
    baseline = json.loads((root / "private/integrity_baseline.json").read_text())
    raw = 0
    for key, folder in [("data", root / "data"), ("reference", REFERENCE)]:
        for name, expected in baseline[key].items():
            actual = (folder / name).stat()
            assert actual.st_size == expected["size"] and actual.st_mtime_ns == expected["mtime_ns"]
            raw += 1
    nb = json.loads((root / "notebooks/20_complete_pipeline_for_review.ipynb").read_text())
    canonical = 0
    for c in nb["cells"]:
        text = "".join(c["source"])
        if c["cell_type"] == "markdown":
            assert all(line.startswith("#") or not line.strip() for line in text.splitlines())
        if text.startswith("%%canonical_source "):
            header, body = text.split("\n", 1)
            module = header.split(" ", 1)[1]
            assert body == source_path(root, module).read_text()
            canonical += 1
    public = root / "reports/complete_pipeline_review"
    transforms = pd.read_csv(public / "transforms.csv")
    assert transforms.candidate_physiology.eq(438).all() and transforms.provisional_used.eq(0).all()
    assert (
        transforms.actual_inputs
        == transforms.retained_physiology
        + transforms.context_numeric
        + transforms.survey_encoded
        + transforms.missing_indicators
    ).all()
    support = pd.read_csv(public / "support.csv")
    counts = (
        support[support.role.eq("validation")]
        .groupby(["checkpoint", "task", "view"])
        .responses.sum()
    )
    assert counts.eq(3689).all()
    predictions = pd.read_parquet(root / "private/pipeline_review/classification_replays.parquet")
    for _, group in predictions.groupby(["checkpoint", "task", "view"]):
        assert group.n_ema_id.is_unique
    audit = json.loads((public / "audit.json").read_text())
    results = json.loads((public / "result_verification.json").read_text())
    assert audit["feature_value_checks"] == 6780 and results["regression_model_replays"] == 70
    assert np.isfinite(pd.read_csv(public / "classification_metrics.csv").roc_auc).all()
    return dict(
        canonical_modules=canonical,
        candidates=678,
        supported_candidates=438,
        provisional_pulseox=240,
        feature_value_checks=audit["feature_value_checks"],
        classifier_replays=audit["classification_model_replays"],
        regressor_replays=results["regression_model_replays"],
        inner_group_separation_checks=audit["inner_separation_checks"],
        prior_files_preserved=len(before["files"]),
        prior_manifests_preserved=len(before["manifests"]),
        raw_reference_metadata_checks=raw,
        new_model_searches=0,
        final_test_evaluated=False,
    )
