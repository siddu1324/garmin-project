"""Cache verified review outputs; never invoke original training or extraction entrypoints."""

import json
from pathlib import Path
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import artifact_hashes, write_json
from .audit import execute
from .results import run as results


def identity(root):
    paths = list((root / "src").rglob("*.py")) + [root / "config/complete_pipeline_review.json"]
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def execute_review(root):
    root = Path(root)
    marker = root / "private/pipeline_review/executed.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["source_sha256"] != identity(root):
            raise ValueError("Review code changed: validate a new checkpoint explicitly")
        for group in ["public_sha256", "private_sha256"]:
            assert all(sha256(root / p) == h for p, h in saved[group].items())
        return saved["verification"]
    verification = execute(root)
    verification.update(results(root))
    public = root / "reports/complete_pipeline_review"
    private = root / "private/pipeline_review"
    kept = {
        str((public / p).relative_to(root)): h
        for p, h in artifact_hashes(public).items()
        if not p.endswith((".zip", ".html", ".md"))
    }
    records = {str(p.relative_to(root)): sha256(p) for p in private.glob("*.parquet")}
    write_json(
        marker,
        dict(
            source_sha256=identity(root),
            public_sha256=kept,
            private_sha256=records,
            verification=verification,
        ),
    )
    return verification
