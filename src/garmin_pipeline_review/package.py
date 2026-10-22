"""Deterministic allowlisted source archive; no data, estimators or local configuration."""

import json
from pathlib import Path, PurePosixPath
import zipfile
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import write_json


CONFIG_NAMES = [
    "binary_6hour.json",
    "binary_improvement.json",
    "complete_pipeline_review.json",
    "comprehensive_features.json",
    "comprehensive_regression.json",
    "ema_baseline.json",
    "hr_activity.json",
    "hr_support_loss.json",
    "hrv_input_diagnostics.json",
    "hrv_input_diagnostics_corrected.json",
    "ibi_eligibility.json",
    "ibi_pilot.json",
    "measurement_alignment.json",
    "measurement_source.json",
    "morning_recall.json",
    "past_hr_reference.json",
    "pss_classification.json",
    "recall_diagnostics.json",
    "root_cause_analysis.json",
    "temporal_personalization.json",
    "window_benchmark.json",
]


def allowed(path):
    p = PurePosixPath(path)
    if (
        p.is_absolute()
        or ".." in p.parts
        or any(x in p.parts for x in ["private", "data", ".git", "__pycache__", ".venv"])
    ):
        return False
    if p.parts[0] in ["src", "tests"]:
        return p.suffix == ".py"
    if p.parts[0] == "config":
        return len(p.parts) == 2 and p.name in CONFIG_NAMES
    return str(p) in [
        "pyproject.toml",
        "uv.lock",
        ".gitignore",
        "notebooks/20_complete_pipeline_for_review.ipynb",
        "REVIEW_README.md",
        "SOURCE_MANIFEST.json",
    ]


def package(root):
    root = Path(root)
    dest = root / "reports/complete_pipeline_review/garmin_complete_pipeline_source.zip"
    if dest.exists():
        raise ValueError("Source archive exists; preserve it")
    paths = [
        p
        for folder in ["src", "tests", "config"]
        for p in (root / folder).rglob("*")
        if p.is_file() and allowed(str(p.relative_to(root)))
    ]
    paths += [
        root / p
        for p in [
            "pyproject.toml",
            "uv.lock",
            ".gitignore",
            "notebooks/20_complete_pipeline_for_review.ipynb",
        ]
    ]
    contents = {str(p.relative_to(root)): p.read_bytes() for p in paths}
    manifest = {str(p.relative_to(root)): sha256(p) for p in paths}
    contents["SOURCE_MANIFEST.json"] = (
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    ).encode()
    contents["REVIEW_README.md"] = (
        b"""# Complete Garmin + EMA code review\n\nOpen notebooks/20_complete_pipeline_for_review.ipynb for executed aggregate outputs and literal canonical source.\n678 candidates comprise 438 supported observed summaries and 240 provisional PulseOx columns.\nFull logistic fits encode 880 or 881 inputs after training-only preprocessing; these are not 880 physiological features.\n\nThe archive contains unchanged canonical source, tests, versioned configurations and locked dependencies.\nPrivate datasets, splits, participant mappings, estimators, credentials and local configuration are deliberately absent.\nExecute the notebook in the authorized project checkout with its verified immutable private artifacts.\nSource cells verify and import exact canonical modules; raw preprocessing and model-search entrypoints are shown but not rerun.\n\nEnvironment: Python 3.11; uv sync --frozen. Synthetic tests: uv run pytest -q.\nNotebook execution: PYTHONPATH=src uv run jupyter nbconvert --to notebook --execute notebooks/20_complete_pipeline_for_review.ipynb --inplace.\nSome original entrypoints reference project-local/private paths; those are preserved exactly, not substituted with invented data.\nSOURCE_MANIFEST.json gives SHA256 for each supplied canonical file. No new search or final-test evaluation was performed.\n"""
    )
    assert all(allowed(n) for n in contents)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None and all(allowed(n) for n in z.namelist())
        import hashlib

        for name, h in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == h
    result = dict(
        path=str(dest.relative_to(root)),
        sha256=sha256(dest),
        entries=len(contents),
        bytes=dest.stat().st_size,
        private_data_included=False,
        canonical_files_verified=len(manifest),
    )
    write_json(root / "reports/complete_pipeline_review/archive_verification.json", result)
    return result
