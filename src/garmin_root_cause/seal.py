"""Finalize verified outputs once; subsequent calls verify the immutable checkpoint."""

import importlib.metadata
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import artifact_hashes, verify_bundle, write_json
from .preservation import completed
from .privacy import scan


def seal(root):
    root = Path(root)
    if completed(root):
        return json.loads((root / "reports/root_cause_analysis/completion.json").read_text())
    base = root / "private/root_cause_analysis"
    public_root = root / "reports/root_cause_analysis"
    ptr = json.loads((base / "current.json").read_text())
    data, run = root / ptr["data"], root / ptr["run"]
    public = public_root / run.name
    before = json.loads((base / "before.json").read_text())
    assert all(sha256(root / p) == h for p, h in before["files"].items())
    assert all(sha256(root / p) == h for p, h in before["manifests"].items())
    assert "231 passed" in (base / "tests_full.log").read_text()
    assert "All checks passed!" in (base / "ruff.log").read_text()
    verify = json.loads((public / "verification.json").read_text())
    assert verify["validation_replays"] == verify["training_replays"] == 185
    assert not json.loads((public / "failures.json").read_text())
    assert verify["complete_original_to_prediction_traces"] == 1
    nbpath = root / "notebooks/18_root_cause_diagnostics.ipynb"
    nb = json.loads(nbpath.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 8 and all(type(c["execution_count"]) is int for c in cells)
    assert not any(o["output_type"] == "error" for c in cells for o in c["outputs"])
    browser = json.loads((base / "render/browser_audit.json").read_text())
    assert browser["desktop"]["sections"] == 9 and len(browser["desktop"]["images"]) == 7
    assert (
        browser["screenshots"] == 15 and not browser["errors"] and not browser["externalRequests"]
    )
    assert not browser["desktop"]["bodyOverflow"] and not browser["mobile"]["bodyOverflow"]
    sources = list((root / "src/garmin_root_cause").glob("*.py")) + [
        root / "config/root_cause_analysis.json",
        root / "tests/test_root_cause.py",
        public_root / "specification.md",
    ]
    for p in sources:
        dest = run / "checkpoint_source" / p.relative_to(root)
        dest.parent.mkdir(exist_ok=True, parents=True)
        assert not dest.exists()
        shutil.copyfile(p, dest)
    for bundle in [data, run]:
        identity = json.loads((bundle / "identity.json").read_text())
        for name, value in identity["code"].items():
            assert sha256(root / "src/garmin_root_cause" / name) == value
    write_json(
        run / "environment.json",
        dict(
            python=sys.version,
            packages={
                n: importlib.metadata.version(n)
                for n in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow"]
            },
            lockfile_sha256=sha256(root / "uv.lock"),
        ),
    )
    for p in sorted((run / "jobs").glob("*/manifest.json")):
        verify_bundle(p.parent)
    verify_bundle(data)
    write_json(
        run / "manifest.json",
        dict(
            complete=True,
            version=run.name,
            artifact_sha256=artifact_hashes(run),
            job_manifest_sha256={
                str(p.relative_to(run)): sha256(p)
                for p in sorted((run / "jobs").glob("*/manifest.json"))
            },
            data_manifest_sha256=sha256(data / "manifest.json"),
        ),
    )
    write_json(
        public / "manifest.json",
        dict(
            complete=True,
            version=run.name,
            artifact_sha256=artifact_hashes(public),
            private_manifest_sha256=sha256(run / "manifest.json"),
        ),
    )
    presentation = list(public_root.glob("*.md")) + [public_root / "index.html", nbpath]
    result = dict(
        complete=False,
        complete_evidence_supported_scope=True,
        created_at=datetime.now(timezone.utc).isoformat(),
        data=str(data.relative_to(root)),
        run=str(run.relative_to(root)),
        public=str(public.relative_to(root)),
        source_sha256={str(p.relative_to(root)): sha256(p) for p in sources},
        presentation_sha256={str(p.relative_to(root)): sha256(p) for p in presentation},
        data_manifest_sha256=sha256(data / "manifest.json"),
        private_manifest_sha256=sha256(run / "manifest.json"),
        public_manifest_sha256=sha256(public / "manifest.json"),
        blocked=[
            "Prospective recall alignment: verified boundaries/acquisition/availability absent",
            "EMA-history prediction: submission/receipt/version clocks absent",
            "Definitive physical validation: only five supported inactive bins and unverified counter semantics",
        ],
        rows={
            str(p.relative_to(root)): pq.read_metadata(p).num_rows
            for folder in [data, run]
            for p in folder.glob("*.parquet")
        },
        verification={
            **verify,
            "tests_passed": 231,
            "ruff": "passed",
            "executed_notebook_cells": 8,
            "browser_screenshots": 15,
            "privacy": "pending",
        },
    )
    write_json(public_root / "completion.json", result)
    count = scan(root)
    result["verification"].update(privacy="passed", public_files_scanned=count)
    result["complete"] = True
    write_json(public_root / "completion.json", result)
    assert completed(root)
    return result


if __name__ == "__main__":
    result = seal(Path.cwd())
    print(
        json.dumps(
            {k: result[k] for k in ["complete", "data", "run", "public", "verification"]}, indent=2
        )
    )
