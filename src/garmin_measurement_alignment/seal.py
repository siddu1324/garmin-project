"""Seal the verified Path A checkpoint; subsequent calls verify, never rewrite."""

import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import pyarrow.parquet as pq
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import artifact_hashes, verify_bundle, write_json
from .privacy import scan
from .verify import verify


def completed(root):
    root = Path(root)
    p = root / "reports/measurement_alignment/completion.json"
    if not p.exists():
        return False
    c = json.loads(p.read_text())
    if not c.get("complete"):
        raise ValueError("Incomplete checkpoint seal")
    for group in ["source_sha256", "presentation_sha256"]:
        assert all(sha256(root / k) == v for k, v in c[group].items())
    verify_bundle(root / c["run"])
    verify_bundle(root / c["public"])
    return True


def seal(root):
    root = Path(root)
    base = root / "private/measurement_alignment"
    public_root = root / "reports/measurement_alignment"
    if completed(root):
        return json.loads((public_root / "completion.json").read_text())
    ptr = json.loads((base / "current.json").read_text())
    run = root / ptr["run"]
    public = root / ptr["public"]
    verification = verify(root)
    tests = (base / "tests_full.log").read_text()
    matches = re.findall(r"(\d+) passed", tests)
    assert matches and int(matches[-1]) == 247 and "FAILED" not in tests
    assert "All checks passed!" in (base / "ruff.log").read_text()
    nbpath = root / "notebooks/19_measurement_alignment.ipynb"
    nb = json.loads(nbpath.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 8 and all(type(c["execution_count"]) is int for c in cells)
    assert not any(o["output_type"] == "error" for c in cells for o in c["outputs"])
    browser = json.loads((base / "render/browser_audit.json").read_text())
    assert (
        browser["desktop"]["sections"] == 8
        and len(browser["desktop"]["images"]) == 7
        and browser["screenshots"] == 14
    )
    assert (
        not browser["errors"]
        and not browser["externalRequests"]
        and not browser["desktop"]["bodyOverflow"]
        and not browser["mobile"]["bodyOverflow"]
    )
    sources = list((root / "src/garmin_measurement_alignment").glob("*.py")) + [
        root / "config/measurement_alignment.json",
        root / "tests/test_measurement_alignment.py",
        public_root / "specification.md",
    ]
    for p in sources:
        dest = run / "checkpoint_source" / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists()
        shutil.copyfile(p, dest)
    write_json(
        run / "environment.json",
        dict(
            python=sys.version,
            lockfile_sha256=sha256(root / "uv.lock"),
            packages={
                n: importlib.metadata.version(n)
                for n in [
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "scipy",
                    "scikit-learn",
                    "matplotlib",
                    "nbconvert",
                ]
            },
        ),
    )
    write_json(
        run / "manifest.json",
        dict(
            complete=True,
            version=ptr["version"],
            artifact_sha256=artifact_hashes(run),
            new_real_fits=0,
        ),
    )
    write_json(
        public / "manifest.json",
        dict(
            complete=True,
            version=ptr["version"],
            artifact_sha256=artifact_hashes(public),
            private_manifest_sha256=sha256(run / "manifest.json"),
        ),
    )
    presentation = list(public_root.glob("*.md")) + [
        public_root / "index.html",
        public_root / "supersession.json",
        nbpath,
    ]
    result = dict(
        complete=False,
        complete_evidence_supported_scope=True,
        execution_path="A_missing_provider_metadata",
        **ptr,
        source_sha256={str(p.relative_to(root)): sha256(p) for p in sources},
        presentation_sha256={str(p.relative_to(root)): sha256(p) for p in presentation},
        verification={
            **verification,
            "tests_passed": int(matches[-1]),
            "ruff": "passed",
            "executed_notebook_cells": len(cells),
            "browser_screenshots": browser["screenshots"],
            "privacy": "pending",
        },
        rows={
            str(p.relative_to(root)): pq.read_metadata(p).num_rows for p in run.glob("*.parquet")
        },
        blocked=[
            "Prospective recall-aligned prediction: approved acquisition/receipt contract and independent boundary logs absent",
            "Previous-answer prediction: exact submitted versions and availability clocks absent",
        ],
        prior_snapshot_sha256=sha256(base / "before.json"),
    )
    write_json(public_root / "completion.json", result)
    count = scan(root)
    result["verification"].update(privacy="passed", public_files_scanned=count)
    result["complete"] = True
    write_json(public_root / "completion.json", result)
    assert completed(root)
    return result


if __name__ == "__main__":
    r = seal(Path.cwd())
    print(
        json.dumps(
            {
                k: r[k]
                for k in ["complete", "execution_path", "run", "public", "verification", "rows"]
            },
            indent=2,
        )
    )
