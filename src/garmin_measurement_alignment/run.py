"""Execute all evidence-supported Path A diagnostics without fitting real models."""

import json
from pathlib import Path
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, write_json, verify_bundle
from .diagnostics import run as diagnostics


def run(root):
    root = Path(root)
    cfg = json.loads((root / "config/measurement_alignment.json").read_text())
    if (root / cfg["metadata_inbox"]).exists() and any((root / cfg["metadata_inbox"]).iterdir()):
        raise ValueError(
            "Provider inbox is nonempty: use reviewed import; do not silently run missing-evidence path"
        )
    identity = dict(
        config=cfg,
        code={
            p.name: sha256(p)
            for p in (root / "src/garmin_measurement_alignment").glob("*.py")
            if p.name in ["diagnostics.py", "run.py"]
        },
        specification_sha256=sha256(root / "reports/measurement_alignment/specification.md"),
        input_manifests={
            k: sha256(root / cfg[k] / "manifest.json")
            for k in ["normalized_bundle", "feature_bundle", "development_bundle", "ledger_bundle"]
        },
    )
    version = digest_object(identity)[:16]
    base = root / "private/measurement_alignment"
    private = base / "runs" / version
    public = root / "reports/measurement_alignment" / version
    ptr = dict(
        version=version, run=str(private.relative_to(root)), public=str(public.relative_to(root))
    )
    if (private / "diagnostics_complete.json").exists():
        assert json.loads((private / "identity.json").read_text()) == identity
        if (private / "manifest.json").exists():
            verify_bundle(private)
        write_json(base / "current.json", ptr)
        return ptr
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    write_json(private / "identity.json", identity)
    result = diagnostics(root, cfg, private, public)
    write_json(public / "status.json", result)
    write_json(private / "diagnostics_complete.json", result)
    write_json(base / "current.json", ptr)
    return ptr


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd())))
