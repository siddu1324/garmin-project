"""Fail closed if a sealed checkpoint has changed; never overwrite its presentation."""

import json
from pathlib import Path

from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import verify_bundle


def completed(root):
    root = Path(root)
    path = root / "reports/root_cause_analysis/completion.json"
    if not path.exists():
        return False
    seal = json.loads(path.read_text())
    if not seal.get("complete"):
        raise ValueError("Checkpoint seal incomplete; inspect private verification")
    for group in ["source_sha256", "presentation_sha256"]:
        if not all(sha256(root / p) == h for p, h in seal[group].items()):
            raise ValueError("Sealed checkpoint changed; create an explicit new version")
    for key in ["data", "run", "public"]:
        verify_bundle(root / seal[key])
    return True
