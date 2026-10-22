"""Exact reviewed publication paths; no blanket CSV or archive exception."""

import json
from pathlib import Path, PurePosixPath


def allowed(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    if any(
        p.startswith(".env")
        or p in {"private", "data", "outputs", "models", "__pycache__", ".git", ".venv"}
        for p in path.parts
    ):
        return False
    if path.suffix in {".parquet", ".joblib", ".pkl", ".pickle", ".zip", ".xlsx", ".pdf"}:
        return False
    manifest = Path(root) / "config/publication_allowlist.json"
    if not manifest.exists():
        return False
    return name in json.loads(manifest.read_text())["paths"]
