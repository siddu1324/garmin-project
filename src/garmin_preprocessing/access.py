"""Current eligibility access: reject obsolete clocks before reading any windows."""

import json
from pathlib import Path

import pandas as pd

from garmin_inventory.privacy import AuditError, sha256
from .provenance import assert_current_windows


def load_eligibility_windows(normalized, audited, directory):
    """Caller verifies full bundle hashes once; this loader enforces version and cohort on each read."""
    normalized, audited = Path(normalized), Path(audited)
    manifest = json.loads((audited / "manifest.json").read_text())
    assert_current_windows(audited, manifest)
    if manifest["inputs"]["normalization_manifest_sha256"] != sha256(normalized / "manifest.json"):
        raise AuditError("Window provenance does not match normalized source.")
    people = json.loads((normalized / "participant_directories.json").read_text())
    matches = [p for p in people if p["directory"] == directory]
    if len(matches) != 1:
        raise AuditError("Window directory outside development catalog.")
    f = pd.read_parquet(audited / "windows" / f"{directory}.parquet")
    if not f.n_partition.eq("development").all() or not f.n_fold.eq(matches[0]["fold"]).all():
        raise AuditError("Window cohort/fold mismatch.")
    return f
