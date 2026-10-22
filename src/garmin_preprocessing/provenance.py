"""Reject known-invalid window artifacts, including copies renamed on disk."""

from pathlib import Path

from garmin_inventory.privacy import AuditError, sha256

REVOKED_MANIFESTS = {
    "ddeddedfde4e9449776179cbad9e9209e992fd54833e7164e93000e2476e320a",
    "c6e897386d7de6e70e0d7d0aa27b6139488f9f1829274210eaaf35df6aa3faff",
    "f1a95d52b36d29722979969b59e108f50ac9d312a979bcc4c4f68600d368e2fd",
    "571373500a732850c089ec22e0bc64c7fc672c43a4961c447b00683637437db7",
}
REVOKED_VERSIONS = {
    "6747223eb50269ba",
    "bc13763b35972be4",
    "6747223eb50269ba_90911a7ae670",
    "30a0bbe188d56b40",
}


def assert_current_windows(folder, manifest):
    if (
        manifest.get("version") in REVOKED_VERSIONS
        or sha256(Path(folder) / "manifest.json") in REVOKED_MANIFESTS
    ):
        raise AuditError(
            "Superseded int16 window artifact rejected. See reports/window_supersession.json; historical checksums do not establish valid boundaries."
        )
