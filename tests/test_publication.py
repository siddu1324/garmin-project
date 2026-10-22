"""Fail-closed publishing and narrowly scoped maintenance exceptions."""

import hashlib
import json
import pytest
from garmin_publication.paths import allowed
from garmin_publication.preservation import preserved


@pytest.mark.parametrize(
    "name",
    ["private/a.parquet", ".env.example", "reports/a.zip", "../reports/a.csv", "/reports/a.csv"],
)
def test_sensitive_path_cannot_be_allowlisted(tmp_path, name):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/publication_allowlist.json").write_text(json.dumps({"paths": [name]}))
    assert not allowed(tmp_path, name)


def test_csv_requires_exact_review(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/publication_allowlist.json").write_text(
        json.dumps({"paths": ["reports/reviewed.csv"]})
    )
    assert allowed(tmp_path, "reports/reviewed.csv")
    assert not allowed(tmp_path, "reports/unreviewed.csv")


def test_maintenance_cannot_exempt_research_data(tmp_path):
    (tmp_path / "reports/repository_publication").mkdir(parents=True)
    old, new = hashlib.sha256(b"old").hexdigest(), hashlib.sha256(b"new").hexdigest()
    names = ["README.md", "reports/metrics.json"]
    for name in names:
        (tmp_path / name).write_bytes(b"new")
    (tmp_path / "reports/repository_publication/maintenance.json").write_text(
        json.dumps({"changes": {n: {"before_sha256": old, "after_sha256": new} for n in names}})
    )
    assert preserved(tmp_path, "README.md", old)
    assert not preserved(tmp_path, "reports/metrics.json", old)
    (tmp_path / "README.md").write_bytes(b"unexpected")
    assert not preserved(tmp_path, "README.md", old)


def test_blob_audit_rejects_private_table_columns():
    from garmin_publication.audit import inspect_blob

    with pytest.raises(AssertionError, match="Private table columns"):
        inspect_blob("reports/example.csv", b"n_participant,value\nSYNTHETIC,1\n", lambda _: False)


def test_blob_audit_rejects_known_private_text():
    from garmin_publication.audit import inspect_blob

    with pytest.raises(AssertionError, match="Known private token"):
        inspect_blob(
            "reports/example.md", b"SYNTHETIC_PRIVATE", lambda text: "SYNTHETIC_PRIVATE" in text
        )


def test_blob_audit_rejects_credential_pattern():
    from garmin_publication.audit import inspect_blob

    synthetic = ("ghp_" + "a" * 40).encode()
    with pytest.raises(AssertionError, match="Credential pattern"):
        inspect_blob("reports/example.md", synthetic, lambda _: False)
