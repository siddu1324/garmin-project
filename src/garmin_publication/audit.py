"""Audit exact Git blobs before upload. Reports contain counts, never offending values."""

import csv
import io
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

from garmin_comprehensive_regression.privacy import token_matcher
from garmin_inventory.privacy import public_scan_text, validate_notebook_output, git_guard
from .paths import allowed

SECRET = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})\b"
)
PRIVATE_COLUMNS = {
    "n_participant",
    "participant_id",
    "n_ema_id",
    "rsp_id",
    "mbl_cod",
    "n_source_row",
    "n_cutoff_epoch",
    "actual_start_local",
}


def inspect_blob(name, raw, contains):
    if name.endswith(".png"):
        assert raw.startswith(b"\x89PNG\r\n\x1a\n"), "Unexpected binary artifact"
        return
    text = raw.decode("utf-8")
    proxy = SimpleNamespace(name=Path(name).name, read_text=lambda: text)
    assert not contains(public_scan_text(proxy)), "Known private token found; details withheld"
    assert not SECRET.search(text), "Credential pattern found; details withheld"
    if name.endswith(".ipynb"):
        nb = json.loads(text)
        validate_notebook_output(Path(name), nb)
        for cell in nb["cells"]:
            assert not any(o.get("output_type") == "error" for o in cell.get("outputs", []))
    if name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        assert not PRIVATE_COLUMNS.intersection(reader.fieldnames or []), "Private table columns"
    if name.endswith(".json"):
        json.loads(text)


def run(root, staged=True):
    root = Path(root)
    tokens = json.loads((root / "private/privacy_tokens.json").read_text())
    contains = token_matcher(tokens["identifiers"], tokens["names"])
    allow = json.loads((root / "config/publication_allowlist.json").read_text())["paths"]
    if staged:
        git_guard(root)
        paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        paths = [p for p in paths if p]
    else:
        paths = [p for p in allow if (root / p).is_file()]
    total = 0
    for name in paths:
        assert allowed(root, name), "Unreviewed publication path"
        assert not (root / name).is_symlink(), "Symlinks are not publication inputs"
        raw = (
            subprocess.check_output(["git", "show", ":" + name], cwd=root)
            if staged
            else (root / name).read_bytes()
        )
        if staged:
            assert raw == (root / name).read_bytes(), "Index differs from inspected working file"
        inspect_blob(name, raw, contains)
        total += len(raw)
    # Inspect all reachable history, not just filenames in the latest commit.
    commits = subprocess.check_output(["git", "rev-list", "HEAD"], cwd=root).decode().splitlines()
    seen = set()
    for commit in commits:
        tree = (
            subprocess.check_output(["git", "ls-tree", "-r", "-z", commit], cwd=root)
            .decode()
            .split("\0")
        )
        for item in filter(None, tree):
            info, name = item.split("\t", 1)
            mode, kind, oid = info.split()
            assert kind == "blob" and mode != "120000" and allowed(root, name), (
                "Sensitive/unreviewed path in history; stop"
            )
            if oid not in seen:
                inspect_blob(
                    name,
                    subprocess.check_output(["git", "cat-file", "blob", oid], cwd=root),
                    contains,
                )
                seen.add(oid)
    return dict(
        files=len(paths),
        bytes=total,
        history_commits=len(commits),
        history_blobs=len(seen),
        exact_index_blobs=staged,
        privacy="passed",
        credentials="passed",
        final_test_accessed=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--working", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(Path.cwd(), staged=not args.working), indent=2))
