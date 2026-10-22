"""Efficient equivalent known-token scanning, preserving existing aggregate exceptions."""

import json
import re
from pathlib import Path

from garmin_inventory.privacy import AuditError, public_scan_text, validate_notebook_output


def token_matcher(identifiers, names):
    identifiers = [re.escape(str(t)) for t in identifiers if len(str(t)) >= 4]
    names = [re.escape(str(t)) for t in names if len(str(t)) >= 4]
    ids = (
        re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(identifiers) + r")(?![A-Za-z0-9])")
        if identifiers
        else None
    )
    named = re.compile("|".join(names)) if names else None
    return lambda text: bool((ids and ids.search(text)) or (named and named.search(text)))


def scan(root):
    root = Path(root)
    tokens = json.loads((root / "private/privacy_tokens.json").read_text())
    contains = token_matcher(tokens["identifiers"], tokens["names"])
    files = [
        p
        for folder in ["reports", "notebooks", "src", "tests", "config"]
        for p in (root / folder).rglob("*")
        if p.is_file() and p.suffix in {".json", ".md", ".ipynb", ".py", ".svg"}
    ]
    files += [root / p for p in ["README.md", ".gitignore", "pyproject.toml", "uv.lock"]]
    files += list((root / "reports/comprehensive_regression").rglob("*.html"))
    for path in files:
        text = public_scan_text(path)
        if contains(text):
            raise AuditError("Known private token in public artifact; details withheld.")
        if path.suffix == ".ipynb":
            validate_notebook_output(path, json.loads(text))
    return len(files)


if __name__ == "__main__":
    print(json.dumps(dict(public_files_scanned=scan(Path.cwd()), status="passed")))
