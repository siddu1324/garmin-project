"""Checkpoint-local aggregate authorization without editing earlier validators."""

import json
from pathlib import Path

from garmin_binary_6hour.privacy import validate_15
from garmin_comprehensive_regression.privacy import token_matcher
from garmin_inventory.privacy import AuditError, public_scan_text, validate_notebook_output


def scan(root):
    root = Path(root)
    tokens = json.loads((root / "private/privacy_tokens.json").read_text())
    contains = token_matcher(tokens["identifiers"], tokens["names"])
    files = [
        p
        for folder in ["reports", "notebooks", "src", "tests", "config"]
        for p in (root / folder).rglob("*")
        if p.is_file() and p.suffix in [".json", ".md", ".ipynb", ".py", ".svg"]
    ]
    files += [root / p for p in ["README.md", ".gitignore", "pyproject.toml", "uv.lock"]]
    files += list((root / "reports/binary_improvement").rglob("*.html"))
    files += list((root / "reports/temporal_personalization").rglob("*.html"))
    for p in files:
        content = public_scan_text(p)
        if contains(content):
            raise AuditError("Known private token in public artifact; details withheld.")
        if p.suffix == ".ipynb":
            nb = json.loads(content)
            if p.name in [
                "15_binary_pss_6hour_experiments.ipynb",
                "16_binary_stress_improvement.ipynb",
                "17_temporal_personalized_stress.ipynb",
            ]:
                validate_15(nb)
            else:
                validate_notebook_output(p, nb)
    return len(files)


if __name__ == "__main__":
    print(json.dumps(dict(status="passed", public_files_scanned=scan(Path.cwd()))))
