"""Aggregate-only checkpoint authorization without changing previous validators."""

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
        if p.is_file() and p.suffix in [".json", ".md", ".ipynb", ".py", ".svg", ".html", ".csv"]
    ]
    files += [root / p for p in ["README.md", ".gitignore", "pyproject.toml", "uv.lock"]]
    approved = {
        "15_binary_pss_6hour_experiments.ipynb",
        "16_binary_stress_improvement.ipynb",
        "17_temporal_personalized_stress.ipynb",
        "18_root_cause_diagnostics.ipynb",
        "19_measurement_alignment.ipynb",
        "20_complete_pipeline_for_review.ipynb",
    }
    for p in files:
        content = public_scan_text(p)
        if contains(content):
            raise AuditError("Known private token in public artifact; details withheld")
        if p.suffix == ".ipynb":
            if p.name in approved:
                validate_15(json.loads(content))
            else:
                validate_notebook_output(p, json.loads(content))
    return len(files)


if __name__ == "__main__":
    print(json.dumps(dict(status="passed", files=scan(Path.cwd()))))
