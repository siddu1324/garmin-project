"""Checkpoint-local aggregate notebook authorization; preserve prior sealed validators."""

import json
from pathlib import Path

from garmin_comprehensive_regression.privacy import token_matcher
from garmin_inventory.privacy import AuditError, public_scan_text, validate_notebook_output


def validate_15(nb):
    if nb.get("metadata", {}).get("garmin_public_aggregate_notebook") is not True:
        raise AuditError("Notebook 15 requires explicit aggregate metadata.")
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        if type(cell.get("execution_count")) is not int:
            raise AuditError("Notebook 15 is not fully executed.")
        for output in cell.get("outputs", []):
            if output.get("output_type") not in {"display_data", "execute_result", "stream"}:
                raise AuditError("Notebook 15 contains an error or unsupported output.")
            if set(output.get("data", {})) - {
                "text/plain",
                "text/html",
                "text/markdown",
                "image/png",
            }:
                raise AuditError("Unreviewed notebook MIME output.")


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
    files += list((root / "reports/binary_6hour").rglob("*.html"))
    for path in files:
        text = public_scan_text(path)
        if contains(text):
            raise AuditError("Known private token in a public artifact; details withheld.")
        if path.suffix == ".ipynb":
            nb = json.loads(text)
            if path.name == "15_binary_pss_6hour_experiments.ipynb":
                validate_15(nb)
            else:
                validate_notebook_output(path, nb)
    return len(files)


if __name__ == "__main__":
    print(json.dumps(dict(status="passed", public_files_scanned=scan(Path.cwd()))))
