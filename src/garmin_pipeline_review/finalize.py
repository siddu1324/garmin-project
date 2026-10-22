"""Verify the executed notebook and source-only ZIP, then seal this review checkpoint."""

import hashlib
import json
import re
import zipfile
from pathlib import Path
from garmin_inventory.privacy import sha256, public_scan_text
from garmin_comprehensive_regression.privacy import token_matcher
from garmin_preprocessing.common import write_json
from .privacy import scan
from .verify import verify
from .package import package, allowed


def finalize(root):
    root = Path(root)
    public = root / "reports/complete_pipeline_review"
    private = root / "private/pipeline_review"
    seal = public / "completion.json"
    if seal.exists():
        old = json.loads(seal.read_text())
        assert old["complete"] and all(sha256(root / p) == h for p, h in old["sha256"].items())
        return old
    checks = verify(root)
    nbpath = root / "notebooks/20_complete_pipeline_for_review.ipynb"
    nb = json.loads(nbpath.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(type(c["execution_count"]) is int for c in cells)
    assert not any(o["output_type"] == "error" for c in cells for o in c["outputs"])
    tests = (private / "tests.log").read_text()
    m = re.findall(r"(\d+) passed", tests)
    assert m and "FAILED" not in tests
    assert "All checks passed!" in (private / "ruff.log").read_text()
    archive = package(root)
    tokens = json.loads((root / "private/privacy_tokens.json").read_text())
    contains = token_matcher(tokens["identifiers"], tokens["names"])
    with zipfile.ZipFile(root / archive["path"]) as z:
        manifest = json.loads(z.read("SOURCE_MANIFEST.json"))
        for name in z.namelist():
            assert allowed(name)
            text = z.read(name).decode()
            if name.endswith(".ipynb"):
                text = public_scan_text(nbpath)
            assert not contains(text), "Private token in archive; details withheld"
            if name in manifest:
                assert hashlib.sha256(z.read(name)).hexdigest() == manifest[name]
    checks.update(
        tests_passed=int(m[-1]),
        ruff="passed",
        executed_code_cells=len(cells),
        privacy_files=scan(root),
        source_zip_entries=archive["entries"],
        archive_sha256=archive["sha256"],
    )
    paths = (
        list((root / "src/garmin_pipeline_review").glob("*.py"))
        + list(public.rglob("*"))
        + [
            root / "config/complete_pipeline_review.json",
            root / "tests/test_pipeline_review.py",
            nbpath,
        ]
    )
    result = dict(
        complete=True,
        verification=checks,
        notebook=str(nbpath.relative_to(root)),
        archive=archive["path"],
        sha256={str(p.relative_to(root)): sha256(p) for p in paths if p.is_file()},
        private_predictions="Ignored deterministic replays only; no new fits or searches",
        final_test_evaluated=False,
    )
    write_json(seal, result)
    return result


if __name__ == "__main__":
    r = finalize(Path.cwd())
    print(
        json.dumps(
            dict(
                complete=r["complete"],
                notebook=r["notebook"],
                archive=r["archive"],
                verification=r["verification"],
            ),
            indent=2,
        )
    )
