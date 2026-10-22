"""Disclosed post-result matched context controls, no new modeling search space."""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, write_json, verify_bundle, artifact_hashes
from .data import data
from .run import job


def run_controls(root):
    root = Path(root)
    primary = (
        root / json.loads((root / "private/binary_improvement/current.json").read_text())["run"]
    )
    assert not (primary / "manifest.json").exists(), "No alteration to a completed checkpoint."
    _, _, cfg = data(root)
    specs = [
        dict(
            name=f"context_control_{method}",
            block="context",
            cohort="all",
            mode="main",
            sampler=method,
            context=True,
            compact=False,
            flags=False,
            permutation=None,
            reuse=None,
        )
        for method in ["weighted", "over", "under", "conditional"]
    ]
    identity = dict(
        primary_identity=sha256(primary / "identity.json"),
        amendment=sha256(root / "reports/binary_improvement/context_control_amendment.md"),
        source=sha256(Path(__file__)),
        config=cfg,
        fitting_source={
            n: sha256(Path(__file__).parent / n)
            for n in ["run.py", "models.py", "sampling.py", "data.py"]
        },
    )
    folder = primary / "controls" / digest_object(identity)[:16]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "jobs").mkdir(exist_ok=True)
    write_json(
        primary / "context_controls.json",
        dict(path=str(folder.relative_to(primary)), identity=identity),
    )
    if (folder / "manifest.json").exists():
        verify_bundle(folder)
        return folder
    if not (folder / "identity.json").exists():
        write_json(folder / "identity.json", identity)
        write_json(folder / "specifications.json", specs)
        (folder / "executed_source.py").write_bytes(Path(__file__).read_bytes())
    jobs = [(t, s, k) for t in ["A", "B"] for s in specs for k in range(5)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(job, root, folder, t, s, k): (t, s["name"], k) for t, s, k in jobs}
        for i, result in enumerate(as_completed(futures), 1):
            print(i, len(jobs), *futures[result], result.result(), flush=True)
    write_json(
        folder / "manifest.json",
        dict(
            complete=True,
            version=folder.name,
            artifact_sha256=artifact_hashes(folder),
            job_manifest_sha256={
                str(p.relative_to(folder)): sha256(p)
                for p in sorted((folder / "jobs").glob("*/manifest.json"))
            },
        ),
    )
    return folder


if __name__ == "__main__":
    print(run_controls(Path.cwd()))
