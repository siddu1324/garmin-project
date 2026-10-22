"""Verify new reference/run artifacts while preserving the prior checkpoint."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_inventory.privacy import AuditError, scan_public, sha256
from garmin_preprocessing.common import (
    artifact_hashes,
    frozen_assignments,
    input_guard,
    verify_bundle,
)
from .access import load_indexed_window
from .diagnostics import require


def verify(root, reference, diagnostics):
    input_guard(root, reference, root / "data")
    baseline = json.loads((root / "private/hrv_input_validation/baseline.json").read_text())
    permitted_changes = {
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "src/garmin_inventory/privacy.py",
    }
    preserved = 0
    for name, digest in baseline["files"].items():
        if name not in permitted_changes:
            require(sha256(root / name) == digest)
            preserved += 1
    cfg = json.loads((root / "config/hrv_input_diagnostics.json").read_text())
    n = root / cfg["normalized_bundle"]
    a = root / cfg["eligibility_bundle"]
    verify_bundle(n)
    verify_bundle(a)
    m = verify_bundle(diagnostics)
    require(m["inputs"]["frozen_policy"] == sha256(root / "config/ibi_eligibility.json"))
    require(m["inputs"]["diagnostic_config"] == sha256(root / "config/hrv_input_diagnostics.json"))
    for name, digest in m["inputs"]["code"].items():
        require(sha256(root / "src/garmin_hrv_input" / name) == digest)
    assignments, dev = frozen_assignments(root, reference)
    folds = assignments.set_index("participant").fold.to_dict()
    people = json.loads((n / "participant_directories.json").read_text())
    require({p["participant"] for p in people} == dev)
    policies = json.loads((root / "config/ibi_eligibility.json").read_text())["policies"]
    counts = dict(
        index_rows=0,
        member_case_rows=0,
        run_rows=0,
        allowed_pairs=0,
        run_pairs=0,
        sampled_hydrations=0,
    )
    candidate_windows = {"interval_end": set(), "interval_start": set()}
    for person in people:
        directory = person["directory"]
        index = pd.read_parquet(diagnostics / "window_index" / f"{directory}.parquet")
        old = pd.read_parquet(
            a / "windows" / f"{directory}.parquet",
            columns=[
                "n_ema_id",
                "n_step_index",
                "n_window_start_epoch",
                "n_window_end_epoch",
                "n_fold",
            ],
        )
        require(index[old.columns].equals(old))
        require(index.n_fold.eq(folds[person["participant"]]).all())
        logical = n / "logical" / directory / "ibi.parquet"
        t = (
            pd.read_parquet(logical, columns=["n_epoch_s"]).n_epoch_s.to_numpy()
            if logical.exists()
            else np.array([], float)
        )
        require(
            np.array_equal(
                index.logical_slice_start, np.searchsorted(t, index.n_window_start_epoch.to_numpy())
            )
        )
        require(
            np.array_equal(
                index.logical_slice_stop, np.searchsorted(t, index.n_window_end_epoch.to_numpy())
            )
        )
        counts["index_rows"] += len(index)
        member_path = diagnostics / "immediate_members" / f"{directory}.parquet"
        if member_path.exists():
            f = pd.read_parquet(member_path)
            require(
                f.n_partition.eq("development").all()
                and f.n_participant.eq(person["participant"]).all()
                and f.n_fold.eq(person["fold"]).all()
            )
            same = (
                (f.n_ema_id == f.n_ema_id.shift())
                & (f.c_policy == f.c_policy.shift())
                & (f.c_scenario == f.c_scenario.shift())
            )
            links = f.c_link_to_previous_original_record
            valid_previous = (
                same
                & f.c_acceptable
                & f.c_acceptable.shift(fill_value=False)
                & f.n_logical_row.diff().eq(1)
                & f.c_run_id.eq(f.c_run_id.shift())
            )
            require((~links | valid_previous).all())
            require((f.c_run_id.eq(-1) == ~f.c_acceptable).all())
            require(
                (
                    ~(
                        same
                        & f.c_acceptable
                        & f.c_acceptable.shift(fill_value=False)
                        & f.c_run_id.eq(f.c_run_id.shift())
                    )
                    | links
                ).all()
            )
            counts["member_case_rows"] += len(f)
            counts["allowed_pairs"] += int(links.sum())
        runs_path = diagnostics / "immediate_runs" / f"{directory}.parquet"
        if runs_path.exists():
            runs = pd.read_parquet(runs_path)
            require(
                (runs.allowed_pair_count == runs.interval_count - 1).all()
                and not runs.real_hrv_authorized.any()
            )
            require(runs.normal_beat_status.eq("unverified").all())
            require(
                (runs.last_logical_row - runs.first_logical_row == runs.interval_count - 1).all()
            )
            counts["run_rows"] += len(runs)
            counts["run_pairs"] += int(runs.allowed_pair_count.sum())
            pilots = runs[
                (runs.c_policy == "screen")
                & runs.frozen_window_eligible
                & (runs.timestamp_span_seconds >= 240)
                & (runs.supported_seconds >= 240)
            ]
            for scenario in candidate_windows:
                candidate_windows[scenario].update(
                    (directory, str(x))
                    for x in pilots.loc[pilots.c_scenario == scenario, "n_ema_id"]
                )
        for row in sorted(set([0, len(index) // 2, len(index) - 1])):
            if row < 0:
                continue
            for scenario in ["interval_end", "interval_start"]:
                w = load_indexed_window(n, diagnostics, directory, row, policies[1], scenario)
                require(
                    w is None
                    or len(w.records)
                    == index.logical_slice_stop.iloc[row] - index.logical_slice_start.iloc[row]
                )
                counts["sampled_hydrations"] += 1
    d = json.loads((diagnostics / "aggregate_results.json").read_text())
    require(
        counts["index_rows"] == d["window_index_rows"]
        and counts["member_case_rows"] == d["immediate_member_case_rows"]
        and counts["run_rows"] == d["immediate_run_rows"]
        and counts["allowed_pairs"] == counts["run_pairs"]
    )
    for row in d["overview"]:
        require(
            row["ema_windows"]
            == row["no_recorded_ibi"] + row["recorded_rejected"] + row["eligible"]
        )
    for row in d["continuity"]:
        bins = [v for k, v in row.items() if k.startswith("span_") and "_to_" in k]
        require(sum(bins) == row["windows"])
    require(len(candidate_windows["interval_end"] & candidate_windows["interval_start"]) == 1)
    private = [p for p in (root / "private/hrv_input_validation").rglob("*") if p.is_file()]
    result = subprocess.run(
        ["git", "check-ignore", "--stdin", "-z"],
        cwd=root,
        input=b"\0".join(str(p.relative_to(root)).encode() for p in private) + b"\0",
        capture_output=True,
    )
    require(len([p for p in result.stdout.split(b"\0") if p]) == len(private))
    tokens = json.loads((root / "private/privacy_tokens.json").read_text())
    scanned = scan_public(root, tokens["identifiers"], tokens["names"])
    public = [
        p
        for folder in ["src", "tests", "reports", "notebooks", "config"]
        for p in (root / folder).rglob("*")
        if p.suffix in {".py", ".md", ".json", ".ipynb"}
    ]
    patterns = [
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        r"gh[pousr]_[A-Za-z0-9]{30,}",
        r"AKIA[0-9A-Z]{16}",
        "/" + "Users/" + r"[^\s/]+/",
    ]
    require(not any(re.search(pattern, p.read_text()) for p in public for pattern in patterns))
    public_bundles = 0
    for path in (root / "reports/hrv_input_validation").glob("*/manifest.json"):
        require(artifact_hashes(path.parent) == json.loads(path.read_text())["artifact_sha256"])
        public_bundles += 1
    require(not subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=root))
    input_guard(root, reference, root / "data")
    return dict(
        version=m["version"],
        **counts,
        prior_checkpoint_files_unchanged=preserved,
        frozen_policies_unchanged=True,
        frozen_assignments_unchanged=True,
        raw_reference_integrity=True,
        prior_bundles_hash_verified=True,
        new_bundle_hash_verified=True,
        public_report_bundles_verified=public_bundles,
        public_text_artifacts_scanned=scanned,
        private_artifacts_ignored=len(private),
        credential_and_absolute_path_scan=True,
        candidate_immediate_windows_in_both_hypotheses=len(
            candidate_windows["interval_end"] & candidate_windows["interval_start"]
        ),
        all_ema_accounted=True,
        real_data_hrv_calculated=False,
        commit_or_push=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=os.environ.get("GARMIN_REFERENCE_PROJECT")
    )
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.reference is None:
            raise AuditError("Set GARMIN_REFERENCE_PROJECT.")
        print(
            json.dumps(
                verify(Path.cwd(), args.reference, args.diagnostics), indent=2, sort_keys=True
            )
        )
    except Exception as exc:
        print(
            str(exc)
            if isinstance(exc, AuditError)
            else f"Input verification failed ({type(exc).__name__}); private details withheld."
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
