"""Verify pilot lineage, privacy, selected-run arithmetic and corrected clock boundaries."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from garmin_inventory.privacy import scan_public, sha256
from garmin_preprocessing.common import frozen_assignments, input_guard, verify_bundle
from garmin_preprocessing.eligibility import evaluate_windows, parameter_array, support_seconds
from .adapters import FEATURES, guard_run, reference_run
from .selection import choose_runs, config, require


def verify(root, reference, selection, calculation, public):
    input_guard(root, reference, root / "data")
    cfg = config(root)
    normalized, old_audit = root / cfg["normalized_bundle"], root / cfg["eligibility_bundle"]
    for folder in [
        normalized,
        old_audit,
        root / cfg["input_diagnostics_bundle"],
        selection,
        calculation,
        public,
    ]:
        verify_bundle(folder)
    baseline = json.loads((root / "private/ibi_pilot/baseline.json").read_text())
    allowed = {"README.md", "src/garmin_inventory/privacy.py", "src/garmin_preprocessing/audit.py"}
    preserved = 0
    for name, digest in baseline["files"].items():
        if name not in allowed:
            require(sha256(root / name) == digest)
            preserved += 1
    assignments, dev = frozen_assignments(root, reference)
    fold = assignments.set_index("participant").fold.to_dict()
    ledger = pd.read_parquet(selection / "ema_availability.parquet")
    require(len(ledger) == 3706 * 16 and set(ledger.participant) == dev)
    require(ledger.fold.eq(ledger.participant.map(fold)).all())
    require(ledger.groupby(["scenario", "lookback_minutes"]).size().eq(3706).all())
    require(not ledger.duplicated(["ema_id", "scenario", "lookback_minutes"]).any())
    candidates = pd.read_parquet(selection / "candidate_views.parquet")
    selected = pd.read_parquet(selection / "selected_runs.parquet")
    _, replay = choose_runs(candidates, cfg["maximum_runs_per_hypothesis"])
    require(replay.equals(selected.drop(columns="selection_id")))
    require(selected.groupby("scenario").size().le(10).all())
    require(not selected.duplicated(["scenario", "participant"]).any())
    require(set(selected.participant) <= dev)
    require(selected.fold.eq(selected.participant.map(fold)).all())
    require(
        candidates.timestamp_span_seconds.ge(240).all()
        and candidates.supported_seconds.ge(240).all()
    )
    require(candidates.age_support_end_seconds.ge(0).all())
    ema = pd.read_parquet(
        normalized / "ema/development_ema.parquet", columns=["n_ema_id", "n_cutoff_epoch"]
    ).set_index("n_ema_id")
    people = json.loads((normalized / "participant_directories.json").read_text())
    census = json.loads((selection / "aggregate_selection.json").read_text())
    policy = next(
        p
        for p in json.loads((root / "config/ibi_eligibility.json").read_text())["policies"]
        if p["name"] == "screen"
    )
    corrected_total = window_total = 0
    # Independently reconcile union coverage, complete grids and all candidate Screen statuses.
    coverage = {(s, m): 0.0 for s in cfg["scenarios"] for m in cfg["lookbacks_minutes"]}
    for person in people:
        directory = person["directory"]
        w = pd.read_parquet(selection / "corrected_screen_windows" / f"{directory}.parquet")
        prior = pd.read_parquet(
            old_audit / "windows" / f"{directory}.parquet",
            columns=["n_ema_id", "n_step_index", "n_window_end_epoch"],
        )
        require(w[prior.columns[:2]].equals(prior[prior.columns[:2]]))
        require(w.n_partition.eq("development").all() and w.n_fold.eq(person["fold"]).all())
        cutoff = w.n_ema_id.map(ema.n_cutoff_epoch).to_numpy(float)
        expected = cutoff - 60.0 * w.n_step_index.to_numpy(float)
        np.testing.assert_array_equal(w.n_window_end_epoch, expected)
        require(np.all(w.n_window_start_epoch >= cutoff - 86400) and np.all(expected <= cutoff))
        require(w.groupby("n_ema_id").size().eq(1436).all())
        corrected_total += int(w.historical_window_clock_corrected.sum())
        window_total += len(w)
        f = candidates[candidates.directory == directory]
        for row in f.itertuples():
            require(bool(w.loc[row.window_row, f"screen__{row.scenario}__eligible"]))
            require(w.loc[row.window_row, "n_ema_id"] == row.ema_id)
        path = normalized / "logical" / directory / "ibi.parquet"
        if not len(f):
            continue
        source = pd.read_parquet(path, columns=["n_epoch_s", "n_value", "n_logical_row"])
        for si, scenario in enumerate(cfg["scenarios"]):
            for minutes in cfg["lookbacks_minutes"]:
                g = f[(f.scenario == scenario) & (f.step <= minutes - 5)]
                chosen = np.zeros(len(source), bool)
                for row in g.itertuples():
                    chosen[row.first_logical_row : row.last_logical_row + 1] = True
                coverage[(scenario, minutes)] += (
                    support_seconds(
                        source.n_epoch_s.to_numpy(), source.n_value.to_numpy() / 1000, chosen, si
                    )
                    / 60
                )
    require(
        window_total == census["window_rows"]
        and corrected_total == census["corrected_window_clocks"]
    )
    for row in census["feasibility"]:
        g = ledger[
            (ledger.scenario == row["scenario"])
            & (ledger.lookback_minutes == row["lookback_minutes"])
        ]
        passed = g[g.qualifying_windows > 0]
        require(
            len(passed) == row["qualifying_ema"]
            and passed.participant.nunique() == row["qualifying_participants"]
        )
        require(g.qualifying_windows.sum() == row["qualifying_window_instances"])
        np.testing.assert_allclose(
            coverage[(row["scenario"], row["lookback_minutes"])],
            row["unique_supported_minutes"],
            rtol=0,
            atol=1e-8,
        )
    result = pd.read_parquet(calculation / "run_level.parquet")
    members = pd.read_parquet(calculation / "selected_window_members.parquet")
    checks = pd.read_parquet(calculation / "package_comparisons.parquet")
    require(len(result) == len(selected) and len(checks) == 2 * len(selected))
    require(set(result.selection_id) == set(selected.selection_id))
    require(
        result.normal_beat_status.eq("unverified").all()
        and not result.complete_five_minute_hrv.any()
    )
    require(members.n_partition.eq("development").all() and set(members.n_participant) <= dev)
    require(members.n_fold.eq(members.n_participant.map(fold)).all())
    for row in result.itertuples():
        context = members[members.selection_id == row.selection_id]
        rec = context[context.pilot_selected_member]
        source = pd.read_parquet(normalized / "logical" / row.directory / "ibi.parquet")
        # Exact source equality includes timestamps, raw values, provenance and original quality flags.
        pd.testing.assert_frame_equal(
            rec[source.columns].reset_index(drop=True),
            source.iloc[rec.n_logical_row.to_numpy()].reset_index(drop=True),
        )
        span, supported = guard_run(
            rec, row.scenario, screen_eligible=True, development=row.participant in dev
        )
        require(span == row.timestamp_span_seconds and supported == row.supported_seconds)
        require(rec.c_hypothesized_start.ge(row.window_end - 300 - 1e-6).all())
        require(rec.c_hypothesized_end.le(row.window_end + 1e-6).all())
        clock = (
            source[["q_invalid_text_clock", "q_clock_discrepancy", "q_unusual_offset"]]
            .any(axis=1)
            .to_numpy()
        )
        masks, _, _ = evaluate_windows(
            source.n_epoch_s.to_numpy(),
            source.n_value.to_numpy() / 1000,
            source.q_logical_repeated_timestamp.to_numpy(),
            source.q_conflicting_timestamp_value.to_numpy(),
            clock,
            np.array([row.window_end]),
            int(row.scenario == "interval_start"),
            parameter_array(policy),
        )
        require(masks[0] == 0)
        reference_values = reference_run(rec)
        for feature in FEATURES:
            np.testing.assert_allclose(
                getattr(row, feature), reference_values[feature], rtol=1e-10, atol=1e-8
            )
        for check in checks[checks.selection_id == row.selection_id].itertuples():
            if check.status == "compared":
                for feature in FEATURES:
                    value = getattr(check, "package_" + feature)
                    if pd.notna(value):
                        np.testing.assert_allclose(
                            value, reference_values[feature], rtol=1e-10, atol=1e-8
                        )
            else:
                require(
                    check.backend == "neurokit2"
                    and check.reason
                    in {
                        "native_10ms_pair_set_incompatible",
                        "interval_start_not_native_end_semantics",
                    }
                )
    pre = json.loads((calculation / "selection_before_features.json").read_text())
    require(
        pre["selection_manifest"] == sha256(selection / "manifest.json")
        and pre["selection_rows"] == sha256(selection / "selected_runs.parquet")
    )
    tokens = json.loads((root / "private/privacy_tokens.json").read_text())
    public_count = scan_public(root, tokens["identifiers"], tokens["names"])
    local_path = re.compile("/" + "Users/" + r"[^/\s]+/")
    credential = re.compile(
        r"(?:sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
    )
    for folder in ["src", "tests", "config", "reports", "notebooks"]:
        for path in (root / folder).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".ipynb"}:
                text = path.read_text()
                require(not local_path.search(text) and not credential.search(text))
    artifacts = [p for folder in [selection, calculation] for p in folder.rglob("*") if p.is_file()]
    require(
        all(
            subprocess.run(
                ["git", "check-ignore", "-q", str(p.relative_to(root))], cwd=root
            ).returncode
            == 0
            for p in artifacts
        )
    )
    require(
        not subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=root).strip()
    )
    row_counts = {
        str(p.relative_to(root)): pq.ParquetFile(p).metadata.num_rows
        for p in [
            selection / "candidate_views.parquet",
            selection / "selected_runs.parquet",
            selection / "ema_availability.parquet",
            selection / "selected_exact_views.parquet",
            calculation / "run_level.parquet",
            calculation / "package_comparisons.parquet",
            calculation / "selected_window_members.parquet",
        ]
    }
    return dict(
        preserved_prior_files=preserved,
        corrected_window_rows=window_total,
        corrected_clocks=corrected_total,
        all_ema_accounting_rows=len(ledger),
        public_files_scanned=public_count,
        ignored_artifacts_checked=len(artifacts),
        package_compared=int(checks.status.eq("compared").sum()),
        package_blocked=int(checks.status.eq("blocked").sum()),
        input_integrity=True,
        frozen_participant_counts=[125, 32],
        frozen_folds=5,
        final_test_processed=False,
        source_members_exactly_preserved=True,
        deterministic_selection=True,
        selection_precedes_features=True,
        original_synthetic_adapters_unchanged=True,
        row_counts=row_counts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["reference", "selection", "calculation", "public", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    result = verify(Path.cwd(), a.reference, a.selection, a.calculation, a.public)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("Pilot integrity, privacy, clock, selection and arithmetic checks passed.")


if __name__ == "__main__":
    main()
