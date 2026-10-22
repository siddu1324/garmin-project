"""Outcome-free candidate census and deterministic selection; no feature calculation."""

import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_hrv_input.contract import window_input
from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (
    digest_object,
    frozen_assignments,
    input_guard,
    publish_bundle,
    verify_bundle,
    write_json,
)
from garmin_preprocessing.eligibility import support_seconds
from garmin_preprocessing.normalize import write_parquet

ORDER = [
    "first_timestamp",
    "last_timestamp",
    "ema_cutoff",
    "window_end",
    "ema_id",
    "directory",
    "first_logical_row",
    "last_logical_row",
]
COLUMNS = [
    "directory",
    "participant",
    "fold",
    "scenario",
    "ema_id",
    "window_row",
    "step",
    "window_end",
    "ema_cutoff",
    "run_id",
    "first_logical_row",
    "last_logical_row",
    "first_timestamp",
    "last_timestamp",
    "timestamp_span_seconds",
    "supported_seconds",
    "interval_count",
    "allowed_pair_count",
    "support_end",
    "age_last_timestamp_seconds",
    "age_support_end_seconds",
]


def require(condition):
    if not condition:
        raise AuditError("Pilot invariant failed; private details withheld.")


def config(root):
    c = json.loads((root / "config/ibi_pilot.json").read_text())
    require(
        c["policy_name"] == "screen"
        and c["minimum_run_span_seconds"] == 240
        and c["minimum_run_supported_seconds"] == 240
    )
    require(
        0 <= c["maximum_runs_per_hypothesis"] <= 10
        and c["maximum_runs_per_participant_per_hypothesis"] == 1
    )
    require(
        c["scenarios"] == ["interval_end", "interval_start"]
        and c["lookbacks_minutes"] == [5, 15, 30, 60, 120, 360, 720, 1440]
    )
    require(c["tie_breakers"] == ORDER)
    return c


def overlap_components(views):
    """Transitive overlap of source-row ranges; never create a merged feature input."""
    result = views.copy()
    result["component"] = -1
    component = -1
    for _, g in result.groupby(["directory", "scenario"], sort=True):
        cursor = -1
        for idx, row in g.sort_values(
            ["first_logical_row", "last_logical_row"], kind="stable"
        ).iterrows():
            if row.first_logical_row > cursor:
                component += 1
            result.loc[idx, "component"] = component
            cursor = max(cursor, int(row.last_logical_row))
    return result


def choose_runs(views, limit=10):
    if not 0 <= limit <= 10:
        raise ValueError("Pilot cap is ten per hypothesis.")
    grouped = overlap_components(views)
    representatives = grouped.sort_values(ORDER, kind="stable").drop_duplicates("component")
    per_person = representatives.drop_duplicates(["scenario", "directory"])
    selected = per_person.groupby("scenario", sort=True, group_keys=False).head(limit).copy()
    return grouped, selected.sort_values(["scenario", *ORDER], kind="stable").reset_index(drop=True)


def select(root, reference):
    from .windows import corrected_windows

    c = config(root)
    input_guard(root, reference, root / "data")
    assignments, dev = frozen_assignments(root, reference)
    folds = assignments.set_index("participant").fold.to_dict()
    normalized = root / c["normalized_bundle"]
    audited = root / c["eligibility_bundle"]
    verify_bundle(normalized)
    verify_bundle(audited)
    verify_bundle(root / c["input_diagnostics_bundle"])
    policy_cfg = json.loads((root / "config/ibi_eligibility.json").read_text())
    require(json.loads((audited / "frozen_policies.json").read_text()) == policy_cfg)
    policy = next(p for p in policy_cfg["policies"] if p["name"] == "screen")
    inputs = dict(
        config_sha256=sha256(root / "config/ibi_pilot.json"),
        protocol_sha256=sha256(root / "reports/ibi_pilot_protocol.md"),
        normalized_manifest=sha256(normalized / "manifest.json"),
        eligibility_manifest=sha256(audited / "manifest.json"),
        assignments_sha256=sha256(root / "private/frozen_participant_assignments.parquet"),
        policy_sha256=sha256(root / "config/ibi_eligibility.json"),
        code_sha256=sha256(Path(__file__)),
        clock_correction_sha256=sha256(Path(__file__).with_name("windows.py")),
        contract_sha256=sha256(root / "src/garmin_hrv_input/contract.py"),
        environment_sha256=sha256(root / "uv.lock"),
    )
    version = digest_object(inputs)[:16]
    parent = root / "private/ibi_pilot/selection"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / version
    if destination.exists():
        verify_bundle(destination)
        return destination
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    write_json(build / "frozen_selection_protocol.json", c)
    ema = pd.read_parquet(
        normalized / "ema/development_ema.parquet",
        columns=["n_ema_id", "n_participant", "n_fold", "n_cutoff_epoch", "n_invalid_clock"],
    )
    require(set(ema.n_participant) == dev and ema.n_fold.eq(ema.n_participant.map(folds)).all())
    emap = ema.set_index("n_ema_id")
    people = json.loads((normalized / "participant_directories.json").read_text())
    require({p["participant"] for p in people} == dev)
    views = []
    account = []
    support = Counter()
    hydrated = 0
    corrected_count = 0
    window_count = 0
    for number, person in enumerate(people):
        require(person["fold"] == folds[person["participant"]])
        columns = ["n_ema_id", "n_step_index", "n_window_end_epoch", "n_fold", "n_partition"]
        for scenario in c["scenarios"]:
            columns += [
                f"screen__{scenario}__{x}" for x in ["eligible", "longest_sequence_seconds"]
            ]
        windows = pd.read_parquet(
            audited / "windows" / f"{person['directory']}.parquet", columns=columns
        )
        require(
            windows.n_partition.eq("development").all() and windows.n_fold.eq(person["fold"]).all()
        )
        path = normalized / "logical" / person["directory"] / "ibi.parquet"
        frame = pd.read_parquet(path) if path.exists() else None
        if frame is not None:
            require(
                frame.n_participant.eq(person["participant"]).all()
                and frame.n_partition.eq("development").all()
                and frame.n_fold.eq(person["fold"]).all()
                and np.array_equal(frame.n_logical_row.to_numpy(), np.arange(len(frame)))
            )
        windows = corrected_windows(windows, emap, frame, policy, policy_cfg)
        corrected_count += int(windows.historical_window_clock_corrected.sum())
        window_count += len(windows)
        write_parquet(
            windows, build / "corrected_screen_windows" / f"{person['directory']}.parquet"
        )
        local = []
        for si, scenario in enumerate(c["scenarios"]):
            eligible = windows[f"screen__{scenario}__eligible"] & (
                windows[f"screen__{scenario}__longest_sequence_seconds"] >= 240
            )
            for window_row, w in windows.loc[eligible].iterrows():
                require(
                    frame is not None
                    and emap.loc[w.n_ema_id, "n_participant"] == person["participant"]
                )
                cutoff = float(emap.loc[w.n_ema_id, "n_cutoff_epoch"])
                require(w.n_window_end_epoch == cutoff - 60 * w.n_step_index)
                h = window_input(frame, float(w.n_window_end_epoch), policy, scenario)
                hydrated += 1
                for _, run in h.runs.iterrows():
                    if run.timestamp_span_seconds < 240 or run.supported_seconds < 240:
                        continue
                    rec = h.records.loc[h.records.c_run_id == run.run_id]
                    local.append(
                        dict(
                            directory=person["directory"],
                            participant=person["participant"],
                            fold=person["fold"],
                            scenario=scenario,
                            ema_id=w.n_ema_id,
                            window_row=int(window_row),
                            step=int(w.n_step_index),
                            window_end=float(w.n_window_end_epoch),
                            ema_cutoff=cutoff,
                            run_id=int(run.run_id),
                            first_logical_row=int(run.first_logical_row),
                            last_logical_row=int(run.last_logical_row),
                            first_timestamp=float(rec.n_epoch_s.iloc[0]),
                            last_timestamp=float(rec.n_epoch_s.iloc[-1]),
                            timestamp_span_seconds=float(run.timestamp_span_seconds),
                            supported_seconds=float(run.supported_seconds),
                            interval_count=int(run.interval_count),
                            allowed_pair_count=int(run.allowed_pair_count),
                            support_end=float(rec.c_hypothesized_end.max()),
                            age_last_timestamp_seconds=cutoff - float(rec.n_epoch_s.iloc[-1]),
                            age_support_end_seconds=cutoff - float(rec.c_hypothesized_end.max()),
                        )
                    )
        f = pd.DataFrame(local, columns=COLUMNS)
        views.extend(local)
        for si, scenario in enumerate(c["scenarios"]):
            for minutes in c["lookbacks_minutes"]:
                g = f[(f.scenario == scenario) & (f.step <= minutes - 5)]
                counts = g.groupby("ema_id").window_row.nunique()
                for row in ema.loc[ema.n_participant == person["participant"]].itertuples():
                    account.append(
                        dict(
                            ema_id=row.n_ema_id,
                            participant=row.n_participant,
                            fold=int(row.n_fold),
                            scenario=scenario,
                            lookback_minutes=minutes,
                            qualifying_windows=int(counts.get(row.n_ema_id, 0)),
                            invalid_cutoff=bool(row.n_invalid_clock),
                        )
                    )
                if frame is not None and len(g):
                    chosen = np.zeros(len(frame), bool)
                    for row in g.itertuples():
                        chosen[row.first_logical_row : row.last_logical_row + 1] = True
                    support[(scenario, minutes)] += float(
                        support_seconds(
                            frame.n_epoch_s.to_numpy(), frame.n_value.to_numpy() / 1000, chosen, si
                        )
                    )
        if (number + 1) % 20 == 0:
            print(
                f"Candidate census: {number + 1}/{len(people)} development participants", flush=True
            )
    candidates = pd.DataFrame(views, columns=COLUMNS)
    grouped, selected = choose_runs(candidates, c["maximum_runs_per_hypothesis"])
    selected["selection_id"] = [f"pilot_{i:02}" for i in range(len(selected))]
    accounting = pd.DataFrame(account)
    require(len(accounting) == len(ema) * 2 * len(c["lookbacks_minutes"]))
    feasibility = []
    for scenario in c["scenarios"]:
        for minutes in c["lookbacks_minutes"]:
            ledger = accounting[
                (accounting.scenario == scenario) & (accounting.lookback_minutes == minutes)
            ]
            qualifying = ledger[ledger.qualifying_windows > 0]
            g = candidates[(candidates.scenario == scenario) & (candidates.step <= minutes - 5)]
            components = overlap_components(g)
            feasibility.append(
                dict(
                    scenario=scenario,
                    lookback_minutes=minutes,
                    ema_denominator=len(ledger),
                    participant_denominator=len(dev),
                    qualifying_ema=len(qualifying),
                    qualifying_participants=qualifying.participant.nunique(),
                    qualifying_window_instances=int(ledger.qualifying_windows.sum()),
                    qualifying_run_views=len(g),
                    exact_source_ranges=len(
                        g.drop_duplicates(["directory", "first_logical_row", "last_logical_row"])
                    ),
                    overlapping_source_episodes=components.component.nunique(),
                    unique_supported_minutes=support[(scenario, minutes)] / 60,
                    age_support_end_min_seconds=float(g.age_support_end_seconds.min())
                    if len(g)
                    else None,
                    age_support_end_median_seconds=float(g.age_support_end_seconds.median())
                    if len(g)
                    else None,
                    age_support_end_max_seconds=float(g.age_support_end_seconds.max())
                    if len(g)
                    else None,
                )
            )
    write_parquet(grouped, build / "candidate_views.parquet")
    write_parquet(selected, build / "selected_runs.parquet")
    write_parquet(accounting, build / "ema_availability.parquet")
    exact = []
    for row in selected.itertuples():
        g = grouped[
            (grouped.directory == row.directory)
            & (grouped.scenario == row.scenario)
            & (grouped.first_logical_row == row.first_logical_row)
            & (grouped.last_logical_row == row.last_logical_row)
        ].copy()
        g["selection_id"] = row.selection_id
        exact.append(g)
    write_parquet(
        pd.concat(exact, ignore_index=True) if exact else grouped.iloc[:0],
        build / "selected_exact_views.parquet",
    )
    result = dict(
        version=version,
        inputs=inputs,
        feasibility=feasibility,
        ema_rows=len(ema),
        development_participants=len(dev),
        candidate_views=len(candidates),
        corrected_window_clocks=corrected_count,
        window_rows=window_count,
        overlap_components=grouped.component.nunique(),
        hydrated_windows=hydrated,
        selection_counts={s: int((selected.scenario == s).sum()) for s in c["scenarios"]},
        features_calculated=False,
        selection_fixed_before_calculation=True,
        final_test_processed=False,
    )
    write_json(build / "aggregate_selection.json", result)
    input_guard(root, reference, root / "data")
    publish_bundle(
        build,
        destination,
        dict(
            version=version,
            inputs=inputs,
            selection_sha256=sha256(build / "selected_runs.parquet"),
            features_calculated=False,
        ),
    )
    return destination
