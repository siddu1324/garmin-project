"""Two-phase guarded CLI for selection, then an explicitly authorized small pilot."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_hrv_input.contract import window_input
from garmin_hrv_input.synthetic import package_evidence
from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (
    digest_object,
    frozen_assignments,
    input_guard,
    publish_bundle,
    verify_bundle,
    write_json,
)
from garmin_preprocessing.eligibility import union_seconds
from garmin_preprocessing.normalize import write_parquet
from .adapters import FEATURES, LABEL, guard_run, reference_run, package_comparison
from .selection import config, select, choose_runs, require


def summarize(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(
        count=len(values),
        minimum=float(values.min()) if len(values) else None,
        median=float(np.median(values)) if len(values) else None,
        maximum=float(values.max()) if len(values) else None,
    )


def calculate(root, reference, selection, *, authorized=False):
    if not authorized:
        raise AuditError("Real calculations require the explicit limited-pilot flag.")
    c = config(root)
    sm = verify_bundle(selection)
    input_guard(root, reference, root / "data")
    assignments, dev = frozen_assignments(root, reference)
    normalized = root / c["normalized_bundle"]
    audited = root / c["eligibility_bundle"]
    verify_bundle(normalized)
    verify_bundle(audited)
    require(
        sm["inputs"]["config_sha256"] == sha256(root / "config/ibi_pilot.json")
        and sm["inputs"]["protocol_sha256"] == sha256(root / "reports/ibi_pilot_protocol.md")
        and sm["inputs"]["policy_sha256"] == sha256(root / "config/ibi_eligibility.json")
    )
    require(
        sm["inputs"]["normalized_manifest"] == sha256(normalized / "manifest.json")
        and sm["inputs"]["eligibility_manifest"] == sha256(audited / "manifest.json")
    )
    selected = pd.read_parquet(selection / "selected_runs.parquet")
    require(sha256(selection / "selected_runs.parquet") == sm["selection_sha256"])
    _, reselected = choose_runs(
        pd.read_parquet(selection / "candidate_views.parquet"), c["maximum_runs_per_hypothesis"]
    )
    require(selected.drop(columns=["selection_id"]).equals(reselected))
    require(
        set(selected.participant) <= dev
        and selected.groupby("scenario").size().le(10).all()
        and not selected.duplicated(["scenario", "participant"]).any()
    )
    folds = assignments.set_index("participant").fold.to_dict()
    require(selected.fold.eq(selected.participant.map(folds)).all())
    evidence = package_evidence()
    old = json.loads(
        (
            root / "reports/hrv_input_validation/30a0bbe188d56b40/synthetic_validation.json"
        ).read_text()
    )["package_evidence"]
    require(evidence == old)
    inputs = dict(
        selection_manifest=sha256(selection / "manifest.json"),
        selection_rows=sm["selection_sha256"],
        config_sha256=sha256(root / "config/ibi_pilot.json"),
        package_evidence=evidence,
        environment_sha256=sha256(root / "uv.lock"),
        code={p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("adapters.py")]},
    )
    version = digest_object(inputs)[:16]
    parent = root / "private/ibi_pilot/calculation"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / version
    if destination.exists():
        verify_bundle(destination)
        return destination
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    write_json(
        build / "selection_before_features.json",
        dict(
            selection_manifest=inputs["selection_manifest"], selection_rows=inputs["selection_rows"]
        ),
    )
    policy = next(
        p
        for p in json.loads((root / "config/ibi_eligibility.json").read_text())["policies"]
        if p["name"] == "screen"
    )
    rows = []
    comparisons = []
    members = []
    selected_records = {}
    for s in selected.itertuples():
        frame = pd.read_parquet(normalized / "logical" / s.directory / "ibi.parquet")
        require(
            frame.n_participant.eq(s.participant).all()
            and frame.n_partition.eq("development").all()
            and frame.n_fold.eq(s.fold).all()
        )
        original = pd.read_parquet(
            selection / "corrected_screen_windows" / f"{s.directory}.parquet"
        ).iloc[s.window_row]
        screen_eligible = bool(original[f"screen__{s.scenario}__eligible"])
        require(original.n_ema_id == s.ema_id and original.n_window_end_epoch == s.window_end)
        h = window_input(frame, s.window_end, policy, s.scenario)
        records = h.records[h.records.c_run_id == s.run_id].copy()
        require(
            len(records) == s.interval_count
            and records.n_logical_row.iloc[0] == s.first_logical_row
            and records.n_logical_row.iloc[-1] == s.last_logical_row
        )
        span, supported = guard_run(
            records, s.scenario, screen_eligible=screen_eligible, development=s.participant in dev
        )
        require(span == s.timestamp_span_seconds and supported == s.supported_seconds)
        values = reference_run(records)
        row = s._asdict()
        row.pop("Index", None)
        row.update(
            values,
            label=LABEL,
            partition="development",
            normal_beat_status="unverified",
            complete_five_minute_hrv=False,
            source_normalized_version=normalized.name,
        )
        rows.append(row)
        for backend in ["heartpy", "neurokit2"]:
            comp = package_comparison(records, s.scenario, backend)
            errors = {}
            if comp["status"] == "compared":
                for feature, value in comp["values"].items():
                    if value is not None:
                        np.testing.assert_allclose(
                            value,
                            values[feature],
                            rtol=c["package_comparison_rtol"],
                            atol=c["package_comparison_atol"],
                        )
                        errors[feature] = abs(value - values[feature])
            comparisons.append(
                dict(
                    selection_id=s.selection_id,
                    scenario=s.scenario,
                    backend=backend,
                    status=comp["status"],
                    reason=comp["reason"],
                    **{f"package_{k}": v for k, v in comp["values"].items()},
                    maximum_absolute_error=max(errors.values(), default=None),
                    warning_categories=json.dumps(comp.get("warning_categories", [])),
                )
            )
        # Entire original window retained, including exclusions, without copying feature values to EMA views.
        context = h.records.assign(
            selection_id=s.selection_id,
            pilot_selected_member=h.records.c_run_id.eq(s.run_id),
            pilot_scenario=s.scenario,
            pilot_label=LABEL,
        )
        members.append(context)
        selected_records[(s.directory, s.scenario)] = records
    result = pd.DataFrame(rows) if rows else selected.assign(**{f: np.nan for f in FEATURES})
    checks = pd.DataFrame(
        comparisons,
        columns=[
            "selection_id",
            "scenario",
            "backend",
            "status",
            "reason",
            *[f"package_{k}" for k in FEATURES],
            "maximum_absolute_error",
            "warning_categories",
        ],
    )
    write_parquet(result, build / "run_level.parquet")
    write_parquet(checks, build / "package_comparisons.parquet")
    write_parquet(
        pd.concat(members, ignore_index=True) if members else pd.DataFrame(),
        build / "selected_window_members.parquet",
    )
    feature_summary = []
    duration_summary = []
    package_summary = []
    for scenario in c["scenarios"]:
        g = result[result.scenario == scenario]
        for feature in FEATURES:
            feature_summary.append(
                dict(scenario=scenario, feature=feature, **summarize(g[feature]))
            )
        for field in [
            "timestamp_span_seconds",
            "supported_seconds",
            "interval_count",
            "allowed_pair_count",
            "age_last_timestamp_seconds",
            "age_support_end_seconds",
        ]:
            duration_summary.append(dict(scenario=scenario, measure=field, **summarize(g[field])))
        for backend in ["heartpy", "neurokit2"]:
            z = checks[(checks.scenario == scenario) & (checks.backend == backend)]
            package_summary.append(
                dict(
                    scenario=scenario,
                    backend=backend,
                    compared=int(z.status.eq("compared").sum()),
                    blocked=int(z.status.eq("blocked").sum()),
                    maximum_absolute_error=float(z.maximum_absolute_error.max())
                    if z.maximum_absolute_error.notna().any()
                    else None,
                    blocked_reasons={
                        str(k): int(v)
                        for k, v in z.loc[z.status.eq("blocked"), "reason"].value_counts().items()
                    },
                )
            )
    end = selected[selected.scenario == "interval_end"]
    start = selected[selected.scenario == "interval_start"]
    common = set(end.directory) & set(start.directory)
    exact = overlap_pairs = shared_records = 0
    shared_support = 0.0
    for directory in common:
        x = selected_records[(directory, "interval_end")]
        y = selected_records[(directory, "interval_start")]
        shared = len(set(x.n_logical_row) & set(y.n_logical_row))
        shared_records += shared
        overlap_pairs += int(shared > 0)
        exact += int(np.array_equal(x.n_logical_row, y.n_logical_row))
        sx, ex = x.c_hypothesized_start.to_numpy(), x.c_hypothesized_end.to_numpy()
        sy, ey = y.c_hypothesized_start.to_numpy(), y.c_hypothesized_end.to_numpy()
        shared_support += max(
            0.0,
            union_seconds(sx, ex)
            + union_seconds(sy, ey)
            - union_seconds(np.r_[sx, sy], np.r_[ex, ey]),
        )
    aggregate = dict(
        version=version,
        selection_version=selection.name,
        label=LABEL,
        inputs=inputs,
        selected_counts={s: int((selected.scenario == s).sum()) for s in c["scenarios"]},
        feature_summary=feature_summary,
        duration_summary=duration_summary,
        package_summary=package_summary,
        sample_overlap=dict(
            shared_participants=len(common),
            identical_source_ranges=exact,
            overlapping_run_pairs=overlap_pairs,
            shared_logical_intervals=shared_records,
            shared_supported_minutes=shared_support / 60,
        ),
        real_calculations_limited_to_selected_runs=True,
        final_test_processed=False,
        pss_targets_calculated=False,
        spectral_nonlinear_calculated=False,
        ema_features_aggregated=False,
    )
    write_json(build / "aggregate_results.json", aggregate)
    input_guard(root, reference, root / "data")
    publish_bundle(
        build,
        destination,
        dict(
            version=version,
            inputs=inputs,
            run_rows=len(result),
            selection_frozen_before_features=True,
        ),
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["select", "calculate"])
    parser.add_argument(
        "--reference", type=Path, default=os.environ.get("GARMIN_REFERENCE_PROJECT")
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--execute-authorized-pilot", action="store_true")
    args = parser.parse_args()
    try:
        if args.reference is None:
            raise AuditError("Set GARMIN_REFERENCE_PROJECT.")
        if args.phase == "select":
            out = select(Path.cwd(), args.reference)
        else:
            if args.selection is None:
                raise AuditError("Provide the immutable selection bundle.")
            out = calculate(
                Path.cwd(), args.reference, args.selection, authorized=args.execute_authorized_pilot
            )
        print("Completed pilot phase:", args.phase, str(out.relative_to(Path.cwd())), flush=True)
    except Exception as exc:
        print(
            str(exc)
            if isinstance(exc, AuditError)
            else f"Pilot stopped ({type(exc).__name__}); private details withheld.",
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
