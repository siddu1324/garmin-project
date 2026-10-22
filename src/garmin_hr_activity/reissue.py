"""Reissue all frozen IBI policies, preserving and explicitly superseding bad versions."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.audit import audit_windows
from garmin_preprocessing.common import (
    input_guard,
    verify_bundle,
    verify_archive_bundle,
    write_json,
)
from garmin_preprocessing.report import publish_report
from .boundaries import verify_grid

OLD = "private/preprocessing/eligibility/6747223eb50269ba"
NORMALIZED = "private/preprocessing/normalized/31e51c934bcdeffa"


def reissue(root, reference):
    input_guard(root, reference, root / "data")
    normalized = root / NORMALIZED
    nm = verify_bundle(normalized)
    verify_archive_bundle(root / OLD)  # Historical comparison only; never a calculation input.
    audited, manifest = audit_windows(root, normalized, nm)
    ema = pd.read_parquet(
        normalized / "ema/development_ema.parquet", columns=["n_ema_id", "n_cutoff_epoch"]
    )
    cutoffs = ema.set_index("n_ema_id").n_cutoff_epoch.to_dict()
    cfg = json.loads((root / "config/ibi_eligibility.json").read_text())
    rows = changed = short_rows = 0
    for p in sorted((audited / "windows").glob("*.parquet")):
        f = pd.read_parquet(p)
        rows += verify_grid(
            f[["n_ema_id", "n_step_index", "n_window_start_epoch", "n_window_end_epoch"]],
            cutoffs,
            cfg["lookbacks_minutes"],
        )
        old = pd.read_parquet(root / OLD / "windows" / p.name)
        changed += int(f.n_window_end_epoch.ne(old.n_window_end_epoch).sum())
        short = f.n_step_index <= 355  # longest unaffected approved lookback is 360 min
        # The dtype itself changed intentionally; values, diagnostics and row order must agree.
        pd.testing.assert_frame_equal(
            f.loc[short].reset_index(drop=True),
            old.loc[short].reset_index(drop=True),
            check_dtype=False,
        )
        short_rows += int(short.sum())
    current = json.loads((audited / "aggregate_results.json").read_text())
    prior = json.loads((root / OLD / "aggregate_results.json").read_text())
    keys = ["policy", "scenario", "lookback_minutes", "survey_category"]
    previous = {tuple(r[k] for k in keys): r for r in prior["coverage"]}
    delta = []
    for row in current["coverage"]:
        old = previous[tuple(row[k] for k in keys)]
        if row["lookback_minutes"] <= 360 and row != old:
            raise AuditError("Previously unaffected short-lookback aggregates changed.")
        values = {k: row[k] for k in keys}
        for field in [
            "eligible_windows",
            "eligible_ema",
            "eligible_participants",
            "recorded_windows",
            "unique_supported_seconds",
            "unique_eligible_window_seconds",
        ]:
            values["obsolete_" + field] = old[field]
            values["corrected_" + field] = row[field]
            values["delta_" + field] = row[field] - old[field]
        delta.append(values)
    old_attr = {tuple(r[k] for k in keys + ["reason"]): r["windows"] for r in prior["attrition"]}
    attrition = []
    for r in current["attrition"]:
        before = old_attr[tuple(r[k] for k in keys + ["reason"])]
        if r["lookback_minutes"] <= 360 and r["windows"] != before:
            raise AuditError("Unaffected short-lookback rejection counts changed.")
        attrition.append(dict(r, obsolete_windows=before, delta_windows=r["windows"] - before))
    # Existing publisher only reads aggregate data; no features or expanded HRV.
    public = publish_report(root, normalized, audited)
    result = dict(
        schema_version=1,
        status="complete",
        reason="signed_int16_seconds_overflow_from_step_547",
        affected_steps=[547, 1435],
        overflow_regression_steps=[546, 547, 1092, 1093, 1435],
        corrected_eligibility=str(audited.relative_to(root)),
        corrected_manifest_sha256=sha256(audited / "manifest.json"),
        normalized_bundle=NORMALIZED,
        window_rows=rows,
        changed_clock_rows=changed,
        unaffected_short_window_rows_compared=short_rows,
        unchanged_lookbacks_minutes=[5, 15, 30, 60, 120, 360],
        changed_lookbacks_minutes=[720, 1440],
        independent_validation="Every row checked using Python int/datetime/timedelta, not vectorized generator arithmetic.",
        obsolete_artifacts=[
            dict(
                path=OLD,
                status="rejected_for_downstream_use",
                replacement=str(audited.relative_to(root)),
            ),
            dict(
                path="reports/ibi_eligibility/6747223eb50269ba_90911a7ae670",
                status="historical_only",
                replacement=str(public.relative_to(root)),
            ),
            dict(
                path="private/hrv_input_validation/diagnostics/bc13763b35972be4",
                status="invalid_long_window_index; loader rejects entire bundle",
                replacement="Use corrected eligibility window loader; immediate materializations are historical-only",
            ),
            dict(
                path="reports/hrv_input_validation/30a0bbe188d56b40",
                status="long-lookback continuity counts obsolete; immediate and synthetic results unchanged",
                replacement="Corrected eligibility counts below; full long-window continuity study not reissued",
            ),
            dict(
                path="notebooks/01_preprocessing_and_ibi_eligibility.ipynb",
                status="historical pinned output; use Notebook 04",
                replacement="notebooks/04_hr_activity_features.ipynb",
            ),
            dict(
                path="notebooks/02_hrv_input_validation.ipynb",
                status="historical pinned output; long-window counts invalid",
                replacement="notebooks/04_hr_activity_features.ipynb",
            ),
        ],
        unaffected=[
            "raw archives",
            "normalized bundle",
            "frozen assignments and crosswalk",
            "inventory lookbacks using direct scalar bounds",
            "Notebook 03 corrected Screen selection and limited pilot",
        ],
        policies_unchanged=True,
        coverage_delta=delta,
        attrition_delta=attrition,
    )
    pointer = root / "private/hr_activity/diagnostics_run.json"
    if pointer.exists():
        dependent = json.loads(pointer.read_text())
        folder = root / dependent["diagnostics"]
        dm = verify_bundle(folder)
        if dm["inputs"]["eligibility_manifest"] != sha256(audited / "manifest.json"):
            raise AuditError("Dependent diagnostics reference a different window version.")
        current_d = json.loads((folder / "aggregate_results.json").read_text())
        prior_d = json.loads(
            (
                root
                / "private/hrv_input_validation/diagnostics/bc13763b35972be4/aggregate_results.json"
            ).read_text()
        )
        unchanged = [
            "overview",
            "reasons",
            "combinations",
            "variations",
            "runs",
            "pair_timing",
            "timestamp_precision",
        ]
        if any(current_d[k] != prior_d[k] for k in unchanged):
            raise AuditError("Unaffected immediate/timing diagnostics changed.")
        result["corrected_input_diagnostics"] = dependent["diagnostics"]
        result["corrected_input_report"] = dependent["report"]
        result["unchanged_dependent_sections"] = unchanged
        keys_d = ["policy", "scenario", "scope", "subset"]
        before = {tuple(r[k] for k in keys_d): r for r in prior_d["continuity"]}
        differences = []
        for row in current_d["continuity"]:
            old_row = before[tuple(row[k] for k in keys_d)]
            differences.append(
                {
                    **{k: row[k] for k in keys_d},
                    **{
                        prefix + k: value
                        for k in row
                        if k not in keys_d
                        for prefix, value in [
                            ("obsolete_", old_row[k]),
                            ("corrected_", row[k]),
                            ("delta_", row[k] - old_row[k]),
                        ]
                    },
                }
            )
        result["continuity_delta"] = differences
        for item in result["obsolete_artifacts"]:
            if item["path"] == "private/hrv_input_validation/diagnostics/bc13763b35972be4":
                item["replacement"] = dependent["diagnostics"]
                item["status"] = "rejected_for_downstream_use; complete corrected index reissued"
            if item["path"] == "reports/hrv_input_validation/30a0bbe188d56b40":
                item["replacement"] = dependent["report"]
                item["status"] = (
                    "superseded; full continuity counts reissued; immediate/synthetic sections unchanged"
                )
    write_json(root / "reports/window_supersession.json", result)
    write_json(
        root / "private/hr_activity/reissue.json",
        dict(eligibility=str(audited.relative_to(root)), report=str(public.relative_to(root))),
    )
    input_guard(root, reference, root / "data")
    print("Reissued eligibility:", audited.relative_to(root), flush=True)
    print("Corrected report:", public.relative_to(root), flush=True)
    return audited


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=os.environ.get("GARMIN_REFERENCE_PROJECT"))
    a = p.parse_args()
    if a.reference is None:
        raise AuditError("Set GARMIN_REFERENCE_PROJECT.")
    reissue(Path.cwd(), a.reference)


if __name__ == "__main__":
    main()
