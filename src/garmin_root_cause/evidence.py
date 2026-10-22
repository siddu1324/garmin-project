"""Focused measurement summaries from existing ledgers; no re-inventory or semantic repair."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_comprehensive_regression.evaluate import finite
from garmin_preprocessing.common import write_json
from .data import HR, STEP, TASKS, recall_gate
from .preservation import completed


def run(root):
    root = Path(root)
    ptr = json.loads((root / "private/root_cause_analysis/current.json").read_text())
    data, folder = root / ptr["data"], root / ptr["run"]
    cfg = json.loads((folder / "identity.json").read_text())["config"]
    out = root / "reports/root_cause_analysis" / folder.name
    if completed(root):
        return out
    f = pd.read_parquet(data / "development.parquet")
    people = json.loads(
        (root / cfg["normalized_bundle"] / "participant_directories.json").read_text()
    )
    counts = dict(
        logical_records=0,
        accepted_distinct_timestamps=0,
        timestamp_conflict_records=0,
        same_time_value_excess=0,
        bad_value_records=0,
        bad_clock_records=0,
        repeated_value_adjacent_pairs=0,
        accepted_adjacent_pairs=0,
    )
    accepted_values = []
    within_deltas = []
    for person in people:
        h = pd.read_parquet(
            root
            / cfg["ledger_bundle"]
            / "reading_dispositions"
            / person["directory"]
            / "hr.parquet"
        )
        counts["logical_records"] += len(h)
        for source, dest in [
            ("a_conflict", "timestamp_conflict_records"),
            ("a_same_value_tie_excess", "same_time_value_excess"),
            ("a_bad_value", "bad_value_records"),
            ("a_bad_clock", "bad_clock_records"),
        ]:
            counts[dest] += int(h[source].sum())
        z = h[h.a_first & h.a_acceptable]
        counts["accepted_distinct_timestamps"] += len(z)
        counts["repeated_value_adjacent_pairs"] += int(z.n_value.diff().eq(0).sum())
        counts["accepted_adjacent_pairs"] += max(0, len(z) - 1)
        accepted_values.extend(z.n_value.to_numpy())
        within_deltas.extend(z.n_epoch_s.diff().dropna().to_numpy())
    counts["accepted_hr_quantiles"] = dict(
        zip(
            ["min", "p25", "median", "p75", "max"],
            np.quantile(accepted_values, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        )
    )
    counts["accepted_timestamp_gap_seconds_quantiles"] = dict(
        zip(
            ["p25", "median", "p75", "p95"],
            np.quantile(within_deltas, [0.25, 0.5, 0.75, 0.95]).tolist(),
        )
    )
    counts["repeated_values_interpretation"] = (
        "Equal values at different clocks are retained; integer resolution, true stable HR or stale/batched reports cannot be separated without provider evidence. Counts overlap. Gaps here do not create HRV adjacency."
    )
    quality = []
    columns = [c for c in f if c.startswith("quality__hr__")]
    for name, g in [("all_development", f), ("matched_observed", f[f.matched])]:
        for col in columns:
            z = g[col].dropna()
            quality.append(
                dict(
                    cohort=name,
                    measure=col,
                    observed=len(z),
                    q25=float(z.quantile(0.25)),
                    median=float(z.median()),
                    q75=float(z.quantile(0.75)),
                )
            )
    # A gate row for every development response; unavailable does not mean a zero-length window.
    blocked = []
    for r in f.itertuples():
        row = dict(cutoff_epoch=r.n_cutoff_epoch)
        reasons = recall_gate(row, {})
        blocked.append(
            dict(
                n_ema_id=r.n_ema_id,
                n_participant=r.n_participant,
                n_fold=r.n_fold,
                survey=r.n_survey_category,
                eligible=False,
                reasons="|".join(reasons),
            )
        )
    pd.DataFrame(blocked).to_parquet(folder / "recall_gate.parquet", index=False)
    missing = []
    for cat, g in f.groupby("n_survey_category"):
        missing.append(
            dict(
                survey=cat,
                responses=len(g),
                participants=g.n_participant.nunique(),
                verified_pre_response_recall_boundaries=0,
                missing_verified_acquisition_clock=len(g),
                missing_verified_answer_availability=len(g),
                missing_complete_prompt_log=len(g) if cat != "Morning" else 0,
                current_response_wake_only=len(g) if cat == "Morning" else 0,
            )
        )
    probe = pd.read_parquet(data / "activity_probe.parquet")
    eligible = probe[probe.eligible]
    person_means = eligible.groupby(["n_participant", "target"])[HR[0]].mean().unstack()
    paired = person_means.dropna()
    delta = paired[1.0] - paired[0.0] if 1.0 in paired and 0.0 in paired else pd.Series(dtype=float)
    rng = np.random.default_rng(cfg["seed"])
    boot = (
        np.mean(rng.choice(delta.to_numpy(), size=(2000, len(delta)), replace=True), axis=1)
        if len(delta)
        else np.array([np.nan])
    )
    physical = dict(
        participants_with_both=len(delta),
        mean_within_person_active_minus_inactive_bpm=float(delta.mean()),
        lower=float(np.nanquantile(boot, 0.025)),
        upper=float(np.nanquantile(boot, 0.975)),
        equal_participant_weight=True,
        meaning="Conditional counter-defined activity; observed supports only, no PSS selection; not ground-truth physical activity",
    )
    person_means.to_parquet(folder / "private_probe_person_means.parquet")
    availability = []
    for state, g in f.groupby(f[HR].notna().all(axis=1)):
        for survey, z in g.groupby("n_survey_category"):
            availability.append(
                dict(
                    all_hr_observed=bool(state),
                    survey=survey,
                    responses=len(z),
                    participants=z.n_participant.nunique(),
                    valid_scores=int(z.pss_combo.notna().sum()),
                    mean_score=float(z.pss_combo.mean()),
                    median_score=float(z.pss_combo.median()),
                )
            )
    label_features = []
    for task in TASKS:
        for label, group in f[f.matched].groupby("target_" + task):
            for col in HR + [STEP]:
                label_features.append(
                    dict(
                        task=task,
                        label=int(label),
                        feature=col,
                        responses=len(group),
                        participants=group.n_participant.nunique(),
                        mean=float(group[col].mean()),
                        q25=float(group[col].quantile(0.25)),
                        median=float(group[col].median()),
                        q75=float(group[col].quantile(0.75)),
                    )
                )
    for name, obj in dict(
        sensor_summary=counts,
        quality_summary=quality,
        missing_evidence_counts=missing,
        physical_consistency=physical,
        sensor_availability_scores=availability,
        feature_distributions_by_label=label_features,
    ).items():
        write_json(out / f"{name}.json", finite(obj))
    return out


if __name__ == "__main__":
    print(run(Path.cwd()))
