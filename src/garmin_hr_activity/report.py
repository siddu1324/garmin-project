"""Aggregate-only Notebook 04 report, dictionary and explanatory figures."""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle, write_json


def table(rows, columns):
    def fmt(x):
        if isinstance(x, (int, np.integer)):
            return f"{x:,}"
        if isinstance(x, (float, np.floating)):
            return f"{x:,.2f}"
        return str(x)

    return "\n".join(
        ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        + ["| " + " | ".join(fmt(r[c]) for c in columns) + " |" for r in rows]
    )


def figures(build, result):
    a = pd.DataFrame(result["availability"])
    o = pd.DataFrame(result["overlap"])
    g = pd.DataFrame(result["gaps"])
    selected = [
        ("hr", "mean_bpm", "HR mean"),
        ("hr", "slope_bpm_per_minute", "HR slope"),
        ("steps", "added_supported", "Steps"),
        ("floorsClimbed", "added_supported", "Floors"),
    ]
    labels = ["5m", "15m", "30m", "1h", "2h", "6h", "12h", "24h"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for sensor, metric, label in selected:
        z = a[
            (a.sensor == sensor) & (a.metric == metric) & (a.survey_category == "All")
        ].sort_values("lookback_minutes")
        axes[0].plot(range(8), 100 * z.available_ema / z.ema_denominator, "o-", label=label)
        axes[1].plot(range(8), z.available_participants, "o-", label=label)
    for ax in axes:
        ax.set_xticks(range(8), labels)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Lookback")
    axes[0].set_ylabel("Available EMA rows (%) / 3,706")
    axes[1].set_ylabel("Participants with any available EMA / 125")
    axes[0].legend()
    fig.suptitle(
        "Provisional observed HR and supported activity — availability is not complete coverage"
    )
    fig.savefig(build / "availability.png", dpi=150, metadata={"Software": "garmin-hr-activity"})
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
    for ax, (sensor, metric, label) in zip(axes.flat, selected):
        z = a[(a.sensor == sensor) & (a.metric == metric) & (a.survey_category != "All")].copy()
        z["percent"] = 100 * z.available_ema / z.ema_denominator
        pivot = z.pivot(
            index="survey_category", columns="lookback_minutes", values="percent"
        ).reindex(["Morning", "Midday", "Endday"])
        ax.imshow(pivot, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        for y in range(3):
            for x in range(8):
                ax.text(
                    x,
                    y,
                    f"{pivot.iloc[y, x]:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if pivot.iloc[y, x] > 60 else "black",
                    fontsize=8,
                )
        ax.set_xticks(range(8), labels)
        ax.set_yticks(range(3), pivot.index)
        ax.set_title(label)
    fig.suptitle("Availability within survey category; all category EMAs stay in denominators")
    fig.savefig(
        build / "survey_availability.png", dpi=150, metadata={"Software": "garmin-hr-activity"}
    )
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    z = o[o.survey_category == "All"].sort_values("lookback_minutes")
    bottom = np.zeros(8)
    for key, label, color in [
        ("hr_and_activity_ema", "HR + activity", "#287d80"),
        ("hr_only_ema", "HR only", "#7199be"),
        ("activity_only_ema", "Activity only", "#d99a42"),
        ("neither_ema", "Neither", "#c8c8c8"),
    ]:
        axes[0].bar(range(8), z[key], bottom=bottom, label=label, color=color)
        bottom += z[key].to_numpy()
    axes[0].set_xticks(range(8), labels)
    axes[0].set_ylabel("EMA rows / 3,706")
    axes[0].legend(fontsize=8)
    for sensor in ["hr", "steps", "floorsClimbed"]:
        z = g[(g.sensor == sensor) & (g.survey_category == "All")].sort_values("lookback_minutes")
        axes[1].plot(range(8), 100 * z.ema_with_gap_gt_60s / z.ema_denominator, "o-", label=sensor)
    axes[1].set_xticks(range(8), labels)
    axes[1].set_ylabel("EMAs with any unobserved gap >1 min (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    axes[0].set_title("Availability overlap, not simultaneous measurement")
    axes[1].set_title("Internal and edge gaps, including absent recordings")
    fig.savefig(
        build / "overlap_and_gaps.png", dpi=150, metadata={"Software": "garmin-hr-activity"}
    )
    plt.close(fig)


def publish(root, bundle):
    verify_bundle(bundle)
    data = json.loads((bundle / "aggregate_results.json").read_text())
    quality = pd.read_parquet(
        bundle / "quality.parquet",
        columns=[
            "sensor",
            "lookback_minutes",
            "n_survey_category",
            "maximum_gap_including_edges_seconds",
        ],
    )
    for row in data["gaps"]:
        z = quality[
            (quality.sensor == row["sensor"])
            & (quality.lookback_minutes == row["lookback_minutes"])
        ]
        if row["survey_category"] != "All":
            z = z[z.n_survey_category == row["survey_category"]]
        row["ema_with_gap_gt_60s"] = int(z.maximum_gap_including_edges_seconds.gt(60).sum())
    supersession = json.loads((root / "reports/window_supersession.json").read_text())
    inputs = dict(
        feature_manifest=sha256(bundle / "manifest.json"),
        supersession_sha256=sha256(root / "reports/window_supersession.json"),
        renderer_sha256=sha256(Path(__file__)),
    )
    version = digest_object(inputs)[:16]
    parent = root / "reports/hr_activity"
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / version
    if dest.exists():
        verify_bundle(dest)
        return dest
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    for key in ["availability", "gaps", "overlap"]:
        write_json(build / f"{key}.json", data[key])
    write_json(
        build / "summary.json",
        {k: v for k, v in data.items() if k not in ["availability", "gaps", "overlap"]},
    )
    write_json(
        build / "feature_dictionary.json",
        json.loads((bundle / "feature_dictionary.json").read_text()),
    )
    figures(build, data)
    all_a = pd.DataFrame(data["availability"])
    all_a = all_a[all_a.survey_category == "All"]
    availability = []
    for m in sorted(all_a.lookback_minutes.unique()):
        z = all_a[all_a.lookback_minutes == m]

        def count(sensor, metric, field="available_ema"):
            return int(z[(z.sensor == sensor) & (z.metric == metric)].iloc[0][field])

        availability.append(
            dict(
                minutes=int(m),
                HR=count("hr", "mean_bpm"),
                HR_participants=count("hr", "mean_bpm", "available_participants"),
                slope=count("hr", "slope_bpm_per_minute"),
                steps=count("steps", "added_supported"),
                step_participants=count("steps", "added_supported", "available_participants"),
                floors=count("floorsClimbed", "added_supported"),
            )
        )
    differences = [
        r
        for r in supersession["coverage_delta"]
        if r["survey_category"] == "All" and r["lookback_minutes"] >= 720
    ]
    changes = [
        dict(
            policy=r["policy"],
            hypothesis=r["scenario"],
            minutes=r["lookback_minutes"],
            old_windows=r["obsolete_eligible_windows"],
            new_windows=r["corrected_eligible_windows"],
            old_EMAs=r["obsolete_eligible_ema"],
            new_EMAs=r["corrected_eligible_ema"],
            new_participants=r["corrected_eligible_participants"],
        )
        for r in differences
    ]
    missing = []
    for m in [5, 1440]:
        for sensor, metric in [
            ("hr", "mean_bpm"),
            ("hr", "slope_bpm_per_minute"),
            ("steps", "added_supported"),
            ("floorsClimbed", "added_supported"),
        ]:
            row = all_a[
                (all_a.lookback_minutes == m) & (all_a.sensor == sensor) & (all_a.metric == metric)
            ].iloc[0]
            for reason, count in row.reasons.items():
                missing.append(
                    dict(minutes=m, feature=f"{sensor}_{metric}", reason=reason, EMA_rows=count)
                )
    report = f"""# Notebook 04 — corrected IBI audit and provisional HR/activity features

All **3,706 development EMAs, 125 participants and five frozen folds** are retained. All 32 final-test
participants remain sealed. This checkpoint reissues the faulty window audit and adds observed HR
summaries plus supported steps/floor increments. No expanded HRV, Garmin stress predictor, PSS target,
imputation, lookback/feature selection or model was calculated. Timing remains provisional because
acquisition versus receipt/batching semantics are unresolved.

## Corrected window provenance

The signed int16 minute-step multiplication wrapped at step 547 (9 h 7 min) and again at 1093.
We reissued all three frozen policies, both interval hypotheses, every approved lookback and all
survey categories using the fixed int64 generator. **{supersession["changed_clock_rows"]:,}** of
**{supersession["window_rows"]:,}** locations change; all thresholds are unchanged. Each window was
independently verified with Python integer/datetime arithmetic: exact 300-second duration,
60-second steps, membership wholly inside its lookback and no end beyond the actual EMA cutoff.
Regression cases include steps 546/547, 1092/1093 and 1435, the full 24-hour boundary.

Every approved result through six hours agrees with the obsolete version, including all
**{supersession["unaffected_short_window_rows_compared"]:,}** corresponding diagnostic rows and all
coverage/attrition aggregates. The 12/24-hour changes below demonstrate why merely checking row
counts or regenerating offsets with the same dtype was insufficient.

{table(changes, ["policy", "hypothesis", "minutes", "old_windows", "new_windows", "old_EMAs", "new_EMAs", "new_participants"])}

The complete 192-row coverage table, all rejection reasons, unique-time unions and category results
are in the [corrected eligibility report](../../ibi_eligibility/{Path(supersession["obsolete_artifacts"][1]["replacement"]).name}/report.md).
[The explicit supersession record](../../window_supersession.json) preserves old/new/delta counts,
supported-time changes and the affected artifact graph. Old bundles remain byte-identical.
Default bundle validation and both window loaders reject revoked versions even when renamed.
Historical checksum verification is explicitly separate and cannot authorize their use as inputs.

Notebook 02's dependent index and full continuity report were also reissued at
`{supersession.get("corrected_input_diagnostics", "pending")}` and
`{supersession.get("corrected_input_report", "pending")}`. Its immediate rejection analyses,
synthetic calculations and whole-stream timing diagnostics are unchanged. Notebook 03's corrected
Screen index and limited pilot remain valid and unchanged. Inventory direct-lookback counts,
raw normalization, mappings, assignments and policies were not affected by the overflow.

## How one EMA becomes one feature row

Read only EMA identity/category/clock metadata. Recover the verified participant's authorized HR
and counter partitions. Derive actual UTC from local start minus its own offset; subtract 60 seconds
only for minute-resolution starts. Query [cutoff−lookback,cutoff) separately for all eight lookbacks.
Keep missing results in the original EMA row; retain its original fold. The private feature table
has **3,706 rows, 104 measurement columns**, of which **88** are explicitly listed as provisional
predictors. The 16 calorie/intensity placeholders stay missing and excluded. Identity, category,
cutoff, fold and quality are not predictors. No PSS outcome is attached.

[The prespecified feature specification](../../hr_activity_feature_specification.md) documents
source fields, units, sampling, duplicates and thresholds before feature retention was compared.
The [machine-readable dictionary](feature_dictionary.json) specifies every column, units, lookback,
minimum support and predictor role. Source strings and physical records stay in the unchanged
normalized bundle; the new reading ledger accounts for **{data["reading_disposition_rows"]:,}**
logical observations with dispositions and source-row/hash links.

## Definitions and exclusions

HR values are provisionally bpm. Accept finite 30–240 inclusive values with no existing clock
flags. Collapse a same-timestamp/same-value group to one analysis reading only if all members
pass; preserve original rows and reject conflicting groups. Identical values at different times
remain separate readings. These range/duplicate rules are provisional, not clinical validation.

Mean, median, sample SD (N−1), min, max, Q25 and Q75 require at least two distinct valid timestamps
and a one-second span. Latest requires one reading. OLS slope against actual elapsed minutes
requires three readings and a 60-second span. Quantiles use the linear order-statistic definition.
All describe observed readings; no time weighting, interpolation, filling or extrapolation occurs.
A fitted slope across sparse readings does not establish a continuously recorded trend.

Steps/floors are provisionally cumulative counters. Difference consecutive original timestamp
groups only when endpoints are accepted nonnegative integers, on the same exported local day,
with the same offset/source content, and 0<dt≤300 seconds. Both endpoints must be inside the
lookback, with the right endpoint strictly before cutoff. Date transitions, decreases/resets,
conflicts, rejected endpoints, source/offset changes, long gaps and boundary-crossing intervals
are excluded. Rejected groups stay as barriers; no new pair jumps over one. An accepted observation
after a reset may begin a new pair. Sum only supported additions, not raw counters or entire-window
activity. One actual supported zero-change pair can produce zero; no supported pair produces missing.
Hidden resets, delayed updates and undocumented corrections remain possible.

Calorie units and intensity-minute weighting/reset semantics are not sufficiently confirmed for
attribution; all such placeholders remain missing with explicit reasons. Their observed recordings
still appear in quality diagnostics. The old 2-second HR / 60-second other-sensor gap diagnostics
are not promoted to eligibility rules. A calorie interval just over 60 seconds is not automatically
invalid. The new 300-second counter-pair limit is an explicit provisional attribution choice, fixed
before retention comparison, not a claim of complete coverage.

## Clearly synthetic teaching examples

A synthetic second-resolution EMA at 12:00 has five-minute window [11:55,12:00). Valid HR readings
60,90,120 bpm at 11:55,11:56,11:57 yield mean/median=90, sample SD=30, min=60, max=120,
Q25=75, Q75=105, latest=120 and slope=30 bpm/min. Freshness is three minutes, a separate quality
value. A reading at exactly 12:00 is excluded. One reading could produce latest but not those
other summaries. A minute-resolution 12:00 EMA instead has cutoff 11:59.

For a synthetic counter, 100→110→3→8 yields supported additions 10+5=15 when all pair timing
conditions hold; the 110→3 reset contributes nothing. If 100 is before the left boundary and 110
is inside, the increment 10 is excluded entirely. An unobserved interval never becomes zero.
These examples are not participant data and do not select the real-data rules.

## Availability and overlap

Every all-cohort count below uses 3,706 EMAs. Availability is not continuous measurement or
representativeness. Participant counts indicate at least one available EMA, not complete coverage.

{table(availability, ["minutes", "HR", "HR_participants", "slope", "steps", "step_participants", "floors"])}

![Availability](availability.png)
![Survey availability](survey_availability.png)

HR/activity overlap means the same EMA has HR mean and at least one supported steps/floors
feature; it does not claim the sensors recorded simultaneously. At five minutes, 648 EMAs have
both, 838 HR only, 18 activity only and 2,202 neither. At 24 hours those counts are 3,230, 229,
65 and 182. Lookbacks and EMA windows overlap; these are not independent observations.

![Overlap and gaps](overlap_and_gaps.png)

`availability.json` contains every feature/lookback/survey count and reason; `gaps.json` contains
reading counts, freshness, internal/edge gaps and occupied-bin fractions; the plot uses a >60-second
gap diagnostic, not an eligibility rule; `overlap.json` contains
HR/activity intersections. Quality is stored separately in **148,240 rows** (3,706 × eight lookbacks
× five streams), keyed by EMA/sensor/lookback. HR point observations have no established duration:
occupied one-minute bins are a density diagnostic, never seconds of physiological support. Counter
support is the union of accepted pair intervals; cohort unique support deduplicates pairs across
EMA views before summing. No number of overlapping windows is multiplied by window duration.

The following selected reason tables distinguish absent recordings, all rejected readings and
insufficient feature-specific support. Reasons within a feature are mutually exclusive; underlying
clock/value/conflict flags can overlap and remain separate diagnostic counts.

{table(missing, ["minutes", "feature", "reason", "EMA_rows"])}

## Limits and next action

All timing claims remain provisional. Earlier observations can support numerical summaries while
leaving large unobserved gaps. More retained rows at 24 hours do not establish an optimal lookback
or relevance to an EMA; no lookback or threshold was selected from retention. Steps/floor sums are
supported additions, not complete-window activity; HR summaries are reading-weighted, not a
continuous-time average. The excluded calorie/intensity placeholders need provider definitions.

The [provider-question draft](../../provider_questions_draft.md) asks about timestamp meaning and
batching, IBI anchoring/missed beats, device and beat-quality annotations, and counter units,
reset periods and measurement intervals. **It is a local draft only and was not sent.**
The next concrete step is provider review of these semantics and the prespecified feature dictionary
before authorizing outcome linkage or modeling. No expanded HRV is warranted by HR availability.

## Outputs and reproducibility

- Feature rows: `{bundle.relative_to(root)}/features.parquet`.
- Separate quality: `{bundle.relative_to(root)}/quality.parquet`.
- Window/source references: `{bundle.relative_to(root)}/window_lineage.parquet`.
- Every logical reading's disposition and counter adjacency: `reading_dispositions/` and `counter_pairs/` in that bundle.
- Corrected all-policy audit: `{supersession["corrected_eligibility"]}`.
- Source notebook: `notebooks/04_hr_activity_features.ipynb`; execution belongs under ignored `outputs/`.

All private outputs are ignored. Versioned manifests bind source hashes, frozen assignments,
configuration, specification, code and corrected provenance; reruns verify/reuse existing versions
and never overwrite validated bundles. Verification is recorded alongside this report. Previous
work remains present; nothing is staged, committed or pushed.
"""
    (build / "report.md").write_text(report)
    publish_bundle(build, dest, dict(version=version, inputs=inputs, aggregate_only=True))
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    a = p.parse_args()
    out = publish(Path.cwd(), a.bundle.resolve())
    write_json(
        Path("private/hr_activity/report_run.json"), dict(report=str(out.relative_to(Path.cwd())))
    )
    print("Published HR/activity report:", out.relative_to(Path.cwd()))


if __name__ == "__main__":
    main()
