"""Publish aggregate-only pilot tables, figures and a plain-language explanation."""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from garmin_hrv_input.synthetic import independent_reference
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle, write_json


def table(rows, columns):
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    return "\n".join(
        ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        + ["| " + " | ".join(fmt(r[c]) for c in columns) + " |" for r in rows]
    )


def publish(root, selection, calculation):
    verify_bundle(selection)
    verify_bundle(calculation)
    census = json.loads((selection / "aggregate_selection.json").read_text())
    result = json.loads((calculation / "aggregate_results.json").read_text())
    inputs = dict(
        selection_manifest=sha256(selection / "manifest.json"),
        calculation_manifest=sha256(calculation / "manifest.json"),
        code_sha256=sha256(Path(__file__)),
    )
    version = digest_object(inputs)[:16]
    parent = root / "reports/ibi_pilot"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / version
    if destination.exists():
        verify_bundle(destination)
        return destination
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    for name, value in [
        ("feasibility", census["feasibility"]),
        ("feature_summary", result["feature_summary"]),
        ("duration_summary", result["duration_summary"]),
        ("package_summary", result["package_summary"]),
        ("sample_overlap", result["sample_overlap"]),
    ]:
        write_json(build / f"{name}.json", value)
    example = independent_reference([800, 900, 1000, 950, 850])
    write_json(
        build / "synthetic_example.json",
        dict(
            label="SYNTHETIC — not participant data",
            intervals_ms=[800, 900, 1000, 950, 850],
            signed_differences_ms=[100, 100, -50, -100],
            results=example,
        ),
    )
    f = pd.DataFrame(census["feasibility"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for scenario, color in [("interval_end", "#2864a5"), ("interval_start", "#ba5726")]:
        g = f[f.scenario == scenario]
        for ax, field, title in zip(
            axes,
            ["qualifying_ema", "qualifying_participants", "unique_supported_minutes"],
            [
                "EMAs with ≥1 qualifying run / 3,706",
                "Participants with ≥1 run / 125",
                "Unique supported minutes (union)",
            ],
        ):
            ax.plot(range(8), g[field], "o-", label=scenario.replace("_", " "), color=color)
            ax.set_xticks(
                range(8), ["5m", "15m", "30m", "1h", "2h", "6h", "12h", "24h"], rotation=40
            )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Approved lookback")
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Full development feasibility census — not the 20-run calculation sample")
    fig.savefig(build / "feasibility.png", dpi=150, metadata={"Software": "garmin-ibi-pilot"})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.barh(
        ["No recorded IBI", "Recorded, fails ≥240s pilot gate", "Qualifies for pilot"],
        [2196, 1509, 1],
        color=["#888888", "#c78840", "#287c74"],
    )
    ax.set_xlabel("Development EMAs (3,706 total)")
    ax.set_title(
        "Immediate five-minute observation window\nSame pilot counts under both timestamp hypotheses"
    )
    for i, v in enumerate([2196, 1509, 1]):
        ax.text(v + 15, i, str(v), va="center")
    ax.set_xlim(0, 2450)
    fig.savefig(
        build / "immediate_attrition.png", dpi=150, metadata={"Software": "garmin-ibi-pilot"}
    )
    plt.close(fig)
    overlap = result["sample_overlap"]
    feasibility_table = table(
        census["feasibility"],
        [
            "scenario",
            "lookback_minutes",
            "qualifying_ema",
            "qualifying_participants",
            "qualifying_window_instances",
            "overlapping_source_episodes",
            "unique_supported_minutes",
        ],
    )
    duration_table = table(
        result["duration_summary"], ["scenario", "measure", "count", "minimum", "median", "maximum"]
    )
    feature_table = table(
        result["feature_summary"], ["scenario", "feature", "count", "minimum", "median", "maximum"]
    )
    package_table = table(
        result["package_summary"],
        ["scenario", "backend", "compared", "blocked", "maximum_absolute_error"],
    )
    report = f"""# Development IBI variability pilot

**IBI variability—normal-beat status unverified.** Calculations are limited to ten selected runs
per timestamp hypothesis, at most one per participant per hypothesis. These are ≥240-second
runs inside five-minute observation windows, not complete five-minute HRV recordings.
Only the five authorized time-domain outputs were retained. No outcomes or PSS were read.
All 32 final-test participants remain sealed; raw inputs, folds and prior bundles are unchanged.

## A cutoff-grid error corrected before any real calculation

The historical window generator multiplied a signed 16-bit step array by 60. At step 547,
the offsets wrapped, putting some nominally pre-EMA windows after the cutoff. The pilot's
cutoff invariant caught this before features. We corrected {census["corrected_window_clocks"]:,}
of {census["window_rows"]:,} stored window clocks in a **new** ignored Screen index and re-evaluated
the unchanged Screen policy at their correct locations. No sensor clocks were shifted.
The generating source now uses int64; a regression test covers both wrap points and 24 hours.
Historical 12/24-hour eligibility/continuity counts in Notebooks 01/02 must not be used;
this report supersedes them for the pilot census. Immediate through six-hour results are
unaffected. Reissuing the full three-policy long-lookback audit remains separate work.

## Selection and input conditions

[The protocol](../../ibi_pilot_protocol.md) fixed selection before calculation. Search all approved
5/15/30/60/120-minute and 6/12/24-hour lookbacks, with 300-second [start,end) windows at 60-second
steps. EMA UTC is local start minus its row's local-minus-UTC offset; Garmin Unix seconds get
no timezone shift. Cutoff is actual start minus 60 seconds for minute precision, actual start
for second precision. These cutoffs and the two interval-boundary hypotheses are unchanged.

Screen requires 250–2500 ms IBI, at least 80% interval/time agreement within max(0.25 s,25% of
expected interval), gap/expected ratio ≤2, each edge gap ≤5 s, ≥270 s supported union,
≥120 s contiguous timestamp span, flagged-change fraction ≤5%, and interval/elapsed ratio
0.8–1.2. Implausible intervals, repeated timestamps/conflicts and clock flags reject the window.
The >20% interval-change flag remains a provisional screen, not proof of an artifact.
For this pilot, one original run must additionally have **both span ≥240 s and supported union
≥240 s**. Each pair in that run passes the original adjacency and Screen timing checks.
Quality measures remain separate from predictors.

An interval is hypothesized as [t−IBI,t] for end, or [t,t+IBI] for start. Both timestamp
membership and whole-interval containment are required; crossing intervals remain in the
ignored member ledger as excluded, never trimmed into new intervals. Exact duplicate physical
records remain preserved upstream; the logical view collapses only exact duplicates. Repeated
interval values at different timestamps remain legitimate. Ties/conflicts are not averaged.
Rejected records, inconsistent pairs and recording gaps sever runs. Small timing residuals
within the frozen tolerance remain uncertainties, not a claim of continuous acquisition.

Exact repeated EMA views are deduplicated by source bounds. Views sharing any original IBI
record are also grouped transitively for conservative source-episode counting. No intervals
are merged: choose the earliest existing run in each component, then earliest per participant,
then earliest ten per hypothesis. Package acceptance and feature values cannot affect selection.
This chronological convenience sample is not representative of all development recordings.

## Full feasibility census, independent of the calculation sample

Every row below uses **3,706 EMAs and 125 development participants** as denominators, including
EMAs without Garmin or qualifying runs. Window instances are overlapping EMA views, not
independent observations. Unique supported minutes use interval unions within each participant;
they are not window counts multiplied by five. One segment within 24 hours is not day-long coverage.

{feasibility_table}

![Full feasibility](feasibility.png)

Across the 24-hour search there are {census["candidate_views"]:,} qualifying run views and
{census["overlap_components"]} overlap components across the two hypotheses. Only 20 selected
run-hypothesis rows receive calculations. The samples share {overlap["shared_participants"]}
participants, {overlap["overlapping_run_pairs"]} overlapping source-run pairs and
{overlap["identical_source_ranges"]} identical source ranges; {overlap["shared_logical_intervals"]}
logical intervals overlap. Their hypothesized supported-time intersection is
{overlap["shared_supported_minutes"]:.3f} minutes. The two samples must not be treated as independent.

### Why immediate availability is so small

Of 3,706 immediate windows, 2,196 contain no recorded IBI and 1,510 contain some IBI.
Only 4 end / 5 start windows pass the original Screen policy, which permits a two-minute run.
Adding this pilot's dual 240-second requirement leaves **one** per hypothesis; 1,509 recorded
windows fail the pilot conditions. This is sparse recording plus fragmentation, not evidence
that a single filter can recover a large immediate sample.

Notebook 02's unchanged immediate end-hypothesis diagnostics found overlapping losses from
short runs (1,469 recorded windows), internal gaps (1,409), insufficient support (926), edge
coverage (227), timing agreement (86), duration/elapsed mismatch (74), jump flags (23), ties
and conflicts (9 each), and clock flags (5). None failed interval plausibility. These counts
overlap and must not be added. Gap-only failures numbered 31 and run-only failures 21.
Removing the jump-flag check alone did not improve Screen retention. Six/seven-decimal timestamps
and compressed arrival clusters are inconsistent with treating all errors as whole-second
rounding. Batching remains possible; retention cannot establish acquisition versus receipt semantics.

![Immediate attrition](immediate_attrition.png)

## Selected-run duration and age

Values below are min/median/max across the ten selected runs per hypothesis, in seconds except
interval/pair counts. Age is relative to the earliest representative EMA cutoff after the chosen
observation window. Last-timestamp age and last-hypothesized-support age remain distinct. The
ignored exact-view ledger retains ages for every repeated EMA view of selected source bounds.
The full-census JSON also reports age ranges/medians across all qualifying, overlapping views.

{duration_table}

## What the five numbers measure

Mean IBI is average interval length; median IBI is the middle interval length. Sample SDNN
summarizes dispersion of interval values about their mean, using N−1. The conventional name
SDNN does not verify normal beats here. RMSSD is the square root of the average squared change
between original valid adjacent intervals. pNN50 is the percentage of those valid pairs whose
absolute change is **strictly greater than 50 ms**. For one accepted uninterrupted run P=N−1;
no differences are formed across removed records or run boundaries. Undefined values stay missing.

**Synthetic example, not participant data:** intervals [800,900,1000,950,850] ms have mean and
median 900 ms. Squared deviations sum to 25,000; sample SDNN = sqrt(25,000/4) = 79.05694 ms.
Original pair differences [100,100,−50,−100] ms give RMSSD = sqrt(32,500/4) = 90.13878 ms.
Three of four differences exceed 50 ms, so pNN50 = 75%; the exactly 50-ms pair does not count.
This five-interval example demonstrates arithmetic only, not the pilot duration criteria.

The following are aggregate ranges/medians of the **real selected-run** outputs. They are not
clinical reference ranges, stress associations, independent population estimates or EMA features.

{feature_table}

## Package agreement and blocked comparisons

Installed versions are NeuroKit2 0.2.13 and HeartPy 1.2.7, unchanged from Notebook 02; installed
source hashes and signatures match its recorded evidence. The old synthetic-only adapter stays
unchanged. The new guarded entrypoint uses its independent arithmetic and adjacency validator.

HeartPy's lower-level `calc_ts_measures` takes intervals and explicitly supplied adjacent
changes. We bypass waveform processing, RR cleaning, interpolation and high-level `process_rr`.
Convert its population SD by sqrt(N/(N−1)), and its pNN50 fraction to percent. It has no median
output, so median remains independently calculated. See the [official interface documentation](https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/heartpy.analysis.html#heartpy.analysis.calc_ts_measures).

NeuroKit `hrv_time` accepts RRI in ms and actual `RRI_Time` in seconds, but its native pair
handling can differ from our input contract. Start anchoring is blocked; end anchoring requires
every actual pair within 10 ms of the current IBI, and sanitization must preserve both arrays.
No timestamps are changed to make it pass. Its native pNN50 denominator N would be converted
to P=N−1 when a call is allowed. See the [official HRV interface](https://neuropsychology.github.io/NeuroKit/functions/hrv.html#neurokit2.hrv.hrv_time).

{package_table}

Agreement uses relative tolerance 1e−10 and absolute tolerance 1e−8. The package table records
blocked calls without invoking them; detailed reasons are in `package_summary.json` and the
ignored package ledger. Lower-level package internals compute some ancillary time-domain
quantities; only the five authorized outputs are retained, and spectral/nonlinear routines are
never called. Numerical agreement tests arithmetic, units and denominators, not physiology.

## Recommendation and unresolved questions

**Obtain specific provider clarification before expanding the IBI pilot as physiological HRV.**
Earlier runs make a small numerical pilot feasible: the corrected 24-hour census reaches
334 EMAs/51 participants under end and 262 EMAs/40 participants under start, versus one immediate
EMA each. That is about 9.0% / 7.1% of all EMAs, still selective and often hours away from the EMA.
This supports investigating earlier valid segments as a separate direction; it does not validate
24-hour coverage or associations with contemporaneous stress.

Ask the provider whether `ts` denotes acquisition or receipt, whether each IBI is start- or
end-anchored, whether export batching/resolution transforms timestamps, which modality/device
produced the intervals, whether units are confirmed milliseconds, how missed/duplicate beats
are represented, and whether normal-beat/artifact annotations exist. Until answered, retain
both hypotheses and the unverified-normal-beat label. No inference about these semantics is
made from package agreement or retention. A separately specified HR/activity feature effort
can proceed while these IBI questions remain unresolved; it is outside this checkpoint.

## Reproducibility and private artifacts

- Selection, corrected Screen index and all-EMA availability: `{selection.relative_to(root)}/`.
- Run-level values: `{calculation.relative_to(root)}/run_level.parquet` (20 rows).
- All original selected-window members, including rejects: `{calculation.relative_to(root)}/selected_window_members.parquet`.
- Package checks: `{calculation.relative_to(root)}/package_comparisons.parquet` (40 rows).
- Source notebook: `notebooks/03_ibi_variability_pilot.ipynb`; executed copy stays in ignored `outputs/`.

Manifests hash inputs/code and refuse changed or incomplete prior versions. Selection was
published and its hash verified before features. The prior raw-preserving normalized data,
assignments, folds, frozen policies and original bundles remain intact. The verification report
alongside this version records tests, notebook execution, package checks, privacy and integrity.
The corrected Screen index intentionally does not repair historical three-policy reports.
No PSS targets, imputation, final feature linkage, EMA aggregation, models, commit or push occurred.
"""
    (build / "report.md").write_text(report)
    publish_bundle(build, destination, dict(version=version, inputs=inputs, aggregate_only=True))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--calculation", type=Path, required=True)
    args = parser.parse_args()
    print(
        publish(Path.cwd(), args.selection.resolve(), args.calculation.resolve()).relative_to(
            Path.cwd()
        )
    )


if __name__ == "__main__":
    main()
