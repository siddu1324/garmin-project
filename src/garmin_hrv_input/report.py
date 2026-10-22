"""Immutable aggregate reporting, explanatory figures and synthetic package evidence."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_inventory.privacy import AuditError, contains_private_token, sha256
from garmin_inventory.render import table
from garmin_preprocessing.common import artifact_hashes, digest_object, write_json
from .synthetic import validate_all


def figures(folder, data):
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/garmin-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    reasons = pd.DataFrame(data["reasons"])
    r = reasons[
        (reasons.policy == "screen")
        & (reasons.scenario == "interval_end")
        & (reasons.survey_category == "All")
    ]
    r = r[~r.reason.isin(["no_recorded_ibi", "fewer_than_two_contained_intervals"])].sort_values(
        "overlapping"
    )
    labels = {
        "insufficient_unbroken_sequence": "Unbroken sequence <120 s",
        "excessive_internal_gap_ratio": "Internal gap >2 × expected IBI",
        "insufficient_supported_duration": "Acceptable support <270 s",
        "insufficient_edge_support": "Timestamp edge gap >5 s",
        "insufficient_interval_time_agreement": "Consistent pair fraction <80%",
        "duration_elapsed_mismatch": "IBI sum / elapsed span outside range",
        "excessive_artifact_flags": "Provisional jump flags >5%",
        "repeated_timestamp": "Remaining timestamp ties",
        "conflicting_timestamp_values": "Conflicting timestamp values",
        "clock_flag": "Clock flag",
        "implausible_or_missing_ibi": "Implausible or missing IBI",
    }
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 2.2]}, layout="constrained"
    )
    axes[0].barh(
        ["Eligible", "Recorded, rejected", "No recorded IBI"],
        [4, 1506, 2196],
        color=["#2a8864", "#c46c22", "#767676"],
    )
    for i, v in enumerate([4, 1506, 2196]):
        axes[0].text(v + 25, i, f"{v:,}", va="center")
    axes[0].set_xlim(0, 2650)
    axes[0].set_xlabel("Immediate windows (all 3,706 EMAs)")
    axes[0].set_title("Screen / interval end")
    axes[1].barh([labels[x] for x in r.reason], r.overlapping, color="#275d8c")
    for i, row in enumerate(r.itertuples()):
        axes[1].text(
            row.overlapping + 10,
            i,
            f"{row.overlapping:,}; only this: {row.only_this_check}",
            va="center",
            fontsize=8,
        )
    axes[1].set_xlim(0, 1950)
    axes[1].set_xlabel("Overlapping failures among 1,510 recorded windows")
    axes[1].set_title("Reasons overlap and must not be added")
    fig.savefig(
        folder / "rejection_explanation.png", dpi=150, metadata={"Software": "garmin-hrv-input"}
    )
    plt.close(fig)
    c = pd.DataFrame(data["continuity"])
    c = c[
        (c.policy == "screen") & (c.scope == "all_24h_instances") & (c.subset == "eligible")
    ].sort_values("scenario")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
    left = np.zeros(len(c))
    for label, col, color in [
        ("2–<4 min", "span_120_to_240s", "#c46c22"),
        ("4–<4.75 min", "span_240_to_285s", "#275d8c"),
        ("4.75–<5 min", "span_285_to_300s", "#2a8864"),
    ]:
        values = 100 * c[col].to_numpy() / c.windows.to_numpy()
        axes[0].barh(c.scenario, values, left=left, label=label, color=color)
        for i, v in enumerate(values):
            if v > 4:
                axes[0].text(
                    left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", color="white"
                )
        left += values
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Screen-eligible EMA/window instances within 24-hour lookbacks (%)")
    axes[0].set_title("Longest unbroken timestamp span; these are not independent samples")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=9)
    axes[1].broken_barh([(0, 300)], (1, 0.55), facecolors="#d8d8d8")
    axes[1].broken_barh([(0, 120), (180, 120)], (0, 0.55), facecolors="#275d8c")
    axes[1].text(150, 0.27, "60 s gap", ha="center", fontsize=9)
    axes[1].set_yticks([0.27, 1.27], ["Two separate 120 s runs", "300 s observation window"])
    axes[1].set_xlim(0, 300)
    axes[1].set_xlabel(
        "Synthetic illustration: elapsed seconds; no connecting pair crosses the gap"
    )
    axes[1].set_title("A five-minute observation does not make each run five minutes long")
    fig.savefig(
        folder / "continuity_explanation.png", dpi=150, metadata={"Software": "garmin-hrv-input"}
    )
    plt.close(fig)
    v = pd.DataFrame(data["variations"])
    order = [
        "baseline",
        "pair_tolerance_100ms_10pct",
        "pair_tolerance_500ms_50pct",
        "maximum_gap_ratio_3",
        "unbroken_span_60s",
        "unbroken_span_240s",
        "jump_flags_disabled_diagnostic",
        "supported_duration_285s",
    ]
    names = [
        "Frozen Screen",
        "Timing tolerance 100 ms / 10%",
        "Timing tolerance 500 ms / 50%",
        "Maximum gap ratio 3",
        "Minimum unbroken span 60 s",
        "Minimum unbroken span 240 s",
        "Jump flags disabled",
        "Minimum supported duration 285 s",
    ]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    y = np.arange(len(order))
    for i, scenario in enumerate(["interval_end", "interval_start"]):
        g = v[v.scenario == scenario].set_index("variation").loc[order]
        ax.barh(
            y + (i - 0.5) * 0.35,
            g.eligible,
            height=0.35,
            label=scenario,
            color=["#275d8c", "#c46c22"][i],
        )
        for j, value in enumerate(g.eligible):
            ax.text(value + 0.25, y[j] + (i - 0.5) * 0.35, str(value), va="center", fontsize=8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Eligible immediate windows out of 3,706; all other settings held fixed")
    ax.set_title("Diagnostic variations only — frozen policies are not replaced")
    ax.legend(loc="lower right")
    fig.savefig(
        folder / "diagnostic_variations.png", dpi=150, metadata={"Software": "garmin-hrv-input"}
    )
    plt.close(fig)


def publish(root, diagnostics):
    from garmin_preprocessing.common import verify_bundle
    verify_bundle(diagnostics)
    data = json.loads((diagnostics / "aggregate_results.json").read_text())
    policies = json.loads((root / "config/ibi_eligibility.json").read_text())
    synthetic = validate_all(policies["policies"][1])
    sources = {
        str(p.relative_to(root)): sha256(p)
        for p in sorted((root / "src/garmin_hrv_input").glob("*.py"))
    }
    inputs = dict(
        diagnostic_manifest=sha256(diagnostics / "manifest.json"),
        source_sha256=sources,
        specification_sha256=sha256(root / "reports/hrv_input_specification.md"),
        environment_sha256=sha256(root / "uv.lock"),
    )
    version = digest_object(inputs)[:16]
    parent = root / "reports/hrv_input_validation"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / version
    if destination.exists():
        m = json.loads((destination / "manifest.json").read_text())
        if artifact_hashes(destination) != m["artifact_sha256"]:
            raise AuditError("Validated HRV input report changed; no overwrite.")
        return destination
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    for name in [
        "overview",
        "reasons",
        "combinations",
        "variations",
        "continuity",
        "runs",
        "timestamp_precision",
        "pair_timing",
    ]:
        write_json(build / f"{name}.json", data[name])
    write_json(build / "synthetic_validation.json", synthetic)
    summary = {
        k: data[k]
        for k in [
            "version",
            "window_index_rows",
            "immediate_windows",
            "immediate_member_case_rows",
            "immediate_run_rows",
            "real_hrv_calculated",
            "final_test_processed",
            "all_development_ema_accounted",
        ]
    }
    summary.update(diagnostic_bundle=str(diagnostics.relative_to(root)), inputs=inputs)
    write_json(build / "summary.json", summary)
    figures(build, data)
    overview = pd.DataFrame(data["overview"])
    o = overview[overview.survey_category == "All"]
    reason = pd.DataFrame(data["reasons"])
    r = reason[
        (reason.survey_category == "All")
        & (reason.policy == "screen")
        & (reason.scenario == "interval_end")
    ]
    variants = pd.DataFrame(data["variations"])
    continuity = pd.DataFrame(data["continuity"])
    c = continuity[(continuity.policy == "screen") & (continuity.subset == "eligible")]
    run = pd.DataFrame(data["runs"])
    p = data["pair_timing"]
    precision = data["timestamp_precision"]
    matched = sum(x["status"] == "matched" for x in synthetic["comparisons"])
    comparison = []
    for x in synthetic["comparisons"]:
        if x["status"] == "matched" and x["case"] == "known_end":
            comparison.append(
                dict(
                    backend=x["backend"],
                    **{k: v for k, v in x["actual"].items() if k != "warning_categories"},
                )
            )
    report = f"""# HRV input validation and explanation of eligibility losses

**No real-data HRV was calculated.** All analyses use the 125 development participants and the
existing immutable bundles. All 32 final-test participants remain sealed. No PSS targets, imputation,
feature linkage, models, commit or push were performed. Frozen policies and assignments are unchanged.

## Verified development findings

Of **3,706 immediate five-minute windows, 2,196 (59.3%) have no recorded IBI**; 1,510 contain records.
Under Screen/end, 1,506 of those recorded windows fail and four pass. Seven recorded windows have
fewer than two contained plausible intervals. Under Screen/start the corresponding counts are 1,505
failures, five passes and ten with fewer than two contained intervals. Recorded but insufficient
windows are a subset of recorded failures, not an additional attrition stage.

{table(o.to_dict("records"), ["policy", "scenario", "ema_windows", "no_recorded_ibi", "recorded_rejected", "recorded_fewer_than_two_contained", "eligible", "eligible_participants"])}

![Rejection explanation](rejection_explanation.png)

The principal losses among recorded Screen/end windows are **unbroken span <120 s (1,469)**,
**an internal gap >2× expected IBI (1,409)** and **acceptable support <270 s (926)**. Only 86 fail the
80% pair-agreement fraction, but *each* inconsistent pair can still break a run even if the overall
fraction passes. Therefore a high percentage of matching pairs does not imply a long intact sequence.
The two most frequent recorded-failure combinations are gap + support + run length (636), and
gap + run length (527). Reasons overlap and must not be summed.

{table(r.to_dict("records"), ["reason", "overlapping", "only_this_check", "pass_if_only_gate_waived"])}

Only 31 windows fail solely the gap-ratio gate, and 21 solely the run-span gate. No other recorded
Screen/end window fails only one gate. The gate-waiver column removes a final decision bit without
changing interval membership, support or run construction; it is a logical dependency diagnostic,
**not permission to use those windows**. It always requires recorded IBI; absence is never waived.
Complete reason overlaps, combinations and survey categories are in the companion JSON tables.

## Diagnostic changes, not replacement policies

The separate diagnostic configuration was saved before its comparisons. These are single-check or
single-mechanism sensitivities around Screen, never selected using outcomes or a retention target.
Timing-tolerance changes also alter which pairs break runs; jump-threshold changes recompute flagged
intervals, support and runs. Raising a final allowed flag fraction alone does not undo exclusions.

{table(variants.to_dict("records"), ["variation", "scenario", "eligible", "eligible_participants"])}

![Diagnostic variations](diagnostic_variations.png)

Disabling the provisional >20% jump flags leaves retention at **4/end and 5/start**. It is not the
main explanation for the severe loss, and those flags do not prove artifact or ectopy. Widening the
pair tolerance to 500 ms/50% retains 20/end and 22/start; raising the gap ratio to three retains
26/end and 20/start. Neither result validates a wider rule or the acquisition clock. Reducing the
minimum run to 60 s retains 16/end and 19/start, illustrating the duration tradeoff, not recommending it.
Plausibility, conflict and edge failures matter locally even when changing their gate cannot rescue
windows that fail several other checks.

## Precision, batching and clocks

The original IBI timestamp strings have mostly six or seven nonzero fractional decimal places:
**{precision.get("raw_ts_fraction_digits_6", 0) + precision.get("raw_ts_fraction_digits_7", 0):,} of 34,427,546 physical IBI rows**.
Only {precision.get("raw_ts_fraction_digits_0", 0):,} physical timestamps have no nonzero fractional part.
This rules out a simple whole-second export representation as the universal explanation. Decimal
precision is not measurement accuracy or proof of acquisition timing.

The logical development stream contains **{p["compressed_pairs"]:,} compressed positive-time pairs**
spaced less than one quarter of both adjacent IBI durations, {p["zero_timestamp_pairs"]:,} timestamp ties,
{p["gaps_over_5s"]:,} gaps above five seconds and {p["gaps_over_60s"]:,} above one minute. These patterns
are compatible with delayed or batched receipt, missing records or exporter behavior; they do not
identify the mechanism. No timestamp was shifted, rounded, reconstructed or concatenated.

As a precision-bound diagnostic, widening an *absolute pair comparison* from 250 to 251 ms changes
end-hypothesis matching pairs from {p["end_pairs_within_250ms"]:,} to {p["end_pairs_within_251ms"]:,}
(out of {p["positive_plausible_pairs"]:,} positive plausible pairs). This is not a policy rerun or a
retiming operation. Millisecond quantization alone would not account for the much larger mismatches;
clock jitter or batching can be much larger than the displayed precision. Only
{p["end_pairs_within_10ms"]:,} pairs match current IBI within NeuroKit's 10 ms native criterion.

## Observation duration versus uninterrupted input

A five-minute window has 300 seconds of calendar duration. Its acceptable supported duration is a
union of hypothesized intervals with holes preserved. Its longest provisional run is a separate
first-to-last timestamp span. Neither should be substituted for the other.

{table(c.to_dict("records"), ["scenario", "scope", "windows", "span_120_to_240s", "span_240_to_285s", "span_285_to_300s"])}

![Continuity explanation](continuity_explanation.png)

Within 24-hour lookbacks, **16,247/18,964 Screen/end eligible window instances** and
**13,161/14,681 Screen/start instances** have longest runs below four minutes. These are overlapping
EMA/window instances, not independent segments. A two-minute run may pass Screen; its future
calculation must be labeled by the actual run duration and cannot be called five-minute HRV.

The [input specification](../../hrv_input_specification.md) defines membership, run breaks, package
conditions and formulas. Every approved window has an ignored immutable slice index; the original
values and quality flags remain in normalized Parquet. The reusable loader hydrates any window.
Immediate-window records and run boundaries are additionally materialized for review:
**{data["window_index_rows"]:,} window-index rows, {data["immediate_member_case_rows"]:,} member/policy/scenario rows,
and {data["immediate_run_rows"]:,} run rows**. Repeated case rows are reference views, not new beats.
Private diagnostics: `{diagnostics.relative_to(root)}`.

{table(run.to_dict("records"), ["policy", "scenario", "runs", "allowed_adjacent_pairs", "native_neurokit_compatible_pairs", "pilot_candidate_runs"])}

## Verified synthetic package behavior

Installed versions: NeuroKit2 **0.2.13**, HeartPy **1.2.7**, NumPy **2.2.6**, SciPy **1.17.1**.
Their installed source/docstrings and signatures were inspected and hashed. The environment was
extended in place; the existing direct dependency pins remain unchanged. A private copy of the
prior lockfile was preserved. Adding package dependencies changes the current lock hash; this
checkpoint explicitly reuses existing bundles and does not regenerate normalization.

**{synthetic["cases"]} synthetic cases produced {len(synthetic["comparisons"])} run/backend checks:
{matched} matched independent reference arithmetic and {len(synthetic["comparisons"]) - matched} were deliberately blocked**
because NeuroKit native pair rules or timestamp anchoring did not match the contract. Cases cover
known intervals, constants, exact 50 ms differences, jitter, gaps, removed/rejected records,
remaining duplicates, conflicts and both window-boundary hypotheses. The adapters accept synthetic
fixtures only and refuse changed clocks, multi-run concatenation or new adjacent pairs.

For the known synthetic sequence **800, 900, 1000, 950, 850 ms**, mean and median are 900 ms.
Sample SDNN is sqrt(25,000/4) = **79.056942 ms**. The differences are +100, +100, −50, −100 ms;
RMSSD is sqrt(32,500/4) = **90.138782 ms** and pair-denominator pNN50 is **75%** (three of four,
strictly greater than 50 ms). These are synthetic illustration values only.

{table(comparison, ["backend", "mean_ibi_ms", "median_ibi_ms", "sdnn_sample_ms", "rmssd_ms", "pnn50_pairs_percent"])}

HeartPy's native SDNN uses population variance (70.710678 ms in this example); convert by
sqrt(N/(N−1)). Its pNN50 is 0.75 as a proportion; multiply by 100. It exposes no median IBI result.
NeuroKit uses sample SDNN but native pNN50 is 60% here, dividing by P+1 instead of P. The adapters
perform explicit documented conversions only after pair identity is verified.

NeuroKit's explicit `RRI`/`RRI_Time` interface is timestamp-aware, but native succession uses current
IBI and 10 ms tolerance. Synthetic jitter with **some** matching pairs triggers pair filtering;
jitter with **no** matching pairs can bypass that filter and use all differences. Its apparent
success is therefore not evidence of valid input. Negative-control results are in
`synthetic_validation.json`; they are not approved gappy-input adapters. Start-anchored timestamps
are not silently shifted into end times to satisfy the package.

HeartPy's high-level `process_rr` is not used: it lacks timestamps and attempts Poincaré/breathing
calculations even with frequency output disabled. Its lower-level `calc_ts_measures` accepts the
externally validated absolute differences and does not correct/interpolate data. No waveform,
spectral, breathing or nonlinear function was used. NeuroKit's other time-domain outputs are not
validated or retained. HeartPy emits a deprecated `pkg_resources` warning; small/constant synthetic
cases can emit numerical warnings for unused ancillary NeuroKit summaries. Warnings are recorded;
the specified outputs are explicitly checked.

## Provisional assumptions and provider questions

Verified facts above concern exports, code and arithmetic. IBI millisecond units, sensor modality,
beat/interval interpretation, clock accuracy, normal-beat status and the frozen numerical screens
remain provisional. Only the provider can establish whether `ts` is acquisition, buffering or
receipt time; whether it anchors interval start/end; how batching and missing beats are represented;
and whether quality or beat-type annotations exist. No retention comparison answers those questions.
The >20% change flag is not a validated artifact detector.

## Concrete next pilot

The arithmetic and single-run guards are ready for **mean/median IBI, sample-convention SDNN,
RMSSD and pair-denominator pNN50**, conditional on a documented interpretation of the source.
A next development-only pilot should select at most ten unique nonoverlapping runs, at most one per
participant, from frozen Screen-eligible windows with **run span ≥240 s AND support ≥240 s**.
Use deterministic chronological selection without outcomes or HRV values. Label actual run duration;
keep both timestamp hypotheses separate. The immediate windows contain only **one window meeting
that pilot restriction under both hypotheses**; do not loosen rules to reach ten. Longer lookbacks
can be inspected via the same indexed contract, with overlap deduplication before selection.

Use HeartPy's validated lower-level time-domain call or the independent formulas with externally
verified adjacency. Use NeuroKit for pair-based cross-checks only when interval-end interpretation
and its exact pair set agree. Until normal-beat status is established, report IBI variability with
NN status unverified. No real pilot values were calculated here. Spectral and additional nonlinear
features remain deferred; Garmin stress derivatives and quality diagnostics remain outside initial
predictors.

Sources: [NeuroKit API/source](https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/hrv/hrv_time.html),
[HeartPy time-domain source](https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/_modules/heartpy/analysis.html),
[HeartPy RR pipeline](https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/_modules/heartpy/heartpy.html).
Installed source hashes in `synthetic_validation.json` identify the tested implementation.
"""
    (build / "report.md").write_text(report)
    tokens = json.loads((root / "private/privacy_tokens.json").read_text())
    for p in build.iterdir():
        if p.suffix in {".md", ".json"} and contains_private_token(
            p.read_text(), tokens["identifiers"], tokens["names"]
        ):
            raise AuditError(
                "Aggregate HRV input report privacy scan flagged a known token; details withheld."
            )
    write_json(
        build / "manifest.json",
        dict(version=version, inputs=inputs, artifact_sha256=artifact_hashes(build)),
    )
    build.rename(destination)
    return destination


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = Path.cwd()
        result = publish(root, args.diagnostics.resolve())
        write_json(
            root / "private/hrv_input_validation/current_run.json",
            dict(
                diagnostics=str(args.diagnostics.resolve().relative_to(root)),
                report=str(result.relative_to(root)),
            ),
        )
        print("Published aggregate input validation:", result.relative_to(root))
    except Exception as exc:
        print(
            str(exc)
            if isinstance(exc, AuditError)
            else f"Input report failed ({type(exc).__name__}); private details withheld."
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
