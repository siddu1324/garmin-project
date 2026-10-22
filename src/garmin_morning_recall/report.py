"""Aggregate-only retrospective experiment report and teaching figures."""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from garmin_ema_baseline.pipeline import records
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle, write_json
from garmin_recall_diagnostics.report import table

TABLES = [
    "availability",
    "cohorts",
    "fold_counts",
    "composition",
    "score_distribution",
    "exclusion_combinations",
    "quality_diagnostics",
    "metrics",
    "paired_intervals",
]
LABEL = {
    "reported_minute_start": "Reported minute start",
    "reported_minute_end_sensitivity": "Later-minute sensitivity",
}
MODELS = {
    "mean": "Mean dummy",
    "median": "Median dummy",
    "context": "Time context",
    "wake": "Context + wake HR",
    "fixed_60": "Context + 60m HR",
    "fixed_5": "Context + 5m HR",
    "fixed_120": "Context + 120m HR",
}


def render(root, bundle):
    root, bundle = Path(root), Path(bundle)
    manifest = verify_bundle(bundle)
    inputs = {
        "bundle_manifest": sha256(bundle / "manifest.json"),
        "report_code": sha256(Path(__file__)),
        "table_helper_code": sha256(root / "src/garmin_recall_diagnostics/report.py"),
    }
    version = digest_object(inputs)[:16]
    parent = root / "reports/morning_recall"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / version
    if destination.exists():
        verify_bundle(destination)
        return destination
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    frames = {name: pd.read_json(bundle / (name + ".json")) for name in TABLES}
    for name, data in frames.items():
        write_json(build / (name + ".json"), records(data))
    nominal = "reported_minute_start"
    pooled = frames["metrics"].loc[frames["metrics"].fold.eq("pooled")].copy()
    primary = pooled.loc[pooled.scenario.eq(nominal) & pooled.cohort.eq("pair_fixed_60")]
    direct = frames["paired_intervals"].loc[
        frames["paired_intervals"].model.eq("wake")
        & frames["paired_intervals"].baseline.str.startswith("fixed")
    ]
    baselines = frames["paired_intervals"].loc[
        frames["paired_intervals"].cohort.eq("pair_fixed_60")
        & frames["paired_intervals"].model.isin(["wake", "fixed_60"])
        & ~frames["paired_intervals"].baseline.str.startswith("fixed")
    ]
    for name, frame in [
        ("pooled_metrics", pooled),
        ("fold_metrics", frames["metrics"].loc[frames["metrics"].fold.ne("pooled")]),
        ("direct_window_differences", direct),
        ("primary_baseline_differences", baselines),
        ("included_excluded_scores", frames["composition"]),
        ("all_fold_counts", frames["fold_counts"]),
    ]:
        (build / (name + ".md")).write_text(
            "# Retrospective Morning comparison: "
            + name.replace("_", " ")
            + "\n\n"
            + table(frame)
            + "\n"
        )
    accounting = pd.read_parquet(bundle / "accounting.parquet")
    unique = accounting.loc[accounting.scenario.eq(nominal)]
    totals = {
        "responses": len(unique),
        "participants": unique.n_participant.nunique(),
        "valid_scores": int(unique.score_available.sum()),
        "missing_scores": int((~unique.score_available).sum()),
        "boundary_reasons": {
            scenario: part.boundary_reason.value_counts().to_dict()
            for scenario, part in accounting.groupby("scenario")
        },
    }
    write_json(build / "morning_accounting_totals.json", totals)
    # Availability stacks retain every Morning EMA. Complete features do not imply full coverage.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = ["#a9a9a9", "#dedede", "#d58a70", "#e4c476", "#447a99"]
    for ax, scenario in zip(axes, LABEL):
        a = (
            frames["availability"]
            .loc[frames["availability"].scenario.eq(scenario)]
            .set_index("interval")
            .loc[["wake", "fixed_60", "fixed_5", "fixed_120"]]
        )
        bottom = np.zeros(4)
        for col, label, color in zip(
            [
                "boundary_unavailable",
                "no_recording",
                "all_readings_rejected",
                "insufficient_feature_support",
                "complete_hr",
            ],
            [
                "Unavailable boundary",
                "No recording",
                "All readings rejected",
                "Insufficient feature support",
                "Four HR features available",
            ],
            colors,
        ):
            ax.bar(np.arange(4), a[col], bottom=bottom, label=label, color=color)
            bottom += a[col].to_numpy()
        ax.set(
            xticks=np.arange(4),
            xticklabels=["Wake", "60m", "5m", "120m"],
            ylabel="Morning responses (all 934)",
            title=LABEL[scenario],
        )
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.savefig(build / "availability.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    order = ["median", "mean", "context", "wake", "fixed_60"]
    values = primary.set_index("model").loc[order]
    axes[0].bar(
        np.arange(5), values.mae, color=["#949494", "#b7b7b7", "#679184", "#c56f3c", "#467fa6"]
    )
    axes[0].set(
        xticks=np.arange(5),
        xticklabels=[MODELS[k] for k in order],
        ylabel="Pooled OOF MAE, composite points",
        title="Primary comparison: same 227 responses / 86 participants",
    )
    axes[0].tick_params(axis="x", labelrotation=30)
    for offset, scenario, color in [
        (-0.07, nominal, "#c56f3c"),
        (0.07, "reported_minute_end_sensitivity", "#467fa6"),
    ]:
        d = (
            direct.loc[direct.scenario.eq(scenario) & direct.cohort.ne("common")]
            .set_index("cohort")
            .loc[["pair_fixed_60", "pair_fixed_5", "pair_fixed_120"]]
        )
        axes[1].errorbar(
            d.mae_difference,
            np.arange(3) + offset,
            xerr=np.vstack(
                [d.mae_difference - d.mae_difference_low, d.mae_difference_high - d.mae_difference]
            ),
            fmt="o",
            capsize=4,
            color=color,
            label=LABEL[scenario],
        )
    axes[1].axvline(0, color="gray", linewidth=1)
    axes[1].set(
        yticks=np.arange(3),
        yticklabels=["vs 60m (primary)", "vs 5m (secondary)", "vs 120m (secondary)"],
        xlabel="Wake minus fixed OOF MAE; negative favors wake",
        title="Paired participant bootstrap; conditional 95% intervals",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(build / "performance.png", dpi=160)
    plt.close(fig)
    q = (
        frames["quality_diagnostics"]
        .loc[
            frames["quality_diagnostics"].scenario.eq(nominal)
            & frames["quality_diagnostics"].cohort.eq("pair_fixed_60")
            & frames["quality_diagnostics"].group.eq("included")
        ]
        .set_index("interval")
        .loc[["wake", "fixed_60"]]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, col, scale, title in [
        (axes[0], "occupied_bin_fraction", 100.0, "Accepted-reading minute occupancy"),
        (
            axes[1],
            "maximum_gap_including_edges_seconds",
            1 / 60,
            "Largest gap, including window edges",
        ),
    ]:
        med, low, high = [
            q[col + "_" + suffix].to_numpy() * scale for suffix in ["median", "q25", "q75"]
        ]
        ax.errorbar(np.arange(2), med, yerr=np.vstack([med - low, high - med]), fmt="o", capsize=8)
        ax.set(
            xticks=np.arange(2),
            xticklabels=["Wake→cutoff", "Fixed 60m"],
            title=title,
            ylabel="Percent: median and IQR" if scale == 100 else "Minutes: median and IQR",
        )
    fig.savefig(build / "recording_support.png", dpi=160)
    plt.close(fig)
    # Entirely synthetic timeline: no participant clock or trace is plotted.
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    for y, (start, label, color) in enumerate(
        [
            (60, "Wake (07:00 → 08:09)", "#c56f3c"),
            (69, "Primary 60m (07:09 → 08:09)", "#467fa6"),
            (124, "Secondary 5m (08:04 → 08:09)", "#679184"),
            (9, "Secondary 120m (06:09 → 08:09)", "#949494"),
        ]
    ):
        ax.plot([start, 129], [y, y], linewidth=5, color=color)
        ax.scatter([start], [y], color=color, zorder=3)
        ax.scatter([129], [y], facecolor="white", edgecolor=color, zorder=3)
        ax.text(
            129 if start > 120 else start,
            y + 0.13,
            label,
            fontsize=9,
            ha="right" if start > 120 else "left",
        )
    ax.axvline(129, color="black", linestyle="--", label="Cutoff 08:09 (right edge excluded)")
    ax.axvline(130, color="black", linestyle=":", label="Actual start 08:10, minute precision")
    ax.set(
        xlim=(0, 143),
        ylim=(-0.35, 3.7),
        yticks=[],
        xticks=[0, 30, 60, 90, 120, 129],
        xticklabels=["06:00", "06:30", "07:00", "07:30", "08:00", "08:09"],
        title="Synthetic timeline — current-response wake defines a retrospective interval",
    )
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(build / "synthetic_timeline.png", dpi=160)
    plt.close(fig)
    primary_composition = frames["composition"].loc[
        frames["composition"].scenario.eq(nominal)
        & frames["composition"].cohort.eq("pair_fixed_60")
    ]
    primary_folds = frames["fold_counts"].loc[
        frames["fold_counts"].scenario.eq(nominal)
        & frames["fold_counts"].cohort.eq("pair_fixed_60")
    ]
    body = f"""# Notebook 08 — retrospective Morning recall-window comparison

**Wake-to-cutoff HR did not show a useful predictive advantage over the specified fixed windows.**
In the primary matched cohort, nominal wake and 60-minute context+HR models have almost identical
OOF MAE (2.2506 vs 2.2510). Both have higher point-estimate error than matching median and context
baselines. This is a development-only, retrospective exploratory experiment; it does not validate
a deployable model, prove absence of physiological signal, or identify an optimal window.

## Specification and what enters the model

The [specification](../../morning_recall_specification.md) was frozen before new performance:
primary wake versus 60 minutes; secondary wake versus 5 and 120 minutes. Both documented wake
scenarios are separate. [Notebook 07's local questionnaire citations](../../recall_alignment/824551e40b8e301d/questionnaire_evidence.md)
establish Morning's **since waking up** wording. The current-survey wake answer defines only the
retrospective left boundary. It is never a predictor or information assumed known before the response.

Reuse the unchanged continuous three-item composite, actual-start cutoff and precision buffer,
HR acceptance and four features: observed-reading mean, sample SD (N−1), latest and elapsed-minute
OLS slope. Mean/SD need ≥2 distinct accepted timestamps spanning ≥1 second; latest needs one;
slope needs ≥3 spanning ≥60 seconds. Accepted HR retains the provisional 30–240 bpm range, clock
flags and entire conflict-group rejection; identical timestamp/value groups count once. No filled
values, extra readings or support relaxation. These are summaries of observed readings, not a
claim that a window was continuously measured.
Mean and SD weight readings equally, not elapsed duration, so densely recorded bursts receive more weight.

All windows are [left,cutoff). Minute-resolution starts retain the 60-second buffer. Wake uses the
reported local date and EMA offset as already documented; no fabricated prior date or clock shift.
The later-minute sensitivity moves only the wake left boundary by 60 seconds; it is not a bound
on recall error. Same-day/offset and acquisition-versus-receipt semantics remain provisional.

Time context is sin/cos of actual local survey start. The existing category encoder sees only
Morning, so it adds no category contrast. All scalers/encoders fit on training folds only. Ridge
uses alpha=1, solver=svd, with intercept; its squared-error objective differs from MAE evaluation.
[Official Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
and [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
document the reused interfaces; the installed version is recorded in the private manifest.
No quantile regression, new alpha, HR-only model, quality predictor or current-answer predictor.

## Accounting: availability is part of the result

All **934 Morning responses / 125 development participants** are preserved per scenario: **931
valid scores and three missing**, without outcome imputation. Nominal wake has 836 positive
boundaries (three missing wake answers; 95 at/after cutoff). The later-minute scenario has 826;
10 further candidates become nonpositive. All these rows remain in accounting.

{table(frames["availability"][["scenario", "interval", "responses", "boundary_unavailable", "no_recording", "all_readings_rejected", "any_accepted_hr", "insufficient_feature_support", "complete_hr", "complete_hr_participants"]])}

Nominal wake's 253 responses with any accepted HR reduce to **237 with all four HR features / 87
participants**; 16 have insufficient feature support. Later-minute wake has 252 with any accepted
HR but the same 237 complete-feature responses. Feature-available rows happen to have complete
scores; the three missing scores remain in the excluded accounting. Reasons can overlap, and the
[exclusion combinations](exclusion_combinations.json) partition each comparison without double-counting.

![All-Morning availability](availability.png)

## Matched cohorts and participant folds

Every model within a comparison is refit on the identical training response keys and evaluated on
the identical held-out keys. Mean/median dummies and context do not borrow a larger cohort.
The two scenarios happen to retain identical comparison memberships; their wake features can differ.
The all-window common cohort coincides with the 5-minute pairwise cohort; these are repeated views
of the same observations, not independent replications. The common cohort supplements the primary.

{table(frames["cohorts"][["scenario", "cohort", "responses", "participants", "excluded_responses", "eligible_without_oof", "limited_interpretation"]])}

Primary retention is **227/934 = 24.3% of Morning responses**, from 86 participants. The 5-minute
comparison/common cohort contains 191 responses / 74 people; the 120-minute comparison 233 / 87.
All original five folds are feasible and no folds are redrawn. The inherited flags (<100 responses,
<30 people or <5 validation people in any fold) do not trigger, but the cohorts remain modest and
selected. Never compare raw errors across these different pairwise populations as a window ranking.

{table(primary_folds[["fold", "training_responses", "training_participants", "validation_responses", "validation_participants", "excluded_validation_responses", "status"]])}

{table(primary_composition[["group", "responses", "participants", "valid_scores", "missing_scores", "mean", "median", "q25", "q75"]])}

Included primary responses have mean score 6.449 (median 6), versus 6.723 (median 7) among the 704
scored excluded responses. This is descriptive selection, not a basis to change support rules.
Participants can have both included and excluded responses; their counts cannot be added.
[All included/excluded distributions](included_excluded_scores.md), [score frequencies](score_distribution.json)
and [all fold counts](all_fold_counts.md) cover both scenarios and all comparisons.

## Primary performance and uncertainty

MAE is primary; RMSE secondary. Equal-participant MAE averages each person's mean absolute error.
Equal-participant RMSE averages each person's within-person RMSE, rather than taking the pooled root.
All values below are fixed OOF predictions on the **same 227 responses**.

{table(primary[["model", "responses", "participants", "mae", "rmse", "participant_mae", "participant_rmse"]])}

Nominal wake minus 60-minute MAE is **−0.0003 points (95% interval −0.0275 to +0.0263)**;
the equal-participant MAE difference is −0.0053 (−0.0407 to +0.0274). Wake versus median dummy
is +0.1097 (−0.0531 to +0.3042), and wake versus context is +0.0649 (−0.0514 to +0.2359).
Point estimates provide no wearable gain beyond those baselines; uncertainty does not establish
harm or equivalence. The later-minute primary difference is −0.0093 (−0.0495 to +0.0270), with
unchanged membership. This sensitivity does not change the practical conclusion.

{table(direct[["scenario", "cohort", "baseline", "responses", "participants", "mae_difference", "mae_difference_low", "mae_difference_high", "participant_mae_difference"]])}

All direct wake-versus-fixed intervals include zero. Nominal pairwise differences for secondary
5 and 120 minutes are +0.0183 and −0.0068 points respectively. The common-cohort results also fail
to establish a wake advantage. These modest differences do not justify selecting a window or
searching more thresholds. See [all pooled results](pooled_metrics.md), [fold-level errors](fold_metrics.md),
[primary baseline contrasts](primary_baseline_differences.md) and [all paired intervals](paired_intervals.json).

Intervals use 2,000 paired participant-cluster bootstrap draws, seed 20261002, keeping all fixed
predictions for each sampled person together. They condition on this exploratory workflow, its
selected cohorts and fixed development fits. They do not include refitting/model-selection
uncertainty, correct for multiple comparisons, establish external validity, or treat overlapping
EMAs as independent. No winner, significance claim or deployable model is declared.

![Matched primary errors and direct differences](performance.png)

## Why timing alignment alone may be insufficient

On the primary included cohort, median wake duration is **115 minutes**, but median occupied-minute
fraction is only **48.1%**, versus **58.3%** for 60-minute windows. Median largest gap including edges
is **39.4 versus 20.0 minutes**; median internal gaps are 22.1 versus 17.1 seconds. Both windows have
median latest-reading age about 0.93 seconds, which does not erase long earlier/edge gaps. Median
left-edge gaps are 35.5 versus 7.7 minutes. Three readings and a long span do not establish coverage.
Quality is descriptive only; none of these quantities, wake duration or missingness is a predictor.
[Quality distributions](quality_diagnostics.json) include both scenarios, included/excluded groups,
all windows, quartiles and nonmissing denominators.

![Primary matched recording support](recording_support.png)

## Synthetic teaching example — no participant records

A minute-resolution survey starts at 08:10 and reports waking at 07:00. Cutoff is 08:09. Wake
uses [07:00,08:09); 60m uses [07:09,08:09); 5m uses [08:04,08:09); 120m uses [06:09,08:09).
With offset −240 minutes, add four hours to express these clocks in UTC. A 120-minute window can
include pre-wake data. The later-minute wake sensitivity starts at 07:01. A reading at 08:09 is
excluded. The current wake answer remains retrospective in every case.

![Synthetic interval timeline](synthetic_timeline.png)

For synthetic accepted readings 60, 70 and 80 bpm at elapsed 0, 1 and 2 minutes, mean=70 bpm,
sample SD=10 bpm, latest=80 bpm and slope=10 bpm/minute. If only the last reading lies in another
window, latest exists but the other features do not; that EMA cannot enter that paired comparison.
An EMA with three valid stored items (4,2,3) has the existing composite 4+(6−2)+(6−3)=11. Its
held-out prediction must come from a model trained on other participants' folds, using the same
matched response cohort as every comparator. All teaching numbers are synthetic.

## Professor-facing conclusion and next justified work

The specified retrospective wake alignment does **not currently support a better HR baseline**.
A conceptual recall match did not translate into useful held-out error reduction, and HR models
remain worse by point estimate than matching simple baselines. This does not show that stress has
no physiological correlate; recording support, self-reported boundaries and event-clock semantics
still limit the measurement. Keep the common cohort and later-minute sensitivity as checks, not
independent confirmations or opportunities to select a favorable result.

Do not expand a window/alpha search on these results. The next justified work is measurement
clarification: obtain acquisition/receipt and batching definitions, confirm survey start and wake
report date/offset semantics, and determine whether available device-quality metadata can explain
recording gaps. A stronger measurement/coverage protocol would need a separate specification and
fresh evaluation; no data-provider message was sent and no further experiment was run here.

## Artifacts and safeguards

Private immutable bundle: `{bundle.relative_to(root)}`.
`features.parquet` and `accounting.parquet`: 1,868 rows each (934 × two scenarios).
`quality.parquet` and `membership.parquet`: 7,472 rows each. `out_of_fold_predictions.parquet`:
1,684 matched comparison views, not independent responses. `lineage.parquet` retains wake source
slices; `training_audit.json` retains matched key hashes, fitted scalers/encoders, coefficients and
matching dummy constants. Public artifacts contain only aggregate tables/figures and synthetic examples.
All 32 final-test participants remain sealed; prior outcomes/features, raw data, assignments, folds
and immutable bundles are preserved. No HRV, Garmin stress, imputation, quality predictor, binary
cutoff, nonlinear model, commit or push. See the separate checkpoint verification for executed checks.
"""
    (build / "report.md").write_text(body)
    briefing = """# Professor summary — retrospective Morning comparison

**Result:** reported wake-to-cutoff HR does not show a useful advantage over the specified fixed
windows. This is an exploratory development experiment using current-survey wake reports
retrospectively; it is not a deployable prediction model.

The primary wake-versus-60-minute comparison uses the same 227 responses from 86 participants,
with identical training and validation memberships in the original five participant folds. OOF
MAE is 2.2506 for context+wake HR and 2.2510 for context+60m HR. Paired difference: −0.0003 composite
points (conditional 95% participant-bootstrap interval −0.0275 to +0.0263). Matching median dummy
and context-only MAE are lower by point estimate: 2.1410 and 2.1858. No interval establishes a gain.

Secondary 5- and 120-minute comparisons also fail to establish a wake advantage. Later-minute wake
sensitivity leaves comparison membership unchanged and has primary difference −0.0093
(−0.0495 to +0.0270). The common cohort is the same 191 responses / 74 participants as the
5-minute pairwise cohort; these are not independent replications.

Only 24.3% of all 934 Morning responses enter the primary comparison. All exclusions remain
accounted for, including three missing scores and invalid wake boundaries. Nominal wake has
253 responses with any HR, but only 237 with all four required features. Among primary included
responses, median wake-window minute occupancy is 48.1% and median largest edge-inclusive gap
39.4 minutes. A fresh final reading does not establish full recording coverage. Quality, wake
clock/duration and current answers never enter predictors.

**Recommendation:** do not search for an optimal window or tune models from these results.
Prioritize provider clarification of acquisition versus receipt/batching, start-event definitions,
wake date/timezone reporting and device-quality metadata. Any revised measurement protocol needs
a separately specified evaluation. The observed differences are too small/uncertain to support
window selection; they do not rule out physiological signal under better measurement.

All 32 final-test people remain sealed. No HRV, stress predictor, imputation, nonlinear model,
commit or push. Questions are not sent. See [report](report.md), [full matched errors](pooled_metrics.md)
and [conditional paired intervals](paired_intervals.json).
"""
    (build / "professor_summary.md").write_text(briefing)
    publish_bundle(
        build,
        destination,
        {
            "version": version,
            "inputs": inputs,
            "aggregate_only": True,
            "retrospective_only": True,
            "source_run": manifest["version"],
        },
    )
    return destination
