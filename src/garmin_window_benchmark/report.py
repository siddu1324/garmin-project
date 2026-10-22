"""Aggregate-only teaching report and figures, immutable and separately versioned."""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle
from garmin_recall_diagnostics.report import table


def render(root, bundle):
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    verify_bundle(bundle)
    inputs = {
        "bundle_manifest": sha256(bundle / "manifest.json"),
        "renderer": sha256(Path(__file__)),
    }
    identifier = digest_object(inputs)[:16]
    parent = root / "reports/window_benchmark"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / identifier
    if destination.exists():
        verify_bundle(destination)
        return destination
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    names = [
        "window_overlap",
        "feature_differences",
        "density_summary",
        "metrics",
        "paired_intervals",
        "fold_counts",
        "selected_settings",
        "composition",
    ]
    frames = {n: pd.read_json(bundle / (n + ".json")) for n in names}
    for n in names:
        (build / (n + ".json")).write_bytes((bundle / (n + ".json")).read_bytes())
        (build / (n + ".md")).write_text(table(frames[n]) + "\n")
    for n in ["source_verification", "synthetic_positive_control", "reference_agreement"]:
        (build / (n + ".json")).write_bytes((bundle / (n + ".json")).read_bytes())
    overlap = frames["window_overlap"]
    nominal = overlap[
        (overlap.scenario == "reported_minute_start") & (overlap.cohort == "pair_fixed_60")
    ]
    metrics = frames["metrics"]
    pooled = metrics[
        (metrics.stage == "oof") & (metrics.fold == "pooled") & (metrics.survey_category == "All")
    ]
    density = frames["density_summary"]
    ds = density[
        (density.scenario == "reported_minute_start") & (density.cohort == "pair_fixed_60")
    ]
    diff = frames["feature_differences"]
    fd = diff[
        (diff.scenario == "reported_minute_start")
        & (diff.cohort == "pair_fixed_60")
        & (diff.window_a == "wake")
        & (diff.window_b == "fixed_60")
    ]
    plt.rcParams.update(
        {"font.size": 10, "figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(nominal))
    width = 0.26
    for i, (col, label) in enumerate(
        [
            ("identical_sets", "Identical source sets / all responses"),
            ("identical_latest_identity", "Same latest source / all responses"),
            ("near_identical_vector", "Near vectors / complete vectors"),
        ]
    ):
        denominator = (
            nominal.complete_vectors if col == "near_identical_vector" else nominal.responses
        )
        ax.bar(x + (i - 1) * width, 100 * nominal[col] / denominator, width, label=label)
    ax.set_xticks(
        x,
        [
            a.replace("fixed_", "") + " vs " + b.replace("fixed_", "")
            for a, b in zip(nominal.window_a, nominal.window_b)
        ],
    )
    ax.set_ylabel("Percent of stated denominator")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.21))
    fig.suptitle("Matched Morning inputs: repeated information across windows", y=1.08)
    fig.tight_layout()
    fig.savefig(build / "window_information.png", bbox_inches="tight")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
    x = np.arange(len(ds))
    axs[0].bar(x, 100 * ds.dense_decile_reading_share_median, color="#36827f")
    axs[0].set_ylabel("Median share of accepted readings (%)")
    axs[0].set_title("Densest 10% of occupied bins")
    axs[1].bar(x, ds.absolute_weighting_difference_median, color="#946846")
    axs[1].plot(x, ds.absolute_weighting_difference_q95, "ko", label="95th percentile")
    axs[1].legend()
    axs[1].set_ylabel("Absolute difference in HR mean (bpm)")
    axs[1].set_title("Reading mean vs observed-minute mean")
    for ax in axs:
        ax.set_xticks(x, ds.window.str.replace("fixed_", "", regex=False))
    fig.suptitle("Descriptive weighting audit; empty minutes stay missing")
    fig.tight_layout()
    fig.savefig(build / "density_weighting.png")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, block in zip(axs, ["hr", "hr_steps"]):
        f = pooled[pooled.block == block]
        x = np.arange(len(f))
        ax.barh(
            x, f.mae, color=["#777777" if n in ["mean", "median"] else "#3983a5" for n in f.model]
        )
        ax.set_yticks(x, f.model.str.replace("__", " / ", regex=False))
        ax.invert_yaxis()
        ax.set_xlabel("Pooled OOF MAE (composite points)")
        ax.set_title(f"{block}: {int(f.responses.iloc[0])} responses")
    fig.suptitle("Every model within a panel uses identical outer memberships")
    fig.tight_layout()
    fig.savefig(build / "benchmark_errors.png")
    plt.close(fig)
    selected = frames["selected_settings"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, block in zip(axs, ["hr", "hr_steps"]):
        for model, g in selected[
            (selected.block == block) & ~selected.model.str.endswith("__context")
        ].groupby("model"):
            label = model.replace("fixed_ridge", "fixed Ridge").replace(
                "tuned_ridge", "tuned Ridge"
            )
            label = (
                label.replace("hgb", "HGB")
                .replace("__context", ": ctx")
                .replace("_hr", "+HR")
                .replace("_steps", "+steps")
            )
            ax.scatter(g.training_mae, g.validation_mae, label=label, s=28)
        ax.plot([1.7, 2.5], [1.7, 2.5], "k--", alpha=0.4)
        ax.set_xlabel("Outer-training MAE")
        ax.set_ylabel("Outer-validation MAE")
        ax.set_title(block)
        ax.legend(fontsize=7)
    fig.suptitle("Fold stability: each dot is one outer-fold fit")
    fig.tight_layout()
    fig.savefig(build / "training_vs_oof.png")
    plt.close(fig)
    stability = (
        selected.groupby(["block", "model", "setting"])
        .size()
        .rename("folds_selected")
        .reset_index()
    )
    (build / "setting_stability.md").write_text(table(stability) + "\n")
    methods = """The fixed specification precedes these calculations. Five frozen participant folds form outer
validation; three GroupKFold splits inside each training cohort select by mean inner-fold MAE.
Each scaler/encoder is fitted inside its training split. Ridge grid is .1/1/10/100. HGB uses absolute
error, learning rate .05, minimum leaf 20, L2=1, no early stopping; only depth 1/2 and iterations
50/150 vary. Context is actual-start time sin/cos and survey category. Wearables are the existing
four HR summaries, plus supported steps only on the separate secondary cohort. No quality variables,
missingness, imputation, answers or IDs enter X. Fixed Ridge alpha=1 is independently reproduced.
The 60-minute window is a fixed benchmark, not an optimal or universally recall-aligned interval.
HGB differs from Ridge in loss, regularization and function class, so differences do not isolate
nonlinearity. See the [frozen specification](../../window_benchmark_specification.md)."""
    body = f"""# Notebook 09 — window information and bounded nonlinear benchmark

**Repeated readings explain much of the similarity between Morning window inputs. The new models
still do not establish a useful wearable advantage beyond matching median/context baselines.**
This development-only experiment is exploratory after Notebooks 04–08. Timing remains provisional;
none of these results establish validated real-time prediction or absence of physiological signal.

## What information changes with the window?

The primary Notebook 08 matched Morning cohort contains 227 responses / 86 participants. We also
report its all-four-window intersection (191 / 74), separately for nominal wake and later-minute
sensitivity. Current-survey wake defines a retrospective boundary only. No wake variable enters the
new benchmark. All six pairs are in [window_overlap.md](window_overlap.md).

Wake and 60 minutes have **identical accepted source sets in 116/227 (51.1%)** and identical latest
source readings in **227/227**. Wake and 120 minutes are identical in **173/227 (76.2%)**; fixed
60 and 120 in **123/227 (54.2%)**. Thus different calendar boundaries often add no recorded readings.
Five-minute windows are different more often: only 16/227 source sets match wake, and 193/227 have
any accepted five-minute reading; 191/227 have all four features. The latest reading agrees whenever
both contain readings. Shared/latest agreement is not proof of acquisition timing.

{table(nominal[["window_a", "window_b", "responses", "complete_vectors", "identical_sets", "identical_latest_identity", "near_identical_vector", "shared_median", "only_a_median", "only_b_median"]])}

Reading identity uses source-content hash plus original row, within verified participant. Conflicting
and rejected groups stay excluded under unchanged rules; repeated equal HR at different times stays
separate. Source equality and feature equality are different questions. Numerical vector identity uses
absolute 1e-10; descriptive near identity uses ≤1 bpm for mean/SD/latest and ≤.1 bpm/min for slope.
These are fixed descriptive tolerances, not clinical or eligibility thresholds. Near wake/60 vectors
occur in 143/227 (63.0%). Denominators for vector comparisons require four finite features on both
sides; no missing feature is replaced by zero. Only-a/only-b are signed window-specific additions;
medians of zero can coexist with substantial differences in a minority of responses.

![Window source information](window_information.png)

Wake-versus-60 feature differences (wake minus fixed):

{table(fd[["feature", "responses", "mean_signed_difference", "median_absolute_difference", "q95_absolute_difference", "pearson", "spearman"]])}

Mean HR correlation is .973; latest is exactly identical. Some differences remain meaningful in
size: the 95th percentile absolute mean difference is 6.61 bpm. Repeated inputs help explain the
near-identical Notebook 08 predictions but cannot alone explain all weak results. All pairs, both
cohorts and both wake scenarios are retained in [feature_differences.md](feature_differences.md).

## Sampling density changes the weighting

Each observed HR reading contributes equally to the existing mean. A densely sampled minute can
therefore outweigh a sparsely sampled minute. The descriptive alternative averages each **observed**
one-minute mean equally, with bins anchored to that window's start; a partial final bin counts as
one observed bin. Empty bins stay missing. Neither mean establishes continuous coverage, and this
alternative is **not used in the benchmark**. Bin anchoring can differ across wake/fixed windows.

{table(ds[["window", "count_median_median", "count_cv_median", "dense_decile_reading_share_median", "dense_decile_mean_contribution_bpm_median", "absolute_weighting_difference_median", "absolute_weighting_difference_q95"]])}

The densest ceil(10% of occupied bins) contribute a median 14.4% of wake readings and about 14.3%
of 60-minute readings. Their contribution in bpm is their sum divided by *all* window readings, not
their own mean. Short windows have coarse bin counts (one bin can exceed 10% of occupied bins).
For wake, median absolute difference between weighting methods is .484 bpm (95th percentile 1.889);
for 60 minutes it is .437 bpm (95th percentile 2.042). Count dispersion and quantiles, including
empty five-minute views, are in [density_summary.md](density_summary.md). Occupancy and observation
counts describe support, not representative measurement throughout the interval.

![Density and weighting](density_weighting.png)

## Does the implemented pipeline work on traceable inputs?

The deterministic trace uses two distinct participants from each original fold (10 responses,
80 window/scenario views). It checks **320 feature values against 18,153 original accepted HR rows**
from ten development sources, with zero mismatches. It independently checks EMA UTC/buffer arithmetic,
participant/fold mapping provenance, source hashes and row numbers, Boolean window boundaries,
raw timestamps/values, duplicate/conflict handling, saved slice lineage and scalar feature arithmetic.
Absent source columns are verified as absent/null, not invented raw values. Only aggregates appear
here; private trace rows remain ignored. This bounded check does not prove the exporter's timestamp
semantics or rule out every possible implementation defect.

A **synthetic-only** positive control injects a known linear relationship into generated HR and
three survey items for 300 responses / 30 synthetic participants. Readings before the left boundary
and exactly at cutoff are decoys. Production clock parsing, HR acceptance/window summaries, shuffled
one-to-one outcome linkage and the actual model-fitting function recover the relationship across
five held-out participant folds: fixed Ridge MAE **.00774**, fixed synthetic HGB MAE **.26608**, versus
median **3.6**. The positive control validates numerical plumbing for this constructed relationship;
it supplies no evidence of real physiological validity or expected real-world effect size.

## Matched nested benchmark

{methods}

Primary HR cohort: **1,634 responses / 118 participants** of the 3,706 development responses.
Secondary HR+supported-steps cohort: **1,355 / 113**. Every secondary comparator is refitted on those
same rows. Do not interpret error differences between cohorts as incremental activity gains.
All five outer folds are feasible. [Composition](composition.md) includes retained/excluded survey
representation and score distributions; [fold counts](fold_counts.md) reports exact memberships.
Morning asks since waking; Midday and Endday refer to the last prompt. No missing prompt or waking
boundary is inferred from previous completed surveys or first sensor readings.

{table(pooled[["block", "model", "responses", "participants", "mae", "rmse", "participant_mae", "prediction_sd"]])}

Primary HGB context+HR minus median MAE: **+.01038 points, conditional 95% interval
[−.01600,+.03918]**; versus HGB context: **+.00732 [−.01986,+.03622]**. HGB is lower than fixed
Ridge context+HR by .03826 points, but its paired interval [−.08996,+.01785] includes zero. This does
not demonstrate HR-specific gain; the median dummy remains lower in point-estimate MAE.
Secondary HGB context+HR+steps MAE is **2.06983**, versus median **2.06125** and HGB context **2.06985**.
Adding steps to HGB context+HR changes MAE by **+.00078 [−.00392,+.00588]**. No convincing incremental
activity benefit is established. [All 52 paired contrasts](paired_intervals.md) include matching mean,
median, same-family context, fixed Ridge, and same-family HR for the steps addition.

![Matched model errors](benchmark_errors.png)

## Training fit, variation and fold stability

Primary HGB context+HR training MAE is **2.05435**, versus OOF **2.08138**. It produces varying
predictions (pooled SD .286), so it has not simply collapsed to the constant median; flexibility has
not established an OOF wearable benefit. Fixed Ridge context+HR training/OOF MAE is 2.09938/2.11963.
Training pooled metrics count repeated outer-training memberships (each response four times), not
independent new observations. All fold/category MAE, RMSE, equal-participant MAE and prediction SD
are in [metrics.md](metrics.md); selected settings, fold losses, ranges and distinct prediction counts
are in [selected_settings.md](selected_settings.md).

All ten primary tuned Ridge fits select alpha=100; secondary settings vary across folds. Most HGB
fits choose depth 1 / 50 iterations, with some depth 2 or longer fits. [Setting stability](setting_stability.md)
shows every selection. All **600 inner learned fits and 75 outer learned fits** complete; the 20
outer dummy predictions use matching training means/medians. No outer-fit warnings are recorded.
The verification replays split/transform checks and HGB outer predictions and independently solves
Ridge, recomputes metrics and paired intervals. **13,311 fixed-reference prediction values** match
Notebook 05 to numerical tolerance.

![Train versus validation errors](training_vs_oof.png)

Intervals use 2,000 paired participant-cluster resamples of fixed OOF predictions. They retain all
responses for a sampled participant and do not treat overlapping windows as independent. They are
conditional on this repeated exploratory workflow, not refitting/tuning uncertainty, external
validation, multiplicity-adjusted significance or proof of equivalence. Nested tuning protects outer
folds from this search; it does not erase prior development inspection. No winning window or model
is declared and the search is not expanded.

## What this resolves and what still needs evidence

**Verified here:** many Morning windows reuse exactly the same source information; linked sampled
features match original readings; synthetic injected signal is recovered; the bounded nonlinear
family varies predictions but does not establish gain beyond matched baselines.
**Provisional:** HR acceptance, observation-weighted summaries, activity-counter interpretation,
minute wake sensitivity and timestamp use remain assumptions, not validated device standards.
**Unknown:** whether timestamps locate acquisition or receipt, whether batching changes apparent
cadence, and which quality annotations distinguish trustworthy physiological measurements.

**Recommendation:** do not automatically expand model/window searches. Seek provider documentation
and instrumented timing/quality evidence first, then specify an independent development comparison
of measurement-valid intervals and recall boundaries. If an observed-minute weighting comparison is
pursued, freeze it as a separate experiment; this descriptive audit does not authorize substituting
features or claim that weighting correction will help. Final-test evaluation remains sealed.

Provider-question draft (not sent):

1. Does Garmin `ts` timestamp acquisition, interval start/end or server receipt? Can batch delivery
   change timestamp spacing, ordering or duplicate values? Is device acquisition time retained?
2. Can notification/scheduled/delivered/displayed prompt logs be supplied with timezone and clock
   resolution, so Midday/Endday's previous-prompt recall boundary can be observed?
3. Which per-reading device, wear/contact, motion, confidence and beat-quality flags exist? How are
   invalid or stale HR readings exported, and what intervals do step-counter updates represent?

## Artifacts

Source notebook: `notebooks/09_window_information_and_nonlinear_benchmark.ipynb`.
Private immutable bundle: `{bundle.relative_to(root)}`. It contains ignored `window_pairs.parquet`,
`window_density.parquet`, `source_trace.parquet`, `membership.parquet`, `training_predictions.parquet`,
`out_of_fold_predictions.parquet` and training audits. Aggregate tables/figures are in this directory.
Raw sources, earlier bundles, corrected boundaries and assignments/folds are preserved. No HRV,
Garmin stress, imputation, binary targets, final-test evaluation, commit, push or provider messages.
"""
    (build / "report.md").write_text(body)
    (build / "professor_summary.md").write_text("""# Professor briefing — Notebook 09

The new checks do not establish a useful wearable prediction advantage. Wake and 60-minute HR
windows contain identical readings for 116/227 matched Morning responses and share the latest
reading for all 227; wake and 120 minutes match in 173/227. Distinct boundaries often add no data.

A deterministic trace verifies 320 feature values against 18,153 original readings with no mismatch.
A separate synthetic positive control recovers injected signal through the actual linkage/feature/
model pipeline. These checks support implementation correctness but do not validate physiology.

On the fixed 60-minute cohort (1,634 responses, 118 development participants), nested HGB context+HR
MAE is 2.081 versus median 2.071: difference +.010 [−.016,+.039]. It varies predictions but adds no
clear value beyond HGB context (+.007 [−.020,+.036]). Secondary supported steps (1,355 responses,
113 people) add essentially nothing: +.0008 [−.0039,+.0059] versus matched HGB context+HR.
Tuned Ridge also fails to establish improvement. These are loss-and-model-family comparisons,
not an isolated test of nonlinearity; intervals condition on repeated exploratory development fits.

Recommended next step: obtain timestamp/batching, previous-prompt logs and sensor-quality metadata,
then freeze a measurement-valid comparison. Do not expand the grid or open final-test data.
We can test weighting choices ourselves; acquisition semantics, delivery logs and device flags
require provider evidence. Questions are drafted in the report only and have not been sent.
""")
    publish_bundle(build, destination, dict(version=identifier, inputs=inputs, aggregate_only=True))
    return destination
