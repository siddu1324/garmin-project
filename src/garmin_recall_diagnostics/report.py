"""Aggregate-only Notebook 07 report and figures; local evidence links, no IDs."""

import json
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

TABLES = [
    "boundary_availability",
    "candidate_coverage",
    "matched_fixed_coverage",
    "paired_coverage",
    "representation",
    "training_diagnostics",
    "saved_model_inspection",
]


def table(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "—" if np.isnan(value) else f"{value:.4f}"
        return str(value)

    return "\n".join(
        [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        + [
            "| " + " | ".join(cell(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def evidence_text(evidence):
    text = [
        "# Exact questionnaire recall wording",
        "",
        "Visual transcription of supplied blank rendered forms (outlined text); field names and stored codes",
        "cross-checked against the accompanying codebooks. Links resolve to ignored, byte-identical local PDF",
        "copies named by SHA-256. The private source catalog maps each hash to its original archive member.",
        "",
        "All three composite items use the same category-specific recall stem. Stored codes remain 1–5;",
        "displayed anchors are 0 = Not at all through 4 = Extremely. No outcome recoding was performed.",
        "",
    ]
    for category in ["Morning", "Midday", "Endday"]:
        text += ["## " + category, ""]
        for item in evidence["questionnaire_items"]:
            if item["survey_category"] != category:
                continue
            form = (
                "../../../private/recall_diagnostics/questionnaires/"
                + item["rendered_form_sha256"]
                + ".pdf"
            )
            codebook = (
                "../../../private/recall_diagnostics/questionnaires/"
                + item["codebook_sha256"]
                + ".pdf"
            )
            text += [
                f"- `{item['export_field']}`: **{item['exact_wording']}** "
                f"[Blank form, p. {item['rendered_form_page']}, Q{item['question_number']}]({form}); "
                f"[codebook, p. {item['page']}, `{item['questionnaire_field']}`]({codebook})."
            ]
        text.append("")
    wake = evidence["wake_question"]
    text += [
        "## Wake-time evidence",
        "",
        f"**{wake['wording']}**",
        f"[Morning blank form, p. 2, Q7](../../../private/recall_diagnostics/questionnaires/{wake['rendered_form_sha256']}.pdf).",
        "This answer is collected in the current response. It is not a wake event known before that response.",
        "",
        "The mood block’s “Right before starting this survey” wording is not the PSS item recall period.",
        "Some other questions refer to the last survey; they do not redefine PSS’s last prompt.",
        "Endday is a survey category, not documentary evidence for an all-day PSS outcome.",
        "",
    ]
    return "\n".join(text)


def render(root, bundle):
    root, bundle = Path(root), Path(bundle)
    verify_bundle(bundle)
    inputs = {
        "bundle_manifest": sha256(bundle / "manifest.json"),
        "report_code": sha256(Path(__file__)),
    }
    identifier = digest_object(inputs)[:16]
    parent = root / "reports/recall_alignment"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / identifier
    if destination.exists():
        verify_bundle(destination)
        return destination
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    frames = {name: pd.read_json(bundle / (name + ".json")) for name in TABLES}
    for name, f in frames.items():
        write_json(build / (name + ".json"), records(f))
    evidence = json.loads((bundle / "source_evidence.json").read_text())
    write_json(build / "source_evidence.json", evidence)
    (build / "questionnaire_evidence.md").write_text(evidence_text(evidence))
    stats = frames["training_diagnostics"]
    summary = (
        stats.groupby(["model", "alpha"], sort=True)
        .agg(
            fits=("converged", "size"),
            nonconstant_training_fits=(
                "training_unique_predictions_8dp",
                lambda s: int((s > 1).sum()),
            ),
            mean_training_mae_change=("training_mae_difference_vs_median", "mean"),
            largest_training_mae_reduction=("training_mae_difference_vs_median", "min"),
            max_training_prediction_sd=("training_prediction_sd", "max"),
            maximum_nonzero_coefficients=("nonzero_coefficients", "max"),
        )
        .reset_index()
    )
    write_json(build / "training_summary.json", records(summary))
    private_candidates = pd.read_parquet(bundle / "candidate_coverage.parquet")
    nominal = private_candidates.loc[private_candidates.scenario.eq("reported_minute_start")]
    accepted = nominal.loc[nominal.valid_readings.gt(0)]
    # Extra explanatory aggregates use only the same already audited development boundary rows.
    config = json.loads((bundle / "frozen_configuration.json").read_text())
    baseline_config = json.loads(
        (root / config["baseline_bundle"] / "frozen_specification.json").read_text()
    )
    ema = pd.read_parquet(
        root / baseline_config["normalized_bundle"] / "ema/development_ema.parquet",
        columns=["n_ema_id", "n_actual_utc"],
    )
    b = pd.read_parquet(bundle / "boundaries.parquet")
    bad = b.loc[
        b.scenario.eq("reported_minute_start") & b.boundary_reason.eq("wake_not_before_cutoff")
    ].merge(ema, on="n_ema_id", validate="one_to_one")
    actual = bad.n_actual_utc.astype("int64") / 1e9
    extra = {
        "nonpositive_nominal_wake_after_actual": int((bad.start_epoch > actual).sum()),
        "nonpositive_nominal_wake_equal_actual": int((bad.start_epoch == actual).sum()),
        "nonpositive_nominal_wake_between_cutoff_and_actual": int((bad.start_epoch < actual).sum()),
        "accepted_candidate_median_occupancy": float(accepted.occupied_bin_fraction.median()),
        "accepted_candidate_median_maximum_gap_seconds": float(
            accepted.maximum_gap_including_edges_seconds.median()
        ),
        "candidate_duration_min_seconds": float(nominal.duration_seconds.min()),
        "candidate_duration_max_seconds": float(nominal.duration_seconds.max()),
    }
    write_json(build / "explanatory_aggregates.json", extra)
    (build / "training_fold_diagnostics.md").write_text(
        "# Training-only fold diagnostics\n\nNo held-out scores or predictions are calculated. Negative MAE changes favor the fitted model on its own training data only.\n\n"
        + table(stats.drop(columns=["warning_categories"]))
        + "\n"
    )
    available = frames["boundary_availability"]
    fixed = (
        frames["matched_fixed_coverage"]
        .query("scenario == 'reported_minute_start'")
        .sort_values("lookback_minutes")
    )
    candidate_summary = frames["candidate_coverage"]
    selected = candidate_summary.query("scenario == 'reported_minute_start'").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = ["Wake→cutoff\nretrospective"] + [str(int(m)) + "m" for m in fixed.lookback_minutes]
    axes[0].bar(
        np.arange(len(labels)),
        [len(accepted)] + fixed.with_accepted_hr.tolist(),
        color=["#c4682f"] + ["#38749c"] * 8,
    )
    axes[0].set(
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        ylabel="Responses with ≥1 accepted HR reading",
        title=f"Same {len(nominal)} Morning responses; this is not continuous coverage",
    )
    axes[0].tick_params(axis="x", labelrotation=45)
    q = nominal.duration_seconds.quantile([0.25, 0.5, 0.75]).to_numpy() / 60
    axes[1].errorbar(
        [0], [q[1]], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt="o", capsize=8, color="#c4682f"
    )
    axes[1].set(
        xlim=(-0.5, 0.5),
        xticks=[0],
        xticklabels=["Wake→cutoff (median and IQR)"],
        ylabel="Candidate duration, minutes",
        title="Variable recall duration; self-reported minute precision",
    )
    fig.savefig(build / "recall_coverage.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for name, f in summary.groupby("model"):
        x = np.arange(3)
        axes[0].plot(x, f.nonconstant_training_fits, marker="o", label=name)
        axes[1].plot(x, f.mean_training_mae_change, marker="o", label=name)
    for ax in axes:
        ax.set_xticks(np.arange(3), ["0", "0.001", "0.01"])
        ax.set_xlabel("Fixed alpha; no model selection")
        ax.legend()
    axes[0].set(
        ylabel="Nonconstant training fits out of 5",
        ylim=(-0.2, 5.2),
        title="Constant fits persist without regularization",
    )
    axes[1].set(
        ylabel="Mean training MAE change vs median dummy",
        title="Training fit only—not generalization",
    )
    fig.savefig(build / "training_collapse.png", dpi=160)
    plt.close(fig)
    text = f"""# Notebook 07 — recall alignment and median-regression diagnostics

**The questionnaire recall periods differ from fixed pre-EMA windows. Available data support only
retrospective Morning wake-to-cutoff candidates. Removing regularization does not, by itself,
resolve constant HR-only median predictions.** No new validation scores or predictions were made.
This exploratory diagnostic follows inspection of Notebooks 05–06. Its [specification](../../recall_alignment_specification.md)
and three alphas were frozen before the new fits and coverage comparisons.

## What the questionnaires actually ask

[Exact item wording, question/page numbers and local PDF citations](questionnaire_evidence.md)
show that all three composite items refer to **since waking up** in Morning, and **since the last
prompt** in both Midday and Endday. They ask about feeling stressed/nervous, being able to cope,
and being on top of things. Endday does not establish a whole-day recall period. The current
continuous composite and stored 1–5 coding are unchanged; no new target calculation was needed.

## Which clocks are available

| Clock or field | Evidence and limitation | Prediction-time status |
| --- | --- | --- |
| `actual_start_local` | Observed export field, present on all 3,706 development responses; exact app event needs provider confirmation | Used by existing analysis cutoff; retrospective export does not itself prove live availability |
| `timezone_offset` | Observed local-minus-UTC offset on every row | Used to derive UTC at actual start |
| `scheduled_start_local`, `instance_id` | Present on every response; no scheduled/instance discrepancy or actual-before-scheduled flags; instance corresponds to scheduled Unix time | Current scheduled instance is not a log of prior delivered prompts |
| `SLEEP_WAKETIME` | Morning answer to final waking today; 931/934 nonmissing, minute precision | Learned from current response: retrospective-only |
| `SLEEP_SLEEPTIME` | Current Morning self-report, 931/934 nonmissing | Not an observed wake event or a pre-response boundary |
| Notification, delivery, complete scheduled-prompt history | No dedicated log or corresponding timestamp fields supplied | Unavailable; need all prompts, including unanswered/expired/suppressed ones |
| Submission/completion time | No explicit timestamp supplied; completion/reminder flags are answers/instructions | Unavailable; a completion flag is not an event clock |
| `Location Timestamp` | Optional context sample timestamp; semantics need confirmation | Cannot substitute for a prompt, wake or submission event |
| Garmin `ts`, `dte_tme` | Original sensor clocks preserved; Unix `ts` gets no extra timezone shift | Acquisition versus receipt/batching remains unresolved |
| UTC actual start and cutoff | Derived from observed start/row offset; minute starts buffered by 60 seconds, second starts unbuffered | Same frozen cutoff policy as previous notebooks |

The structural audit inspected three EMA headers, 154 sensor headers, six blank PDFs and all
supplied archive member types. No final-test response or sensor values were explored. Absence
means absent from the supplied corrected sources, not necessarily absent from the provider's
system. [Field counts and source inventory](source_evidence.json) retain schema-level evidence.
Previous completed surveys cannot substitute for previous prompts, and a first wearable reading
cannot establish waking. No clock was shifted to improve alignment.

## Retrospective feasibility, not a deployable boundary

The wake question says **today** and **final waking**. The nominal candidate places its reported
HH:MM on the actual-start local date and applies that EMA's offset. Same-day interpretation and
unchanged UTC offset since waking are provisional; cross-midnight answering, DST/travel and
respondent interpretation cannot be verified. There is no previous-day repair. Each candidate
ends at the existing cutoff, excluding readings at the right edge. The buffer can omit part of
the item’s recalled period. The reported minute start and +60-second limiting sensitivity are
kept separate; neither supplies a measured second of waking. This sensitivity addresses recorded
clock granularity only: self-reported waking can be rounded or inaccurate by more than a minute.
The two boundaries are not confidence limits on the true wake event.

{table(available[["scenario", "survey_category", "reason", "responses", "participants"]])}

Nominally **836/934 Morning responses across all 125 development participants** have positive
candidates. Three lack a wake answer; 95 have a nominal wake at/after cutoff: {extra["nonpositive_nominal_wake_after_actual"]}
after actual start, {extra["nonpositive_nominal_wake_equal_actual"]} equal to actual start, and {extra["nonpositive_nominal_wake_between_cutoff_and_actual"]}
in the buffered interval before actual start. These are temporal inconsistencies or buffer exclusions,
not repaired clocks or proof of dishonest responses. Moving to the later minute boundary excludes
10 more, leaving 826. **Zero boundaries are established as available before the current response.**
All 1,838 Midday and 934 Endday rows remain accounted for with unavailable previous-prompt boundaries.
Participant counts across reasons overlap and should not be added.

Nominal candidate duration is **{selected.duration_seconds_median / 60:.0f} minutes** (IQR
{selected.duration_seconds_q25 / 60:.0f}–{selected.duration_seconds_q75 / 60:.0f}; range
{extra["candidate_duration_min_seconds"] / 60:.1f}–{extra["candidate_duration_max_seconds"] / 60:.1f}).
Only **253 candidates / 91 participants** have any accepted HR: **543 have no recorded HR** and
**40 have recordings but all are rejected** under unchanged clock/value/conflict rules. The later
minute sensitivity has 252 with accepted HR, 534 no-recording and 40 all-rejected cases.
Across all nominal candidates, median occupied-bin fraction is zero. Conditional on any accepted
HR it is {extra["accepted_candidate_median_occupancy"]:.1%}, with median largest edge-inclusive gap
{extra["accepted_candidate_median_maximum_gap_seconds"] / 60:.1f} minutes. A recent final reading does not erase earlier gaps.

Bins are anchored at each left boundary, with a partial last minute counted in the ceil(duration/60)
denominator. Occupancy is not supported physiological duration. HR acceptance retains 30–240 bpm,
existing clock flags and timestamp-conflict rejection; identical timestamp/value duplicates contribute
one reading, legitimate readings at different times remain. No interpolation, carry-forward or new
support threshold. Counts are response views of recordings; overlapping windows are not independent.

### Fixed windows on exactly the same nominal candidate responses

{table(fixed[["lookback_minutes", "responses", "with_accepted_hr", "participants_with_accepted_hr", "occupied_bin_fraction_median"]])}

Longer windows find more readings partly because they extend before the reported awakening; this
is not evidence that they match Morning recall better. Both wake scenarios have separately matched
eight-lookback tables, paired availability counts and fold representation in the JSON tables. Responses
without HR remain in every denominator. No boundary/coverage rule was chosen from performance.

![Recall coverage and durations](recall_coverage.png)

## Why median regression can stay constant

The saved five-minute Policy A cohort is unchanged: **1,395 responses / 117 participants**. All 15
saved alpha=.01 context/HR/context+HR fold fits have effectively zero slopes and predictions of 6.
Their coefficients, training means/scales, category encoders and objectives reproduce in this checkpoint;
saved prediction columns were inspected without reading their validation outcomes.

Each new fit uses only four original participant folds: 1,065/1,089/1,125/1,124/1,177 training responses.
Training-only scaling is correct and HR inputs vary. No quality columns, current answers, IDs, activity,
or nonapproved lookbacks enter the models. Targets for the held-out fold are excluded at the Parquet
read before fitting. Each fold serves as training in other rotations, as in the original grouped design.

The median objective is mean pinball loss plus L1 penalty: **0.5 × training MAE + alpha × sum of
absolute coefficients**, excluding the intercept. Alpha=0 removes that penalty. Numeric predictors
are standardized using training population variance; survey categories are encoded on training only.
Full category encoding with an intercept creates expected rank redundancy: coefficient allocation
need not be unique. Coefficient magnitudes are not physiological or causal effects.
[Official QuantileRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html)
and the [linear-model objective](https://scikit-learn.org/stable/modules/linear_model.html#quantile-regression)
support this calculation; installed package versions are recorded in the private manifest.

{table(summary)}

All **45 fits converged**, with no convergence warnings. HR-only and context-only predictions stay
constant in all five folds even at alpha=0. Context+HR at alpha=0 varies in three folds, with training
MAE reductions of approximately **0.0008, 0.0044 and 0.0021 composite points**. At alpha=.001 only one
fold varies; alpha=.01 returns the constant in all five. Median=6 is a strong training solution under
the absolute-error objective; 11.2%–13.6% of each training set lies exactly at 6. The nonsmooth median
loss can have a constant optimum even when predictors vary. This is consistent with weak incremental
linear median information in these training sets, not proof of no wearable signal or of a unique optimum.
Penalty removal alone is therefore an inadequate explanation for HR-only collapse.

[All fold diagnostics](training_fold_diagnostics.md) include rank, sparsity, training loss/penalty,
median mass and prediction variation. Negative loss changes are in-sample only. No new validation
prediction, validation loss, confidence interval, winning alpha or optimal recall window is declared.
Regularization strengths are not directly comparable between Ridge and quantile regression.

![Training-only model diagnostics](training_collapse.png)

## Synthetic explanation (not participant data)

Suppose a Morning response starts at 08:10 with minute precision, offset −240 minutes, and reports
waking at 07:00. The cutoff is 08:09 local / 12:09 UTC; the nominal retrospective interval is
[11:00,12:09) UTC. The later-minute sensitivity starts at 11:01. Neither uses the 08:10 response
answer as information that was known at 08:09. Readings at cutoff are excluded. If only 20 of its
69 bins contain HR, occupancy is 20/69, not 20 minutes of continuously observed physiology.

For synthetic training scores [4,6,6,8], constant 6 has MAE 1 and pinball loss .5. A fitted model
with training MAE .98 and coefficient L1 norm 2 has objective .49 at alpha=0 but .51 at alpha=.01;
the median constant remains feasible at .5. Removing a penalty permits variation, but cannot create
a relationship that the training data do not support. These numbers are teaching examples only.

## What is resolved and what to do next

**Verified:** survey-specific wording; absence of a supplied complete prompt log; current-response
wake field; candidate coverage and attrition; correct scaling/objective replay; persistent unpenalized
HR-only constant fits. **Provisional:** reported-day/offset interpretation, minute anchoring, device
validity rules and timestamp semantics. **Unknown:** actual delivery history and what participants
understood by last prompt, exact event clocks, and real-time availability.

Request provider clarification before treating recall alignment as a prediction-time feature design.
A testable next experiment is a separately frozen, explicitly retrospective Morning-only comparison
of wake-to-cutoff and fixed windows on identical response/fold cohorts, after reviewing accepted HR
support and preserving missingness. Do not pool it with since-last-prompt outcomes or call it real-time
prediction. The current 253 responses with any HR are an upper bound before requiring all four HR
features/support, not a promised model sample. Further alpha searching is not justified by these
training-only gains. A future validation experiment would require a new specification; none was run.

Provider-only questions (draft, not sent):

1. Supply full scheduled, delivered, expired, suppressed and unanswered prompt event logs with
   stable instance keys, UTC timestamps/offsets, start/resume and submission times. Does “last prompt”
   mean scheduled, delivered, noticed or completed? Include first-prompt/day-boundary instructions.
2. Confirm the wake question’s day and timezone semantics, overnight completion behavior, and
   whether a wake event was recorded before the current answer. How should reported wake after
   start be understood?
3. Define Garmin `ts`/`dte_tme`: acquisition or receipt, batching/delay, device clock synchronization;
   confirm what survey actual-start records and whether that event is accessible prospectively.

Choices we can test ourselves: predeclared matched-cohort retrospective window comparisons, the
minute-boundary sensitivity already reported, and training-loss behavior under fixed penalties.
None can resolve provider event semantics by choosing the version with better retention or error.

## Files and safeguards

Private immutable bundle: `{bundle.relative_to(root)}`. `boundaries.parquet` has 7,412 rows (all
3,706 EMAs × two boundary scenarios); `candidate_coverage.parquet` has 1,662 candidate views;
`training_predictions.parquet` has 50,220 training-fit rows (repeated training memberships, not
independent responses). `training_parameters.json` stores private coefficients/scalers and key hashes.
Public tables/figures are aggregate only. Raw data, previous bundles, assignments and all 32 final-test
participants remain unchanged/sealed. No new outcome coding, HRV, stress, imputation, nonlinear model,
feature selection, binary target, commit or push. See the checkpoint verification report for executed
checks and their limits; computational consistency does not establish physiological validity.
"""
    (build / "report.md").write_text(text)
    briefing = """# Professor briefing — Notebook 07

**Two limitations checked; neither supports declaring a new winning model.**

- Questionnaire evidence establishes Morning = since final waking; Midday and Endday = since
  last prompt. Endday is not automatically an all-day outcome. Fixed lookbacks are imperfect proxies.
- Morning has 931 reported wake clocks, but only 836 positive buffered windows (826 under the
  later-minute sensitivity). All are retrospective: the wake answer comes from the current survey.
  Only 253 nominal windows / 91 participants contain any accepted HR; this is not continuous coverage
  or necessarily sufficient support for all features. Midday/Endday need prompt logs not supplied here.
- All 15 saved five-minute median models predicted 6. In 45 training-only fits at alpha 0/.001/.01,
  HR-only and context-only remain constant even without the penalty. Context+HR has tiny unpenalized
  in-sample improvements in three folds (about .0008–.0044 PSS-composite points). No validation outcome
  was used in this diagnostic; these are not generalization improvements.

**Provider knowledge required:** complete prompt/delivery/expiry logs and intended “last prompt”
interpretation; start/resume/submission event definitions; wake date/offset behavior; Garmin acquisition
versus receipt/batching. Questions are drafted in the report and have not been sent.

**Our next testable choice:** prospectively specify a retrospective Morning-only matched-response
comparison of wake-to-cutoff versus fixed HR windows, subject to four-feature/support feasibility.
Keep it separate from a deployable prediction claim. Do not tune penalties from these training gains.
Provider logs are needed before any defensible Midday/Endday recall-aligned experiment.

The source notebook contains only aggregate displays and synthetic examples. Final-test data remain
sealed, original folds and corrected provenance are preserved, and nothing was committed or pushed.
See [report](report.md), [exact local questionnaire citations](questionnaire_evidence.md), and the
checkpoint verification record for details.
"""
    (build / "professor_briefing.md").write_text(briefing)
    publish_bundle(
        build, destination, {"version": identifier, "inputs": inputs, "aggregate_only": True}
    )
    return destination
