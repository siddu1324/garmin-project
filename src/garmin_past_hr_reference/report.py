"""Aggregate-only report, privacy-suppressed small-cell summaries and immutable figures."""

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


def public_table(name, frame):
    """Counts stay visible; no singleton group reconstructs an outcome/quality row."""
    f = frame.copy()
    if name in ["composition", "reference_quality"]:
        small = f.responses.lt(5)
        if "participants" in f:
            small |= f.participants.lt(5)
        keep = {
            "scope",
            "group",
            "survey_category",
            "responses",
            "participants",
            "valid_scores",
            "missing_scores",
        }
        for col in f:
            if col not in keep:
                f.loc[small, col] = np.nan
        f["summary_suppressed_small_cell"] = small
    return f


def render(root, bundle):
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    manifest = verify_bundle(bundle)
    inputs = dict(bundle_manifest=sha256(bundle / "manifest.json"), renderer=sha256(Path(__file__)))
    identifier = digest_object(inputs)[:16]
    parent = root / "reports/past_hr_reference"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / identifier
    if destination.exists():
        verify_bundle(destination)
        return destination
    build = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    names = [
        "variation",
        "repeated_counts",
        "response_count_distribution",
        "component_items",
        "retrospective_associations",
        "composition",
        "reference_quality",
        "attrition",
        "metrics",
        "paired_intervals",
        "fold_counts",
        "selected_settings",
    ]
    frames = {n: public_table(n, pd.read_json(bundle / (n + ".json"))) for n in names}
    for n, f in frames.items():
        write_json(build / (n + ".json"), records(f))
        (build / (n + ".md")).write_text(table(f) + "\n")
    variation = frames["variation"]
    comp = variation[variation.measure.eq("pss_combo")]
    assoc = frames["retrospective_associations"]
    metrics = frames["metrics"]
    pooled = (
        metrics[
            (metrics.stage == "oof")
            & (metrics.fold == "pooled")
            & (metrics.survey_category == "All")
        ]
        if len(metrics)
        else pd.DataFrame()
    )
    differences = frames["paired_intervals"]
    counts = frames["repeated_counts"]
    composition = frames["composition"]
    plt.rcParams.update(
        {"font.size": 10, "figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(comp))
    axs[0].bar(x, comp.within_ss_fraction * 100, label="Within participant", color="#367f9b")
    axs[0].bar(
        x,
        comp.between_ss_fraction * 100,
        bottom=comp.within_ss_fraction * 100,
        label="Between participant",
        color="#b8813e",
    )
    axs[0].set_xticks(x, ["All development", "60-minute HR"])
    axs[0].set_ylabel("Fraction of total sum of squares (%)")
    axs[0].legend(fontsize=8)
    for scope, label in [("all_development", "All development"), ("native_60m_hr", "60-minute HR")]:
        f = frames["response_count_distribution"]
        f = f[(f.scope == scope) & (f.count_kind == "valid_composite")]
        axs[1].plot(f.responses_per_person, f.participants, "o-", label=label, markersize=3)
    axs[1].set_xlabel("Valid composite responses per participant")
    axs[1].set_ylabel("Participants")
    axs[1].legend(fontsize=8)
    fig.suptitle("Retrospective outcome variation; no person means enter prediction")
    fig.tight_layout()
    fig.savefig(build / "outcome_variation.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = []
    for i, row in enumerate(assoc.itertuples()):
        ax.errorbar(
            row.correlation,
            i,
            xerr=[
                [row.correlation - row.correlation_low],
                [row.correlation_high - row.correlation],
            ],
            fmt="o",
            color="#367f9b" if row.mode.startswith("within") else "#b8813e",
        )
        labels.append(
            row.feature.removeprefix("hr_").removesuffix("__60m")
            + " / "
            + ("within" if row.mode.startswith("within") else "between")
        )
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Correlation (descriptive participant-bootstrap 95% interval)")
    ax.set_title("Retrospective HR–score associations")
    fig.tight_layout()
    fig.savefig(build / "retrospective_associations.png")
    plt.close(fig)
    f = composition[(composition.scope == "native_60m_hr") & (composition.survey_category != "All")]
    fig, ax = plt.subplots(figsize=(7, 4))
    cats = ["Morning", "Midday", "Endday"]
    x = np.arange(3)
    included = (
        f[f.group == "eligible"].set_index("survey_category").reindex(cats).responses.to_numpy()
    )
    excluded = (
        f[f.group == "excluded"].set_index("survey_category").reindex(cats).responses.to_numpy()
    )
    ax.bar(x, included, label="Eligible history", color="#367f9b")
    ax.bar(x, excluded, bottom=included, label="History excluded", color="#b8813e")
    ax.set_xticks(x, cats)
    ax.set_ylabel("Responses within existing 60-minute HR cohort")
    ax.legend()
    ax.set_title("Observation-history requirement, unchanged support rule")
    fig.tight_layout()
    fig.savefig(build / "history_availability.png")
    plt.close(fig)
    if len(pooled):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(pooled))
        ax.barh(
            x,
            pooled.mae,
            color=["#777777" if n in ["mean", "median"] else "#367f9b" for n in pooled.model],
        )
        ax.set_yticks(x, pooled.model.str.replace("__", " / ", regex=False))
        ax.invert_yaxis()
        ax.set_xlabel("Pooled OOF MAE (composite points)")
        ax.set_title("Identical 1,575-response cohort for every comparator")
        fig.tight_layout()
        fig.savefig(build / "matched_errors.png")
        plt.close(fig)
        f = differences[
            differences.model.str.contains("E_reference")
            & differences.baseline.str.contains("C_raw_hr|D_reference")
        ]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for i, row in enumerate(f.itertuples()):
            ax.errorbar(
                row.mae_difference,
                i,
                xerr=[
                    [row.mae_difference - row.mae_difference_low],
                    [row.mae_difference_high - row.mae_difference],
                ],
                fmt="o",
                color="#367f9b",
            )
        ax.set_yticks(
            range(len(f)),
            [
                ("Ridge" if a.startswith("tuned") else "HGB")
                + " E − "
                + ("C raw HR" if "C_raw" in b else "D reference only")
                for a, b in zip(f.model, f.baseline)
            ],
        )
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("Paired MAE difference (negative favors E)")
        ax.set_title("Reference/deviations do not establish an improvement")
        fig.tight_layout()
        fig.savefig(build / "paired_differences.png")
        plt.close(fig)
    status = pd.DataFrame(
        [
            (
                "Continuous composite",
                "Implemented, unchanged",
                "Three valid stored 1–5 codes required; no clinical/binary cutoff",
            ),
            (
                "Context + raw mean/SD/latest/slope",
                "Implemented and evaluated",
                "Current 60-minute observed readings; provisional timing",
            ),
            (
                "Earlier-recording reference + deviations",
                "Implemented and evaluated here",
                "24-hour history excluding current hour; ≥30 minutes and ≥3 UTC hours",
            ),
            (
                "Supported steps; other existing HR summaries",
                "Implemented earlier; outside this experiment",
                "No additional activity or HR feature expansion here",
            ),
            (
                "Whole-record person means/centering",
                "Retrospective diagnostics only",
                "Never predictive features or saved person-level means",
            ),
            (
                "Other history lengths/support thresholds",
                "Untested candidates",
                "Would require a separate specification; no automatic search",
            ),
            (
                "Equal-minute current HR summaries",
                "Untested predictive candidate",
                "Notebook 09 descriptive weighting audit only",
            ),
            (
                "Resting HR / complete daily physiological baseline",
                "Blocked by measurement evidence",
                "Requires validated rest/activity/wear and temporal support, not this reference",
            ),
            (
                "Expanded HRV / normal-beat measures",
                "Blocked by measurement evidence",
                "IBI anchoring, missing beats and beat-quality status unresolved",
            ),
            (
                "Recall-aligned prompt intervals",
                "Blocked where logs unavailable",
                "Previous completed EMA is not previous prompt",
            ),
            (
                "Garmin stress, imputation, binary outcomes",
                "Excluded by design",
                "Not authorized predictors/transformations here",
            ),
        ],
        columns=["feature", "status", "meaning_or_limit"],
    )
    write_json(build / "feature_status.json", records(status))
    (build / "feature_status.md").write_text(table(status) + "\n")
    provider = """# Provider request draft — not sent

Could you clarify the following export semantics and available metadata?

1. For Garmin `ts`, is the timestamp acquisition time, device event time, interval start/end, or server receipt? How are clock drift, timezone changes and delayed uploads represented?
2. Can records arrive in batches or be reordered/repeated? Is a separate original acquisition timestamp available, including precision and latency?
3. Can scheduled, notification/delivery/display and actual survey-start/submission logs identify the preceding prompt for the questionnaire recall period?
4. Which device wear/contact, motion, confidence and HR/beat-quality flags are available, and what do missing/invalid/stale measurements look like in the export?
5. For IBI, does `ts` anchor the start or end of an interval, or receipt? Are intervals beat-to-beat/normal-to-normal, how are missed or rejected beats encoded, and are corrected values or reconstructions applied upstream?

This is a local draft only. No provider message has been sent.
"""
    (build / "provider_request_draft.md").write_text(provider)
    summary = f"""# Notebook 10 — within-person signal and earlier-recording HR reference

**The earlier-recording reference/deviation models do not improve on matched raw HR or
reference-only models in this bounded experiment.** Outcome variation is substantial both within
and between participants; that fact alone does not establish useful HR prediction. Timing remains
provisional. This estimates exploratory prediction after a wearable observation period, not validated
real-time prediction, resting HR or a complete daily baseline.

## Where variation occurs — retrospective only

All 3,706 development responses are retained; 3,689 have valid continuous composite scores. The
125 people have a median 31 valid responses (range 17–32). The existing 60-minute HR cohort has
1,634 responses / 118 people, median 13 (range 1–30); one person has a single response in that cohort.
No whole-record participant means, outcome histories or centered rows enter predictive features.

{table(counts)}

{table(comp[["scope", "valid_responses", "participants", "within_ss_fraction", "between_ss_fraction", "pooled_within_sd", "person_mean_sd", "median_person_sd"]])}

Across all development responses, **53.5%** of total squared variation is within participants and
**46.5%** is between their observed means, weighted by response count. On the native HR cohort the
fractions are **52.3% / 47.7%**. Within-person pooled SD is 2.006 and 1.925 composite points respectively.
This is an exact descriptive SS decomposition, **not** an estimated random-effects ICC, reliability
measure or causal explanation. Between-person means can reflect different survey/context sampling.
The decomposition and all item results are in [variation.md](variation.md).

![Outcome variation](outcome_variation.png)

## Three component items, unchanged composite

[Local questionnaire evidence](../../recall_alignment/824551e40b8e301d/questionnaire_evidence.md)
verifies stored codes 1–5 paired with displayed 0–4 anchors. `PSS_NERVOUS` asks stressed/nervous;
`PSS_COPE` asks able to handle/cope; `PSS_CONTROL` asks on top of things. Preserve
**nervous + (6−cope) + (6−control)**, require all three, range 3–15. No item or cutoff is selected
as a replacement outcome. It is the study's three-item perceived-stress composite, not a diagnostic.

All-development valid counts are 3,692 nervous, 3,690 cope, 3,689 control; stored means are
1.856, 3.824 and 3.625. Higher cope/control imply *lower* composite contributions after reversal.
Nervous has 74.7% within-person SS, cope 49.3%, control 49.8%. Each uses its own valid-item denominator;
these are not interchangeable alternative targets. [Component tables](component_items.md) include
all five code frequencies, missingness and survey-category summaries, repeated on the HR cohort.
Morning asks since waking; Midday/Endday since the last prompt. No missing prompt boundary is invented.

## Descriptive HR–score associations

Use the native 60-minute cohort: between-person correlations relate equally weighted full-record
participant means; within-person associations demean both HR and score within each person, weighting
responses and excluding the one singleton. All full-record means are retrospective and can use
future responses relative to an event. They are never exported to the predictive tables. Confidence
intervals resample whole participants, not independent rows; no causal or validated predictive claim.

{table(assoc[["feature", "mode", "participants", "correlation", "correlation_low", "correlation_high", "slope", "slope_low", "slope_high"]])}

For HR mean, between-person r=.096 [−.065,.245]; within-person r=−.032 [−.094,.036]. Other centered
correlations are similarly small in this diagnostic; all displayed association intervals include zero.
This neither proves an absence of physiological relationships nor motivates a replacement target.

![Retrospective associations](retrospective_associations.png)

## Earlier-recording reference and available history

The [specification](../../past_hr_reference_specification.md) was frozen before new prediction results.
For each existing cutoff, keep current **[cutoff−1 hour, cutoff)** and use reference
**[cutoff−25 hours, cutoff−1 hour)**. The history is exactly 24 hours and cannot borrow a current-window
or future reading. Existing precision buffers and HR acceptance remain unchanged. First clip accepted
readings to the exact interval, then average within **observed UTC minutes**, then average those
minute means equally. Empty minutes are missing; boundary minutes can be partial. Same-value timestamp
ties/conflicts follow the original disposition ledger. Repeated equal values at different times remain.

Require **30 occupied UTC minutes AND 3 distinct UTC-hour bins**. This fixed engineering rule is
provisional, not a physiological standard. It does not establish complete coverage or resting HR.
Reference age is derived from actual observed timestamps; quality includes first/latest/equal-minute
mean observation age, span, internal/edge gaps and occupied bins, excluded from predictors.
Current HR mean is reading-weighted, whereas reference mean is minute-weighted: deviations combine
those two summaries, not a pure isolated physiological change. No EMA answers or whole-person
normalization are used. Held-out participants contribute only their own earlier unlabeled HR.

Reference history passes for **3,059/3,706 responses** independent of current HR/outcome availability.
Intersecting with the native HR cohort retains **1,575/1,634 (96.4%) / 116 participants**; 59 responses
are excluded. This is 42.5% of all development responses; most overall loss predates the history rule.
All five original outer folds remain feasible, with 268–364 validation responses and 22–25 people.

{table(composition[(composition.scope == "native_60m_hr")][["group", "survey_category", "responses", "participants", "score_mean", "score_median"]])}

Included scores average 6.291; the 59 history-excluded native responses average 6.678. They represent
51 people, many also contributing eligible responses; participant counts cannot be added. Among all
2,131 excluded development responses, 2,114 have scores (mean 6.493), 17 remain missing. Small-cell
outcome/quality summaries are suppressed publicly, with counts retained; private accounting stays complete.

Of the 59 native exclusions: six have no recording; 15 have recorded but entirely rejected readings;
23 have some accepted readings but fail both minute/hour rules; seven fail only the minute rule;
eight fail only the hour rule. The overlapping totals are 51 failing minutes and 52 failing hours;
these cannot be summed. [Attrition](attrition.md) retains both exclusive combinations and overlaps.

Eligible reference histories have median **572 occupied minutes / 12 occupied UTC-hour bins**.
Median latest-reading age is **3,600.935 seconds**, just beyond the excluded current hour. But the
median largest gap including edges is **41,247 seconds (11.46 hours)**; median internal maximum gap
is 37,855 seconds. A recent endpoint does not fill the preceding long gap. Counts of occupied bins
are recording diagnostics, not observed continuous physiological duration. See [reference_quality.md](reference_quality.md).

![History availability](history_availability.png)

## Matched A–E comparisons

A: mean/median dummies. B: actual-start cyclic time + survey category. C: B + current raw HR
mean/SD/latest/slope. D: B + earlier reference mean. E: B + reference mean + current mean/latest
minus reference + current SD/slope. Every model uses exactly the same eligible training and validation
responses in unchanged outer participant folds. No answer, participant ID, quality/missingness,
Garmin stress, new activity, imputation or HRV enters X.

For B–E, reuse Notebook 09's Ridge [.1,1,10,100] and HGB depth [1,2] × iterations [50,150] grids.
HGB absolute-error loss, learning_rate=.05, min_samples_leaf=20, L2=1, early_stopping=False are fixed.
Three participant-grouped inner folds inside each outer training set select mean inner MAE; training
splits fit their own scaler/encoder. Outer validation outcomes never tune settings. No row-random
holdout or clipping. Ridge/HGB differ in loss and regularization as well as function class.

{table(pooled[["model", "responses", "participants", "mae", "rmse", "participant_mae", "prediction_sd"]]) if len(pooled) else "No folds were feasible; no prediction comparison is made."}

HGB E MAE **2.0949** versus C **2.0646**, D **2.0854**, B **2.0694**, and median **2.0673**.
E−C is **+.0303 [−.0253,+.0840]**; E−D **+.0095 [−.0172,+.0343]**. Ridge E MAE **2.1218** versus
C **2.1158**, D **2.1144**: E−C **+.0060 [−.0010,+.0135]**, E−D **+.0074 [−.0007,+.0155]**.
These reference/deviation representations do not establish gain. D−B tests reference levels alone;
E−D tests added current HR conditional on that reference; E−C tests the reference/reparameterized
representation versus raw HR. Differences and coefficients are not causal; Ridge's penalty geometry
also changes under reparameterization.

HGB raw-HR C is .0027 points below median, but its interval [−.0179,+.0115] includes zero; versus
HGB context the change is −.0048 [−.0214,+.0111]. This small conditional result does not justify a
winning model or further searching. Ridge E is worse than median by .0545 [.0154,.0912] on pooled
MAE; some participant-MAE contrasts also exclude zero on the worse-error side. Report the full
[paired differences](paired_intervals.md), not just favorable comparisons.

![Matched OOF errors](matched_errors.png)
![Primary representation differences](paired_differences.png)

All 480 inner and 40 outer learned fits complete; matching dummies add ten outer predictions.
All Ridge selections are alpha=100. HGB settings vary by fold; selected configurations, prediction
ranges/SD/distinct counts and training/validation MAE are in [selected_settings.md](selected_settings.md).
[Metrics](metrics.md) retain pooled, fold and survey MAE/RMSE/equal-participant MAE and variation.
Training metrics count repeated outer-training memberships, not independent extra observations.
The [fold table](fold_counts.md) records exact cohort sizes. No support/fold rule was relaxed.

Intervals use 2,000 paired participant-cluster resamples of fixed OOF predictions, not independent
responses. They omit refitting/tuning uncertainty, multiplicity adjustment and external validation.
Nested tuning does not erase repeated development exploration. No optimal window or model is declared.

## Synthetic teaching example and feature status

Suppose current cutoff is Tuesday 12:00 UTC. Current window is Tuesday 11:00–12:00; history is
Monday 11:00–Tuesday 11:00, right-exclusive. A reading exactly Tuesday 11:00 is excluded from
reference and can enter current HR. Suppose 30 accepted minute means span three UTC hours: ten
minutes average 60, ten 70, ten 80. Reference is 70 regardless of reading density within minutes.
If current mean=85 and latest=90, deviations are +15 and +20 bpm. Those values are synthetic;
empty minutes stay missing, and no inference of resting physiology follows.

[Feature-status table](feature_status.md) distinguishes implemented/evaluated features, untested
candidates and measurement-blocked concepts. Historical personalization Markdown in the legacy
`notebooks/v2_raw/14_v2_feature_blocks_and_baselines.ipynb` and `15_v2_data_walkthrough_and_descriptives.ipynb`
used prior nonoverlapping EMA-associated wearable windows. That documentation was consulted only
as conceptual background; no obsolete feature/results were imported as evidence or run.

## Recommendation and outstanding provider evidence

**Stop automatic model expansion for this experiment.** History availability is adequate under the
fixed provisional rule, but it is discontinuous and the new representation does not improve matched
prediction. Seek acquisition/receipt and batching semantics, actual prompt logs, sensor-quality flags
and IBI anchoring/missed-beat definitions before proposing another measurement-valid experiment.
The [provider request](provider_request_draft.md) is a local draft only and has not been sent.

Verified facts are numerical consistency, cohort membership and the stated observed results.
Provisional assumptions are timestamp availability, HR acceptance, reference support and meaning.
Only provider evidence can establish acquisition/receipt semantics and missing quality metadata.
Neither outcome decomposition nor retrospective associations demonstrate deployability or absence
of true physiological signal. No expanded HRV, stress, imputation, binary cutoff or final-test evaluation.

## Outputs and verification

Private immutable bundle: `{bundle.relative_to(root)}`. All 3,706 rows remain in `reference_features.parquet`,
`reference_quality.parquet`, `reference_lineage.parquet`, `predictive_inputs.parquet` and `accounting.parquet`.
`out_of_fold_predictions.parquet` has 1,575 response rows; `training_predictions.parquet` has 6,300
repeated training memberships. All are ignored. No row-level full-person means are persisted.
Notebook: `notebooks/10_within_person_signal_and_past_hr_reference.ipynb`.
Preserve raw sources, prior immutable outputs and frozen assignments. No commit, push or provider messages.
"""
    (build / "report.md").write_text(summary)
    (build / "professor_summary.md").write_text("""# Professor briefing — Notebook 10

The earlier-recording HR reference does not improve prediction in this bounded experiment.
About 53.5% of development composite-score variation is within participants and 46.5% between
observed person means. Retrospective person-centered HR associations are small and uncertain;
these means never enter prediction.

The fixed past-only rule retains 1,575/1,634 existing 60-minute HR responses, from 116 people.
Reference history is [cutoff−25h,cutoff−1h), averaging observed UTC-minute means, requiring 30
occupied minutes across three UTC hours. This is an earlier-recording reference, not resting HR.
Eligible histories have median 572 occupied minutes but an 11.46-hour largest gap including edges.

On identical outer folds, HGB reference/deviation E MAE is 2.0949, versus raw HR C 2.0646 and
reference-only D 2.0854. E−C: +.0303 [−.0253,+.0840]; E−D: +.0095 [−.0172,+.0343].
Ridge E is also worse in point estimate. Matching median MAE is 2.0673. Conditional bootstrap
intervals and nested tuning do not remove repeated exploratory development limitations.

Do not expand models automatically. Request acquisition/receipt/batching definitions, prompt logs,
sensor-quality flags and IBI anchoring/missed-beat semantics before another measurement-valid
experiment. The request is drafted locally only. All 32 final-test participants remain sealed.
""")
    publish_bundle(
        build,
        destination,
        dict(
            version=identifier,
            inputs=inputs,
            aggregate_only=True,
            development_only=manifest["development_only"],
        ),
    )
    return destination
