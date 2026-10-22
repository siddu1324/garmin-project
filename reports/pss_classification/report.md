# Notebook 12 — what classification adds, and what it does not

**The primary benchmark improves macro F1 over a majority dummy, but does not establish useful
physiological discrimination beyond context or reliable detection of high composite scores.** Some
longer-window and imputed sensitivities show small positive increments over context. These are
exploratory findings requiring new evidence, not a validated classifier or an optimal window.

All results below come from run `8970bf5ebcc47bfd`, not regression results or a selected favorable fold.
The [specification](specification.md) and [configuration](../../config/pss_classification.json) were frozen
before classification performance. They follow extensive earlier development exploration; this is not
an independent preregistered confirmation. No final-test participant was evaluated.

## Study outcome, cohort and measurement status

The target is the study's three-item perceived-stress composite, computed as nervous + (6−cope) +
(6−control) from verified stored 1–5 items. All three must be valid. Classification maps 3–4 to low,
5–9 to middle, and 10–15 to high. Secondary targets are ≥5 and ≥10 at 60 minutes. These thresholds are
exploratory and neither PSS-10 nor PHQ-9 clinical diagnoses. The questionnaire coding and recall evidence
remain in [Notebook 07's evidence record](../recall_alignment/824551e40b8e301d/questionnaire_evidence.md).
Population stage is not inferred from these model inputs.

All 3,706 development responses and 125 development participants are preserved. There are 3,689 valid
scores: **1,033 low (28.0%), 2,141 middle (58.0%), and 515 high (14.0%)**. Seventeen incomplete scores
remain missing and are excluded from fitting, never imputed. The primary 60-minute joint complete-case
cohort has **1,355 responses / 113 participants**, including 384 low, 808 middle and 163 high responses.

| Lookback, minutes | Joint complete responses | Participants | High responses |
|---|---:|---:|---:|
| 5 | 625 | 104 | 76 |
| 15 | 919 | 111 | 117 |
| 30 | 1,146 | 113 | 146 |
| 60 | 1,355 | 113 | 163 |
| 120 | 1,585 | 118 | 192 |
| 360 | 2,272 | 123 | 282 |
| 720 | 2,693 | 123 | 351 |
| 1440 | 3,207 | 123 | 450 |

The imputed cohorts contain the same 3,689 responses / 125 participants at every lookback. Complete
cases are selected on the union of HR mean, SD, latest, slope and supported steps, so context and
wearable models share membership. A separate HR-native 60-minute sensitivity has 1,634 responses /
118 participants. History-complete analysis has 1,312 responses / 112 participants.

At the primary benchmark, included responses have mean composite 6.27 versus 6.49 among excluded
responses with valid scores (both medians 6). Only 177 Morning responses are included versus 757
excluded; included Midday/Endday counts are 805/373. Participants can contribute both included and
excluded responses. These are selection patterns, not evidence that missingness is random.
[Full accounting](8970bf5ebcc47bfd/cohort_accounting.json) includes original folds, survey category,
missing scores, and class distributions. No row is silently dropped from accounting.

Reuse is limited to corrected current bundles. HR inputs summarize observed readings, not continuous
physiology; even two observations do not establish representative window coverage. Slope retains its
original three-time/60-second-span rule. Supported steps retain the existing intact-pair, boundary and
reset rules. The earlier-recording reference excludes the current hour and uses observed UTC-minute
means with ≥30 occupied minutes across ≥3 UTC-hour bins. It is not resting HR. Acquisition versus
receipt times, batching and activity-counter semantics remain unresolved. Morning recall is since
waking; Midday/Endday recall is since the last prompt, not necessarily the feature lookback. Results are
exploratory prediction from recorded pre-EMA data, not validated real-time prediction.

## Design and completed execution

The unchanged five participant-grouped outer folds each contain a three-fold StratifiedGroupKFold
inner search within outer training participants. Logistic regression searches four C values. XGBoost
and LightGBM each search 12 fixed configurations covering tree complexity, learning rate, tree count,
child support, subsampling and L1/L2 regularization. This is bounded deterministic random search,
an equivalent nested tuning method rather than an Optuna dependency or an exhaustive optimum.

Every model in a cohort uses identical outer training and validation responses and inner splits.
Preprocessing is fitted inside each training split. Balanced weights are n/(K × training class count),
recomputed locally. The dummy always uses the unweighted training majority. No outer-validation early
stopping, probability calibration, decision-threshold search or outcome-based feature selection occurs.
Class predictions use argmax; binary decisions use the fixed default probability boundary.

Complete-case and imputed branches are distinct. The latter fills feature summaries using training
medians, drops wholly unobserved training columns, and adds wearable missingness flags. It never fills
an outcome or raw sensor record. A context-plus-missingness-only diagnostic helps separate physiological
values from the recording process. Sensor counts, gaps, freshness, identifiers and survey item answers
are excluded as predictors. Earlier wearable history contains no previous EMA answers or future readings.

All **115 outer cohort/fold jobs** completed, with **2,130 selected learned estimators** and 115 matching
dummies. The bounded search executed 59,640 inner candidate fits and 2,130 outer learned fits. All inner
training/validation partitions contained every task class; no cohort/fold was redrawn or skipped.
Results comprise 449 pooled model configurations and 2,245 fold metrics. The 1,565 paired differences
are openly reported; this multiplicity is a limitation, not a source of independent replications.

## Primary 60-minute complete-case result

The table is **unweighted** and uses exactly the same 1,355 responses / 113 participants. Learned
wearable models include context + HR + steps. ROC-AUC is macro one-versus-rest.

| Model | Macro F1 | Balanced accuracy | Accuracy | High precision | High recall |
|---|---:|---:|---:|---:|---:|
| Training majority | 0.249 | 0.333 | 0.596 | 0.000 | 0.000 |
| Logistic | 0.251 | 0.334 | 0.596 | 0.000 | 0.000 |
| XGBoost | 0.306 | 0.335 | 0.546 | 0.273 | 0.018 |
| LightGBM | 0.311 | 0.338 | 0.548 | 0.111 | 0.012 |

XGBoost improves macro F1 over the matching dummy by **0.057 [0.037, 0.078]** and over logistic by
0.056 [0.034, 0.076]. LightGBM improves over dummy by **0.062 [0.040, 0.086]** and over logistic by
0.061 [0.038, 0.084]. Brackets are conditional 95% paired participant-bootstrap intervals.

These differences must not be mistaken for reliable discrimination. A majority classifier is a weak
reference for macro F1: changing the prediction-class mix can raise macro F1 while offering little
ranking ability. Here balanced accuracy remains near one-third, macro ROC-AUC is **0.466 (XGBoost)
and 0.469 (LightGBM)**, and high-class detection is almost absent. Macro F1 has no universal chance
level. The lower accuracy versus majority is expected when prediction is spread across classes; neither
accuracy nor a dummy-relative macro-F1 gain alone answers the scientific question.

### Does physiology improve over context?

| Matched comparison | XGBoost Δ macro F1 [95% interval] | LightGBM Δ macro F1 [95% interval] |
|---|---|---|
| Context + HR + steps minus context | +0.015 [−0.010, +0.037] | +0.025 [−0.003, +0.051] |
| Context + HR + steps minus context + HR | −0.001 [−0.022, +0.020] | +0.006 [−0.014, +0.026] |

The primary physiological increments are uncertain. Context-only macro F1 is 0.292 for XGBoost and
0.287 for LightGBM. The HR-native sensitivity also has context-increment intervals containing zero.
Earlier-reference additions do not show a stable gain: all boosting history-increment intervals contain
zero in both complete and imputed cohorts. No biological or causal interpretation follows from these
associations or importance plots.

## High-score detection and weighting

The primary unweighted XGBoost model identifies **3 of 163** high responses; LightGBM identifies **2**.
Balanced context+HR+steps fitting identifies **28/163 (17.2%)** and **24/163 (14.7%)**, respectively,
but high-class precision falls to **10.0% and 9.3%**. This is below the 12.0% high prevalence in this
cohort. Balanced logistic recall is 29.4%, with precision 11.1%. None is a dependable high-score detector.

Balanced context-only XGBoost/LightGBM recall is 30.7%/31.3%, greater than the corresponding wearable
models; context-only macro F1 is 0.309/0.312 versus wearable 0.310/0.304. Thus weighting is changing
the error trade-off rather than establishing physiological benefit. Primary boosting macro-F1 changes
from weighting have intervals containing zero (XGBoost +0.004 [−0.025, +0.034]; LightGBM −0.007
[−0.032, +0.017]). Weighted probabilities are not calibrated risk estimates.

On the all-valid imputed cohort, balanced wearable XGBoost/LightGBM macro F1 is 0.325/0.320; high
recall is 32.8%/33.6%, precision 15.7%/15.3%, against 14.0% prevalence. Macro F1 increases versus
unweighted fitting by 0.049 [0.023, 0.074] and 0.046 [0.019, 0.071]. However, increments over balanced
context are only 0.009 [−0.009, 0.027] and 0.004 [−0.012, 0.021]. A weighting improvement is not evidence
that wearables add information.

## Lookback and missingness sensitivities

There are small positive exploratory increments in some sensitivities. At 60 minutes, unweighted
imputed wearable models improve over matching context by 0.015 [0.001, 0.030] for XGBoost and 0.015
[0.002, 0.027] for LightGBM. Against **context plus missingness alone**, the increments are 0.003
[−0.012, 0.017] and 0.013 [0.002, 0.025]. Balanced logistic's gain over context disappears when compared
with context plus missingness: adding values is worse by −0.023 [−0.036, −0.010]. Availability itself can
carry predictive information, so imputation results cannot all be attributed to physiology.

Imputed versus complete models evaluated on the **same 1,355 complete validation responses** have no
clear boosting improvement: unweighted differences −0.003 [−0.023, 0.017] and −0.016 [−0.038, 0.004];
balanced differences +0.005 [−0.023, 0.032] and +0.012 [−0.012, 0.040]. Training populations differ in
this comparison. Imputation expands response coverage, but a gain in predictive quality is not established.

At 6/12/24 hours, unweighted boosting increments over context are generally larger than at short
lookbacks. For example, on the identical 3,689-response imputed cohort, **24-hour** XGBoost reaches
macro F1 0.301 versus context 0.261: Δ **0.039 [0.021, 0.058]**. LightGBM reaches 0.295 versus 0.260:
Δ **0.036 [0.021, 0.051]**. Both have high recall only **1.7%**. Balanced 24-hour context increments
again include zero. These modest, metric-dependent increments justify a hypothesis about longer-term
recorded context, not a claim that 24 hours is optimal, complete coverage or recall-aligned physiology.
The missingness-only comparator was prespecified only at 60 minutes, so its results cannot isolate
recording-process effects at 24 hours.

Native complete-case lookbacks have different training and validation cohorts. Their common validation
intersection has 625 responses / 104 participants; the [common table](8970bf5ebcc47bfd/common_lookback_metrics.json)
is explicitly evaluation-only because native training sets still differ. The imputed comparisons hold
both populations fixed across lookbacks. No window is selected for deployment or final-test evaluation.

## Secondary binary tasks

The ≥5 task distinguishes low from middle-or-high, not high stress specifically. Unweighted complete-case
wearable XGBoost/LightGBM macro F1 is 0.470/0.469, with context increments about +0.034 and intervals
above zero. However, ROC-AUC is only 0.487/0.482. This again shows why a dummy/context macro-F1
increment alone is insufficient evidence of useful ranking.

For ≥10, unweighted wearable models predict no high responses in both complete and imputed cohorts.
Balanced complete-case XGBoost/LightGBM recall is 10.4%/19.6%, precision 8.5%/11.3%, and ROC-AUC
0.395/0.418. Balanced imputed recall is 36.9%/38.6%, precision 14.7%/14.2%, and ROC-AUC about 0.517
for both. Their context-increment intervals include zero. The binary high cutoff does not resolve the
measurement/prediction limitation. No threshold was optimized after these findings.

## Stability, importance and uncertainty

Primary unweighted XGBoost/LightGBM mean training macro F1 is **0.769/0.709**, versus mean fold-validation
macro F1 **0.302/0.310**. Their validation fold ranges are 0.278–0.337 and 0.288–0.355. Settings vary across
outer folds. Tuning stayed inside training participants, but the large train–validation gap demonstrates
substantial overfitting capacity. Expanding the search is not the justified default response.

Equal-participant-weight confusion-matrix macro F1 is 0.287/0.308 for the two primary wearable models.
This is explicitly not an average of per-person F1 scores. Survey-specific and fold-specific metrics,
class supports and prediction variation are provided, rather than treating repeated responses as independent.

Permutation importance is validation-based and descriptive. Most HR changes are small or unstable across
folds; supported steps have positive permutation decreases in the complete primary models, yet the
matched steps-addition intervals include zero. A variable used by a model is not necessarily an incremental
predictor after refitting. Correlation, unrealistic shuffled combinations and recording selection limit
importance interpretation. No feature is removed or promoted based on these plots.

Bootstrap intervals resample participants (2,000 draws) while retaining all their saved predictions.
They are conditional on fixed folds, selected models and this repeated exploratory workflow. They do
not repeat training/tuning, account for all development exploration, adjust for 1,565 comparisons, or
provide external-validation uncertainty. Response-level sample sizes overstate the independent information;
only 125 development participants exist. Class cutoffs discard within-class severity and remain unvalidated.

## Verification, outputs and next experiment

All **2,130 saved estimators reproduce their OOF probabilities**. Outer medians, scaler means/scales and
encoders match their actual training cohorts. Frozen participant membership, inner-group separation,
majority baselines, selected inner-score maxima, target mapping and corrected provenance were checked.
Raw/reference hashes and frozen assignments are unchanged; 574 previous public/source files and 22
previous private manifests are preserved. The final test has zero evaluated participants.

[Verification](8970bf5ebcc47bfd/verification.json) and [checkpoint checks](verification.md) report tests,
Ruff, notebook execution and privacy review. [Reproduction and exact paths](README.md) document the
versioned ignored input/OOF/job bundles and aggregate outputs. The fully executed Notebook 12 displays
only aggregates and synthetic examples. No HRV, Garmin stress, raw imputation or clinical labels were added.

**Recommendation:** do not expand tuning or open the final test. First obtain acquisition/receipt and
batching definitions, prompt logs and sensor-quality metadata; preserve the distinction between recorded
summaries and coverage. Then specify a narrowly targeted study of high-score detection with verified
sensor/recall alignment and adequate independent high-score participants, ideally newly collected data.
Freeze the high-score operating goal and evaluate precision/recall and calibration as well as macro F1.
If comparing longer-term HR/activity again, include context-plus-missingness controls on those same
windows. Current evidence supports further measurement-focused investigation, not a deployable classifier
or a claim that changing model family has solved the weak wearable signal.
