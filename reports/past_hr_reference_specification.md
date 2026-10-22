# Notebook 10 — frozen within-person diagnostics and earlier-recording reference experiment

Specified after Notebooks 04–09 and before new performance. This is one bounded exploratory
follow-up, not independent preregistration, guaranteed improvement or validated real-time prediction.
Preserve corrected immutable bundles, 125 development participants and original five folds; all
32 final-test participants stay sealed. Earlier personalization material is conceptual history only:
no obsolete features/results are current evidence. No personal outcome calibration is authorized.

## Descriptive diagnostics only

Retain every development response for accounting; use valid continuous composite scores for variance.
Repeat on the unchanged native 60-minute HR/outcome/context cohort. Report response counts per
participant including zero-valid-score people, singleton counts, and valid count quantiles.
Decompose total sum of squared deviations exactly into within-person SS plus response-count-weighted
between-person SS. Report SS fractions, pooled within-person SD sqrt(SSwithin/(N−P)), sample SD of
person means and median within-person SD. These are descriptive decompositions, not random-effects
ICC estimates or causal attribution. Different person means can reflect sampling/context differences.

Describe each stored valid item (1–5): PSS_NERVOUS contributes its code; PSS_COPE and PSS_CONTROL
contribute 6−code. Keep nervous+(6−cope)+(6−control), all three required, range 3–15; no new target.
Report item missingness, frequencies, means and variation, without selecting a replacement item.
Questionnaire source links remain in Notebook 07's exact questionnaire evidence. Morning asks since
waking; Midday/Endday since the last prompt. Those differences remain unresolved by a 60-minute window.

On the native 60-minute cohort, describe each of the four HR summaries versus composite: Pearson
correlation and OLS slope between full-record participant means (equal participant weights), and
within-person centered correlation and OLS slope (response weights). Participants need ≥2 paired
responses for within-person diagnostics. Use 2,000 participant-cluster resamples (seed 20261002)
for descriptive percentile intervals; retain entire person records together. No row-independent
p-values. Full-record means use future responses relative to individual events and are strictly
retrospective. This function returns only aggregate diagnostics; no person means/centered records
are persisted or admitted into predictive features. Associations are not validated prediction.

## Reference features fixed before fitting

Current window stays [cutoff−3600, cutoff), unchanged reading-weighted mean, SD, latest and slope.
Reference history is [cutoff−90000, cutoff−3600): exactly 24 hours ending one hour before cutoff.
Use only existing accepted HR dispositions and verified identity; never infer a clock or change
quality rules. UTC-minute bins are floor(Unix seconds / 60), UTC-hour bins floor(seconds / 3600).
Clip readings to the exact history first, then average readings within observed UTC minutes and
average those minute means equally. A boundary UTC minute may be partially observed; only inside
readings contribute. Empty bins stay missing. No interpolation, imputation or added zero readings.

Require **≥30 occupied UTC minutes AND ≥3 distinct occupied UTC-hour bins**. This is a fixed,
provisional engineering feasibility rule, not a validated physiological standard or resting-HR
criterion. The earlier-recording reference is neither continuous coverage nor a complete daily
baseline. If unavailable, preserve missing predictors and explicit overlapping failure reasons.
Do not relax support or choose boundaries from performance. Count/span/gaps/coverage/freshness and
survey composition are quality diagnostics only, never predictors.

New predictors: reference_mean_bpm; hr_mean_deviation_bpm = existing current mean−reference;
hr_latest_deviation_bpm = existing current latest−reference. Compute deviations only when both
inputs are available. No current/past EMA answers, whole-person normalization, participant IDs,
future readings or current-window readings enter the reference. Held-out development participants
may supply only their own earlier **unlabeled wearable history**. This estimates prediction after a
wearable observation period, with timestamp availability still provisional (acquisition/receipt unknown).
Current mean is reading-weighted while reference is observed-minute-weighted: deviations compare
these two summaries and are not a pure experimentally isolated change in resting physiology.

## Identical-cohort benchmark

Eligible = existing native 60-minute HR/outcome/context cohort AND sufficient earlier history.
All A–E comparators refit on identical outer/inner memberships. A: matching mean/median dummies.
B: context (actual-start cyclic time, survey category). C: context + four unchanged raw HR features.
D: context + reference mean only. E: context + reference mean + mean/latest deviations + current HR
sample SD + current HR slope. Do not add steps, quality, answers, HRV or Garmin stress.

For each B–E use Notebook 09's exact grids: Ridge alpha [.1,1,10,100]; HGB absolute-error loss,
learning_rate=.05, min_samples_leaf=20, l2_regularization=1, early_stopping=False,
categorical_features=None, random_state=20261006; ordered (depth,iterations) grid
(1,50),(1,150),(2,50),(2,150). Other installed defaults recorded. No additional model/grid search.
Original five outer participant folds; three inner GroupKFold(shuffle=False) participant folds,
rows sorted by stable EMA key. Select arithmetic mean inner-fold MAE, exact tie first grid entry.
Fit scaler and dense category encoder on each inner training set, then refit chosen configuration on
outer training only. No validation-outcome access in tuning, no row-random early stopping, no clipping.
Skip all comparators for infeasible folds (no validation, fewer than three training people or inner
training fewer than two people); never redraw folds. Fewer than 100 responses/30 people or five
validation people in any fold flags limited interpretation without changing eligibility.

Compare E versus C (earlier reference/reparameterization versus raw HR) and E versus D (added current
HR information conditional on reference). Also compare every learned model with matching mean/median
and same-family B context. Report D versus B to distinguish reference-level information. E versus C
also changes representation/penalty geometry for Ridge; coefficients and differences are not causal.
Ridge and HGB differ in loss, regularization and function class, not just nonlinearity.

Report pooled/fold/category MAE, RMSE, equal-participant MAE, prediction variation, selected settings,
fold stability and training versus OOF error. Paired participant bootstrap: 2,000 draws, seed 20261002,
percentile 95% intervals; fixed development predictions, no refitting. Intervals omit selection/refitting
uncertainty, external validity and multiplicity correction. Nested tuning does not erase prior exploration.
Report eligible/excluded groups both relative to all development responses and native 60-minute cohort,
with score/category distributions, separate history quality and overlapping/partitioned attrition.
If infeasible, publish feasibility without fitting. If no improvement, stop automatic expansion.

## Verification and deliverables

Persist ignored versioned reference features/quality/lineage, all-response accounting and matched
OOF predictions. Output no IDs/complete rows. Independent datetime/mask/minute arithmetic checks,
future/current exclusion, boundary UTC partial bins, duplicates/conflicts, insufficient history,
participant/fold isolation, training-only preprocessing, metrics/bootstrap replay and deterministic
reruns are required. Run existing tests plus focused tests, Ruff, notebook, input integrity and privacy.
Preserve prior files/bundles. Teaching notebook/report distinguish implemented features, untested
candidates and features blocked by measurement evidence. Draft provider questions; do not send.
No expanded HRV, stress, imputation, outcome cutoff change, final-test evaluation, commit or push.
