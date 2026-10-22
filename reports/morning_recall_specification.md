# Notebook 08 — prospectively specified retrospective Morning experiment

Written and frozen before new performance calculations, after inspecting Notebook 07. This is an
exploratory follow-up, not an independent preregistration or a deployable real-time model.
Question: do observed HR summaries between reported final waking and the existing pre-EMA cutoff
predict the study's continuous three-item Morning perceived-stress composite better than fixed
lookbacks, on identical responses? This is predictive comparison, not causal explanation.

## Comparisons fixed before fitting

Primary: wake-to-cutoff versus the preceding **60 minutes**. Secondary: preceding **5** and **120
minutes**. These are specified comparisons, not an optimal-window search. Repeat each comparison
separately for the reported-minute start and the documented +60-second later-minute sensitivity.
Also report a common cohort requiring wake and all three fixed windows, within each wake scenario;
it supplements and does not replace the primary pairwise cohort. Do not use outcomes/performance
to select boundaries, support thresholds, scenarios, features, models or alpha.

## Preserved inputs and feature rules

Use only the current corrected Notebook 04 feature/disposition bundle, Notebook 05 linked outcomes
and context, and Notebook 07 immutable boundaries. Keep all **934 Morning responses** from the
125 development participants in accounting for both scenarios, even when score, boundary or HR is
missing. Preserve frozen assignments and original five participant folds. The 32 final-test people
remain sealed. Verify parent hashes, current window supersession and raw-file integrity.

Reuse the continuous `pss_combo` without recoding or imputation. It requires all three valid stored
1–5 items, nervous + (6−cope) + (6−control). It is not a diagnostic cutoff; no binary target.
The Morning form asks since waking up. The wake answer comes from this current survey and defines
only a **retrospective** interval; wake clock, wake duration and wake availability never enter X.
Same-date and unchanged-offset assumptions remain provisional. Do not invent a date, move a clock
or use first Garmin reading as waking. The one-minute sensitivity is not a bound on recall error.

All windows are [start,cutoff), excluding readings at cutoff. Preserve actual-start buffers:
minus 60 seconds for minute-resolution starts, no buffer for second-resolution starts. No extra
shift of Garmin Unix seconds. Acquisition versus receipt/batching remains unresolved.

Use the same observed-reading HR **mean, sample SD, latest and elapsed-minute OLS slope** in each
interval. Reuse fixed features unchanged and compute wake summaries with the existing corrected
window function. HR acceptance remains finite timestamp/value, 30–240 bpm inclusive, existing
clock flags, whole conflicting-timestamp-group rejection, and one retained same-timestamp/value
group. Repeated values at distinct timestamps stay. Mean/SD require ≥2 distinct accepted timestamps
and ≥1-second span; sample SD uses N−1. Latest requires ≥1 reading. Slope requires ≥3 timestamps
and ≥60-second span. These are existing provisional support rules, not validated wear standards.
No interpolation, carry-forward, resampling, imputation or relaxed feature gates.

Keep occupied-minute bins (ceil denominator for partial last minute), reading counts, freshness,
span and edge/internal gaps separate from predictors. Undefined-boundary quality is missing, not
zero. Occupancy and any recorded HR do not establish complete coverage; overlapping EMA windows
are not independent minutes. Report no-recording, all-rejected and insufficient-count/span reasons.

## Matching and fixed models

Each pairwise cohort is the intersection of valid outcome, valid existing time-of-day context,
valid wake boundary, and all four observed HR features in BOTH wake and its fixed comparator.
Each common cohort requires all four windows. For a given scenario/cohort/fold all models have
identical training membership and identical held-out membership. Train on the other four original
participant folds. Never redraw folds, borrow extra dummy/context training rows or compare unmatched
native-window scores. Retain every excluded response with explicit overlapping reasons.

Fit training-median and training-mean dummies, context-only Ridge, and context+HR Ridge for each
window. Context is existing actual-start cyclic local time-of-day (sin/cos), not buffered or wake
time. Retain the existing training-only category encoder for compatibility: Morning is constant,
so its column supplies no category contrast and is redundant with the unpenalized intercept.
Use existing StandardScaler (population variance), OneHotEncoder(handle_unknown='ignore', no drop),
Ridge(alpha=1, solver='svd', fit_intercept=True). Pipelines fit only on each training fold. No clipping,
participant weighting, tuning, quantile regression, other alpha, HR-only model or extra predictor.
Predictor allowlist: time sin/cos, constant survey category, and the four chosen-window HR summaries.

Skip every model in a fold together if validation has no responses, or training has fewer than two
responses or two participants. Retain skipped counts/reasons; pooled metrics cover only feasible
held-out folds, with missing OOF accounting visible. Flag limited interpretation if cohort <100
responses, <30 participants, or any validation fold <5 participants. These inherited descriptive
flags never alter membership. Unflagged cohorts are still selected, modest exploratory samples.

## Outcomes, errors and uncertainty

Report all-Morning and included/excluded outcome availability/distributions (including missing
counts), response/participant representation and train/validation counts in every original fold.
Report cohort-specific HR quality distributions separately, never added to model X.

MAE primary, RMSE secondary; pooled OOF and fold-level results. Also report equal-participant mean
MAE and equal-participant mean within-person RMSE (distinct from pooled RMSE). Do not pool different
cohorts/scenarios or repeated views as independent observations.

On identical OOF rows compare each context+HR model with its matching mean, median and context
baselines; compare context with both dummies. Compare wake minus fixed context+HR directly for
60, 5 and 120 minutes, on the appropriate pairwise cohort and on the common cohort. Negative delta
means lower error for the first named model. Use the existing 2,000 paired participant-cluster
bootstrap resamples with seed 20261002 for 95% percentile intervals on pooled MAE differences and
equal-participant MAE differences. Resample participant clusters containing all fixed predictions;
do not refit in the bootstrap or independently sample responses/models. These intervals condition on
this exploratory workflow and fixed development predictions, not external/generalization uncertainty,
model-refitting variation, multiplicity-adjusted significance or causal evidence. No winning window.

Deliver source notebook, reusable guarded code, ignored immutable accounting/features/quality/
OOF/audit outputs, aggregate tables/figures, concise report and professor summary. Include synthetic
timeline/feature examples. State what warrants further investigation versus what remains unresolved.
Verify independent window arithmetic/features, joins, feature exclusions, participant separation,
training-only preprocessing, matched baselines, metrics/bootstrap replay, notebook, Ruff, privacy and
integrity. Preserve previous artifacts and existing work. No expanded HRV, Garmin stress, quality
predictors, new activity, imputation, nonlinear models, feature selection, final-test evaluation,
commit or push.
