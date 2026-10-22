# Notebook 07: frozen exploratory diagnostic specification

This specification precedes new alpha fits and recall coverage comparisons, after inspecting
Notebooks 05–06. It is an exploratory follow-up, not an independent preregistration. No threshold,
recall boundary or model will be selected using validation performance in this checkpoint.

## Recall availability and coverage

Audit all 3,706 development EMAs and preserve their original folds, including absent boundaries
and absent HR. Exact PSS wording comes from the supplied blank rendered questionnaires; the
codebooks connect those items to stored fields and codes. Morning refers to waking; Midday and
Endday refer to the last prompt. Previous completed surveys cannot establish previous prompts.
Scheduled starts attached to completed responses are not a complete notification/delivery log.

Morning's current-response SLEEP_WAKETIME can support retrospective-only candidate windows.
Parse its 12-hour, minute-resolution clock on the actual-start local calendar date, because the
question says today/final awakening. Applying the EMA row's local-minus-UTC offset at waking is
provisional (travel/DST and respondent interpretation remain unknown). Retain the reported-minute
start and a +60-second minute-end sensitivity separately; the latter is a limiting boundary,
not an observed later wake time. Never roll back a date or move a clock to repair an invalid
boundary. Missing/unparseable wake times, invalid clocks and nonpositive intervals remain explicit.
A candidate ends at the existing cutoff: actual start minus 60 seconds for minute-resolution
starts; actual start for second-resolution starts. No candidate uses data at/after the cutoff.
The buffer deliberately omits the final recalled minute when applicable.

Use the corrected current HR disposition ledger and existing [left,right) window engine. Accepted
HR retains Notebook 04 clock, value and duplicate/conflict rules. Report accepted counts, occupied
minute bins, span, freshness, internal/edge gaps and rejected/no-recording cases. For variable
windows, bins start at the candidate left boundary and include a partial final bin; denominator
is ceil(duration/60). Occupancy is a reading-density diagnostic, never continuously supported
physiological time. Compare all eight fixed windows with each candidate on identical response
keys; do not exclude responses without recordings or impose a new coverage threshold. Report
participant and survey representation. Do not calculate aligned-window predictions.

## Bounded training-only median-regression diagnostic

Use only the saved native five-minute Policy A HR cohort, unchanged. For each original held-out
fold, use the other four folds as training. Fit context, HR and context+HR models at quantile .5,
with alpha exactly 0, .001 and .01, solver HiGHS: 45 training fits. Preserve Notebook 05 numeric
scaling and category encoding, fitted only on each training fold. The held-out fold's outcomes
must not be supplied to the fitting/diagnostic function; do not predict held-out responses or
calculate new validation metrics. Per-fold target reads explicitly exclude that fold.

Report training counts, coefficient count/nonzeros (absolute tolerance 1e-8), coefficients and
scaling privately, training prediction SD/range/unique values, solver iterations/warnings and
convergence. Independently calculate mean pinball loss = .5 * training MAE, L1 penalty = alpha *
sum(abs(non-intercept coefficients)), and their sum. Compare with the training-median constant's
pinball loss/MAE. Report training median mass (below/at/above), design rank and constant columns.
Rank deficiency from a full category encoder plus intercept is expected; coefficient allocation
can be nonunique. Training predictions/objective are the relevant diagnostic, not causal weights.
Alpha .01 must reproduce saved Notebook 06 parameters/objectives within numerical tolerance.
Inspect saved five-minute prediction-variation summaries separately; no validation outcomes are
read for this inspection. Improved training fit is not evidence of generalization. Do not choose
a new alpha or combine this diagnostic with support-policy searches.

All final-test outcomes/sensors remain sealed. No score recoding, HRV, Garmin stress, new activity,
imputation, nonlinear model, binary target, feature selection, commit or push. Participant outputs
remain ignored. Preserve prior immutable bundles. Record input hashes, tests, notebook execution,
privacy and raw-integrity checks. Provider questions are drafted only.
