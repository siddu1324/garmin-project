# Notebook 17: temporal information and past-only personalization

Frozen before new performance. This is exploratory follow-up after Notebooks 00–16, not independent
confirmation. No threshold/window search. Preserve 3,706 development rows, five participant folds and
32 sealed test participants. Binary A ≥5, B ≥10; unchanged three valid stored 1–5 items and reverse coding.

## Evidence gate and distinct tasks

An earlier survey start is not a submission/receipt timestamp. Real EMA history requires a verified
answer-availability clock strictly before the current frozen cutoff, and a strictly earlier start.
No synthetic delay, next-survey assumption, scheduled instance, completion flag or current response
may supply that clock. The supplied schema currently has none. Export all real history fields missing
with explicit reasons; do not fit a history or personalized-label model under this missing premise.
Implement and synthetically test history construction, chronological calibration and prediction.
Report retrospective lag transitions/autocorrelation only as diagnostics, never prediction accuracy.

Task A uses the original five unseen-participant folds, with only their earlier unlabeled wearable
history permitted. This is not a verified live system: Garmin acquisition/receipt remains unknown.
Task A online-EMA history is a distinct blocked setting, pending submission/availability evidence.
Task B requires ≥5 completed, timestamp-verified earlier valid responses for calibration; predict every
later eligible response in order, allowing updates only when an earlier answer actually arrives.
Population parameters may use only permitted other participants and labels already available at the
calibration/fitting time. Do not use later responses from those participants either. Change target is
current minus most recent verified available composite ≥2 points, exploratory and not a validated
meaningful clinical change. Preserve decreases/no-change as class0. No real Task B fitting without
availability clocks. Distinguish start-ordered potential support from verified eligibility.

## Fixed wearable features and cohorts

Current window is the existing 60 minutes. Reuse HR mean, sample SD, latest, slope and supported step
rate. Base raw block = these five. Earlier-recording reference mean uses [cutoff−25h,cutoff−1h), ≥30
occupied UTC minutes/≥3 UTC-hour bins, as previously frozen. Personalized block = reference mean,
mean/latest deviations, current SD/slope, step rate, observed-minute SD, half-window HR change,
adjacent nonoverlapping-hour mean difference and supported zero-step HR stratum. Never call it resting HR.

New clock-matched reference: equally weighted accepted UTC-minute HR means in [cutoff−7 days,cutoff−1h),
within circular UTC time-of-day distance ≤60 minutes of cutoff. Require ≥30 occupied minutes across
≥3 UTC dates. Clip actual readings before aggregation; no complete-minute assumption. Add current mean
minus this reference. This is matching UTC clock phase, not verified circadian/local waking adjustment.
No fixed offset inferred across travel/DST; keep original clocks.
New step contrast: supported rates in [cutoff−48h,cutoff−24h) and [cutoff−24h,cutoff), each requiring
≥3,600 supported seconds; difference only when both available. Wholly contained existing accepted
counter pairs only, no partial boundary assignment/reset repair. These are rolling 24-hour blocks, not
complete-day activity totals. No sleep predictor: current Morning sleep answer is unavailable before
response and no validated sleep stream exists. Respiration/PulseOx/IBI/Garmin stress excluded.

All-valid main cohort: valid outcome/context, no sensor-row exclusion. Training-only median feature
imputation without indicators in physiological-value models (still cannot exclude implicit missingness
information). Explicit separate context+missingness and context+recording-quality controls.
Common observed cohort: all base raw and prior reference/deviations observed. Refit context, raw,
reference-only and personalized comparators there. Supported cohort: fixed60-minute ≥80% accepted
minute occupancy and freshness ≤60s with HR4 present; refit context and raw comparators. Do not change
support after retention/performance. Compare support-restricted saved predictions separately from refits.

## Fixed models and fair comparisons

Main logistic blocks: context only; raw physiology only; context+raw; context+reference only;
context+personalized; context+personalized+clock/step contrasts; context+activity; context+missingness;
context+quality; context+personalized+quality. No participant identifier or full-record centering.
Logistic L2 C={.01,.1,1}, no new imbalance/threshold search, default threshold .5. Main secondary HGB
context/context+raw/context+personalized, depth{1,2}×iterations{50,150}, learning_rate=.05,
min_samples_leaf=20,l2_regularization=1,early_stopping=False. Three grouped inner folds and original
five outer folds; mean inner AUROC selects, fixed listed order breaks ties. Identical cohorts/splits
within comparisons. All preprocessing fitted inside each training split. Invalid classes/folds marked,
never redrawn. Add two fixed C=.1 training-label-permutation computational controls for personalized
block (seeds7/19); not a formal longitudinal permutation significance test.

Constant training-probability baseline for each cohort. Report pooled and within-fold AUROC separately,
positive-negative-pair-weighted within-fold AUROC, equal-fold mean, AP/prevalence, BA/precision/recall/F1,
Brier, reliability bins, participant AUROC where both classes exist, and survey/support strata.
Participant-level tables private; public summaries show estimable denominators. Paired 2,000-draw
participant bootstrap on identical saved predictions; conditional intervals exclude model-refitting
uncertainty and multiplicity correction. Primary contrasts: personalized versus raw; personalized
versus reference-only; raw versus context; clock/step extension versus personalized. Other contrasts
exploratory. Pooled-vs-within-fold recomputation of Notebook16 is read-only; no validation-fitted
probability normalization/calibration. Calibration assessed, not repaired using held-out labels.

## Audit and descriptive analyses

Independently replay scores/clocks; inspect exact item coding and recall evidence. Report component
floor/ceiling, within/between sums-of-squares, internal consistency and person-centered consistency
as descriptive diagnostics (not reliability proof; recall periods differ). Retrospective start-ordered
adjacent valid responses: transitions, change distribution and potential first-five/later support,
with ties excluded. No past-answer features exported to predictors without the availability gate.
Associations adjusted descriptively for survey/cyclic time and supported activity on observed cohorts,
participant bootstrap; never export residuals/person means. Reuse prior measurement evidence, verify
current source hashes, derive only the two new cached-ledger summaries. No raw preprocessing repeat.

References: [grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html),
[calibration](https://scikit-learn.org/stable/modules/calibration.html),
[HGB](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
No new model search beyond this bounded registry, no final-test access, commit, push or provider message.
