# Notebook 13: comprehensive recorded-physiology experiment

Specified 2026-10-17 before new feature/model performance. This follows inspected development results;
it is a prospectively specified exploratory follow-up, not independent confirmation. The 32 final-test
participants remain sealed. Preserve all 3,706 development rows, unchanged composite/targets and five folds.
No outcome/sensor imputation, timestamp reconstruction, new HRV, or boosting search.

## Measurement contract

Reuse verified development normalized partitions and existing accepted HR/counter ledgers. Preserve original
values, clocks, provenance, participant mapping and actual-start cutoffs. All direct windows are
[cutoff−60L,cutoff), L=5,15,30,60,120,360,720,1440 minutes; Python integer offsets. No future endpoint.
HR and counter acceptance is exactly Notebook 04. HR 30–240 bpm; conflicting timestamp groups and clock
failures rejected; equal-time/value copies count once; repeated values at distinct times remain.
Steps/floors use original consecutive accepted counter pairs, same source/date/offset, positive duration
≤300 seconds, nonnegative increment, both endpoints inside window. Resets/rejections break pairs.

HR distributions: mean/median/min/max, sample SD/variance, linear p10/p25/p75/p90, range, IQR,
unscaled median absolute deviation, CV=sampleSD/mean; ≥2 readings/span≥1s. First/latest ≥1.
Slope: OLS against elapsed minutes, ≥3/span≥60s. Net change and first-vs-second-half mean difference
require the distribution support in each relevant set. Skewness and Fisher excess kurtosis use bias-corrected
SciPy estimators, ≥8/span≥60s and nonconstant values (engineering sufficiency, not validated stability).
Adjacency differences: only originally neighboring timestamp groups, both accepted, gap≤60s, ≥2 pairs;
never remove a rejected group and create a new pair. Mean/median absolute difference and RMS difference
are HR-reading variability, not beat-to-beat HRV. Gaps remain missing, not zero.
Equal-weight observed UTC-minute mean/SD require ≥2 occupied minutes. Five-minute rolling summaries
use cutoff-aligned one-minute bins, a one-minute step, ≥3 occupied bins; empty bins remain absent.
Report mean/min/max of qualifying rolling means and mean within-rolling-bin SD. These describe partial
observations, never complete time coverage. All rules fixed before outcome comparisons.

Activity: supported additions; total additions / supported seconds ×60; median/p90/max/sampleSD of
supported pair rates; supported-duration fraction with zero change. No sum of raw counter levels.
HR/activity overlap summaries use HR inside wholly supported step intervals, stratified by zero/positive
increments; ≥3 readings/span≥60s per stratum. Their difference is a descriptive co-observation contrast,
not activity-adjusted/resting HR. No physiological adjustment model is invented.
Multiscale contrasts use adjacent non-overlapping intervals [c−2L,c−L) and [c−L,c) for
L=5,15,30,60,120,360,720: mean difference and SD difference with unchanged support.
Reuse the separately documented earlier HR reference [c−25h,c−1h), ≥30 occupied minutes/≥3 UTC hours;
reference mean and existing mean/latest deviations remain observation-history features, not resting HR.

PulseOx exists, but acquisition/quality/sentinel evidence is not sufficient for a validated modality.
Calculate a SEPARATE provisional descriptive branch using the existing inventory's 70–100 percent-like
range plus identical clock/conflict gates. Zeros are preserved in input and rejected in this view, never
reinterpreted as physiological zero. Run its incremental model only as an explicitly provisional sensitivity.
No clinical desaturation burden/cutoff features. Respiration has no preceding timestamped stream; embedded
current-survey fields are excluded. Calories/intensity counter semantics remain unverified. IBI/HRV stays
in the existing isolated limited pilot; no comprehensive beat features are licensed by new evidence.
Garmin stress is excluded entirely. No raw waveform-derived feature can be generated.

Quality (counts, accepted fraction, minute occupancy, span, gaps, age, pair support, missingness reasons)
is stored separately. A context+quality-only diagnostic and context+missingness-only control are evaluated
separately; they are not called physiological features. Main expanded models use feature-level median
imputation trained per split, and explicit feature missingness indicators. No row is excluded solely
because a modality is absent. Rows without a valid score/context remain accounted for but not fitted.

## Matched evaluation and bounded tuning

All three tasks from Notebook12: three classes 3–4/5–9/10–15; binaries ≥5 and ≥10. No cutoff search.
Primary cohort: all valid-outcome/context development responses, shared by every imputed comparison.
Primary outcome metric macro F1; also macro OVR AUROC, balanced accuracy, per-class metrics, confusion,
high-class PR-AUC/recall/precision, log loss and multiclass Brier sum. Binary macro OVR AUC equals
positive-class AUC. Calibration is descriptive (reliability bins), not a tuned post-hoc correction.

Feature blocks (always context: actual local time sine/cosine and survey category): original 60-minute
HR4+steps; expanded HR60; expanded HR all windows; expanded HR+activity60; all supported modalities60
(the same set when no further validated modality exists: report alias, do not fabricate another model);
full supported multiwindow HR+steps+floors+co-observation+history+contrasts; full after selection.
Separate provisional PulseOx addition; context+quality; context+missingness-only; full values without
missingness indicators. Also compare single-window expanded HR at all eight windows to assess scale.
Duplicate feature sets are explicit aliases, not extra scientific evidence.

Majority and training-frequency probability baselines. Logistic L2 C={.01,.1,1,10}, lbfgs.
Full-set selection experiments: L1 and Elastic Net l1_ratio=.5, saga, C={.01,.1,1}; L2 after greedy
|Pearson r|>.95 removal (deterministic dictionary order, training-imputed values); and grouped selection
among expanded-HR, HR+activity, and full sets using inner mean macro F1 only. Report each procedure
separately, including selected features/fold stability; do not choose one winner for deployment.
Unweighted primary; separately balanced weights for original and selected-full L2 models. Recompute
weights from each training split. Fit medians, constant/all-missing removal, scaling, encoding, redundancy
filter and every selection only within training folds. Drop all-missing/constant training columns;
never assign them invented physiological values. Missingness flags may survive only if variable in training.
Fixed seed 20261017, max_iter=10000, tol=.001. A nonconverged candidate is invalid; do not silently increase
iterations or tune thresholds. Infeasible folds are reported; no split redraw.

Five frozen outer participant folds, three StratifiedGroupKFold inner folds within outer training.
Identical outer/inner cohorts across comparators. Choose by mean inner macro F1; fixed order breaks ties.
No outer-outcome feature selection. Also refit original/context/dummy complete-case 60-minute comparators,
and compare saved imputed predictions on those same validation responses (different training population
explicit). Report full-feature complete-case feasibility; do not discard most people to make it run.

Paired participant bootstrap, 1,000 fixed-seed draws, for principal full/context/original/selected comparisons:
macro F1, balanced accuracy, AUROC differences. Fixed OOF predictions; intervals conditional on development
exploration, not multiplicity-adjusted or retuning uncertainty. No optimal lookback/model claim from
post-hoc point maxima. Preserve class support by training/validation fold and survey composition.

## References and reproducibility

Use the pinned local environment. Official estimator and statistical definitions:
[scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[SciPy skew](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html),
[SciPy kurtosis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html).
Versioned private feature/prediction/estimator bundles with checksums; public aggregates only.
Notebook13 is explicitly authorized to retain aggregate execution outputs. Existing artifacts are immutable.

## Source-sheet amendment before performance inspection

The original `TULIP_Features_Simple.xlsx` was located in the local Downloads directory after initial
extraction and while model jobs were running. Its Features sheet has 54 family rows, not a unique count
of fully expanded predictors. The incomplete model run is preserved but superseded without inspecting
its performance. The completed 633-column extraction is preserved as an earlier version.

Implement the sheet's feasible personal-variation normalization as explicit earlier-recording features
for each L: reference [c−(L+1440) minutes,c−L minutes), equally weighted occupied UTC-minute HR means,
≥30 occupied minutes across ≥3 UTC-hour bins. Reference sample SD uses these minute means. Current mean
and latest deviations divided by reference SD are missing if SD=0 or support is absent. This interval
never overlaps the current L-minute window and never uses future or labeled data. For L=60, verify exact
agreement with the previous reference mean and unstandardized deviations. These are not resting-HR
or full-person z-scores. Source units and unobserved history remain limitations.

The original sheet's steps-per-hour request is represented by an explicitly supported-time rate
(convert the stored per-minute rate ×60). It cannot establish a whole-window hourly activity rate.
Its active-minute share and actual time-with-recording are not inferable from counter increments and
point samples; report supported-time/occupied-bin proxies with their own names, never substitutions.
Original HRV-segment counts and beat-flag diagnostics remain in the frozen IBI audit, separate from
current predictors. Garmin-stress families remain intentionally excluded.
