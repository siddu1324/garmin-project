# Notebook 06: prospectively specified exploratory follow-up

This experiment follows inspection of Notebook 05. It is a **prospectively specified exploratory
follow-up, not an independent preregistered confirmation**. This specification and configuration
are frozen before calculating new comparison results. Provider timing questions remain unresolved:
all results are exploratory prediction from recorded pre-EMA data, not validated real-time prediction.

Two questions:

1. Does HR add predictive information when recordings are denser and more recent?
2. Does a model targeting the median improve MAE compared with the squared-error Ridge baseline?

## Fixed inputs and support policies

Use only Notebook 05's development linked rows, continuous three-item composite and original five
participant-grouped folds. Preserve all 3,706 linked rows, including 17 missing outcomes. No score
recoding or binary cutoff. Keep HR mean, sample SD, latest and slope at 5, 15, 30, 60, 120, 360, 720
and 1440 minutes. Context remains survey category plus sin/cos of actual local start time. Other
sensor features, quality, identifiers and current answers cannot enter any model matrix.

A: Notebook 05 complete-case HR cohort, with its existing support rules.
B: A plus at least 80% of the lookback's minute bins containing an accepted HR reading.
C: B plus latest accepted HR reading at most 60 seconds before the cutoff, inclusive.

Use integer comparison `5 * occupied_bins >= 4 * lookback_minutes`; freshness ≤60 seconds. These
are provisional sensitivity choices specified by the task, not validated device standards. No
alternative thresholds, density definitions, lookbacks or policy combinations will be searched.

The inspected Notebook 04 code computes occupancy from `a_acceptable` timestamp groups, not all
recorded readings. Independently verify it using `a_analysis_keep` in the preserved reading ledger.
Accepted means the original finite-time, clock-flag, 30–240 bpm and duplicate/conflict checks passed.
A minute bin is [start+60k,start+60(k+1)), where start=cutoff−60L; it is anchored to this lookback,
not civil-clock minute boundaries. No cutoff reading is included. Duplicate timestamps do not add
bins. Point occupancy does not prove continuous physiological support; keep span, freshness and
maximum internal/edge gaps visible. Do not alter features or manufacture support.

## Experiment 1: evaluation restriction versus retraining

For each policy and lookback, first filter the saved Notebook 05 primary HR predictions to the
eligible validation responses. Label this `saved_subset`: model fitting remains on the original
reference cohort. Saved mean/median/context comparators share that same original training cohort.
This is deliberately an evaluation-population change, not a model retrained on the support policy.

Separately label `support_refit`: refit mean and median dummies and fixed Ridge(alpha=1, solver=svd)
context-only, HR-only and context+HR models within the policy-eligible cohort in each original fold.
All models in this comparison use identical training AND validation rows. Compare refit versus
saved predictions only on shared validation rows, as a separately labeled paired retraining contrast.

Native membership uses one lookback. Common membership intersects eligible responses across all
eight lookbacks within each policy; keep that membership for training AND validation of refits.
Saved common-subset predictions come from Notebook 05's common HR fits, not its native fits.
Do not shrink the lookback set to rescue a sparse intersection. Report empty and small cohorts.

Retained versus excluded composition uses Notebook 05's corresponding A cohort as denominator;
also account for all development EMAs, including those excluded by original complete-case rules.
Report counts, participants, every fold, survey composition, outcome distribution, occupancy,
accepted-reading counts, span, freshness, maximum internal gap and maximum gap including edges.

## Experiment 2: median-loss linear baseline, no support-policy combination

On unchanged Notebook 05 native/common **A HR cohorts only**, fit three models:
QuantileRegressor(quantile=0.5, alpha=0.01, fit_intercept=True, solver='highs', solver_options=None)
with context-only, HR-only and context+HR design matrices. No new activity feature or steps block.
Reuse matched Notebook 05 Ridge and mean/median dummy predictions after verifying exact cohort and
fold identity. No QuantileRegressor fit on B or C. No alpha tuning.

Reuse training-only StandardScaler (continuous columns, including cyclic time; population variance)
and OneHotEncoder(drop=None, handle_unknown='ignore', sparse_output=False) inside pipelines. No
imputation, missingness predictors or prediction clipping. Alpha across families is **not equivalent**:
Ridge uses squared residual loss plus L2 penalty; QuantileRegressor uses average pinball loss plus
L1 penalty. At quantile .5, pinball loss is half absolute error. This tests this particular fixed
loss/penalty combination, not loss in isolation or universal superiority of median regression.

Record convergence warnings, solver iterations, training objective, coefficient/nonzero counts,
prediction range, SD and unique values (rounded to eight decimals), and equality to the matching
median dummy. Constant predictions are legitimate findings, not grounds for retuning. Capture
convergence warnings as failures; do not score a failed fit. If any of the three median models
fails in a fold, exclude that fold from the entire matched loss comparison and record why. Never
replace a failed quantile model with another solver, alpha or dummy prediction.

## Feasibility, metrics and uncertainty

Do not redraw folds. A refit needs ≥2 training responses and ≥2 training participants plus at least
one validation response. Mark smaller training cohorts infeasible; mark empty validation folds.
An entire comparison is flagged small if <100 responses, <30 participants, or <5 validation
participants in any original fold. Small but numerically feasible fits may be shown descriptively
with the flag; no comparative conclusion from a handful of participants. No relaxing support rules.
Saved predictions remain evaluable even if a support refit is infeasible, with separate counts.

Report pooled and fold MAE (primary), RMSE, and participant-averaged MAE; also pooled errors by
survey category, without category-specific fitting or threshold selection. Morning wording refers
to since waking, while Midday/Endday refer to since the last prompt. No waking time or previous
prompt time is invented from the sequence of completed surveys.

Use 2,000 paired participant-cluster bootstrap resamples with seed 20261005 and percentile 95%
intervals for model-minus-baseline MAE and participant-MAE differences. Compare wearable models
with matching mean, median and context baselines; quantile models also with their matching Ridge
model and context-quantile model. Report refit-minus-saved differences separately. Negative favors
the first model. With fewer than two participants, intervals are missing, explicitly uninformative.
All rows from a sampled participant remain together. These are uncertainty intervals for fixed
development predictions, without model-refitting uncertainty, external validation or multiplicity
adjustment. Many overlapping cohorts/lookbacks/comparisons are exploratory; do not declare an
optimal window, quality threshold or causal mechanism.

## Reproducibility and boundaries

Version ignored cohort/prediction tables by parent manifest, specification, configuration, package
and source hashes. Do not overwrite validated artifacts. Keep all 32 final-test participants sealed;
read assignments only for identity/fold checks and raw/reference bytes only for integrity hashing.
Preserve corrected supersession provenance, all previous notebooks/bundles and assignments. Tests
cover occupancy acceptance/boundaries, policy inclusivity, common membership, unchanged targets,
participant separation, training-only transforms, support-versus-loss separation, empty/small
cohorts, bootstrap pairing, solver status and prediction variability. Run notebook, Ruff, privacy
and integrity verification. No commit, push, provider messaging or final-test evaluation.

Official reference checked against installed scikit-learn 1.9.1:
[QuantileRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html).
Installed source uses scipy.optimize.linprog and issues ConvergenceWarning when the solver fails;
`n_iter_` records the solver iterations. Warnings are checked, not silently ignored.
