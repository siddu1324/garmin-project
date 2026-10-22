# Notebook 14: frozen comprehensive PSS regression benchmark

Specified 2026-10-18 after reviewing Notebooks 05–13, before fitting new real-data regressions.
This is a prospectively specified exploratory follow-up, not independent confirmation. No test-set
evaluation, raw reprocessing, new physiological extraction, clinical cutoffs or provider messages.

## Inputs and cohorts

Reuse the checksum-verified 3,706-row Notebook 13 matrix, original 125 development participants,
stable EMA keys and five frozen participant folds. The target is the unchanged continuous three-item
composite, 3–15. Never impute outcomes. Account for every EMA; model valid outcome/context rows (expected
3,689). Keep all 32 final-test participants sealed. Preserve 438 supported observed-summary candidates;
240 provisional PulseOx columns enter only a separately labeled sensitivity. These are arithmetically
supported summaries, not proof of acquisition-time or physiological validity. No HRV, proprietary Garmin
stress, current-survey answers, identifiers or quality variables enter physiological predictor blocks.

Context is actual-local start sine/cosine plus training-encoded survey category. All main blocks also
include context: original four HR summaries + supported steps at 60 minutes; expanded HR60; HR across
eight windows; HR+activity60; HR+activity across eight windows; full 438; full with correlation filtering.
All main comparisons share identical training and validation rows. Preserve existing feature support,
cutoff buffers, timing, lineage, earlier-recording history and [start,end) boundaries unchanged.

Use Notebook 13's training-only transform: discard training-all-missing/constant columns, fit feature
medians, scale numerical values, encode survey category, and add only training-variable missingness flags.
Do not fill raw records or create sensor coverage. Correlation filtering greedily excludes later numerical
physiological columns with |Pearson r|>0.95 on training-imputed values in dictionary order; retain context.
Missingness indicators stay separately named. No validation-dependent feature removal or preprocessing.

## Models, tuning and bounded experiments

Dummy mean and median fit each training cohort. All seven families (Ridge, Lasso, Elastic Net, Huber,
histogram gradient boosting, XGBoost, LightGBM) run on the eight main blocks above. Fixed-order grids:

- Ridge alpha = 0.1, 1, 10, 100, 1000; squared error, deterministic solver.
- Lasso alpha = 0.01, 0.1, 1; Elastic Net same alpha, l1_ratio=0.5. max_iter=10000, tol=1e-4.
- Huber alpha = 1, 100 × epsilon=1.35, 2; max_iter=2000, tol=1e-5. This is a robust mixed loss.
- HGB loss = squared_error, absolute_error × max_depth=1,2 × max_iter=50,150;
  learning_rate=.05, min_samples_leaf=20, l2_regularization=1, early_stopping=False.
- XGBoost objectives reg:squarederror and reg:absoluteerror, each with four fixed profiles
  (depth,trees,min_child_weight,reg_lambda): (2,100,20,10), (2,300,10,1), (4,100,20,10),
  (4,300,10,1). CPU hist, max_bin=64, learning_rate=.05, colsample_bytree=.8, subsample=1.
- LightGBM objectives regression and regression_l1, each with four profiles
  (leaves,trees,min_child_samples,reg_lambda): (7,100,40,10), (7,300,20,1), (15,100,40,10),
  (15,300,20,1). learning_rate=.05, max_bin=63, colsample_bytree=.8, subsample=1,
  deterministic=True, force_col_wise=True. The profile sets are bounded joint grids, not exhaustive searches.

All libraries use one thread per fit, seed 20261018 and no early-stopping validation split. No prediction
clipping, target scaling, weighting of high outcomes or hyperparameter expansion after seeing results.
Different model losses mean any gain cannot be attributed solely to nonlinearity.

Within each outer training fold, stable-EMA-key order and GroupKFold(3, shuffle=False) define identical
participant-grouped inner folds. Every transform and selector is refitted inside each inner training set.
Select lowest arithmetic mean inner MAE; exact ties choose the first candidate. Refit on all outer training
rows; outer validation outcomes are excluded from the tuning API. A nonconverged candidate is ineligible;
if no candidate or selected refit converges, mark that job failed/infeasible without substituting another
model, redrawing folds or inventing predictions. Report failures and use matched available folds only.

Prespecified secondary analyses:

1. Full Lasso/Elastic Net nonzero selection followed by Ridge, HGB, XGBoost or LightGBM. Selector alpha=.05,
   ratio=1 or .5, coefficient magnitude>1e-8, fitted on each training split. Preserve context even if all
   physiological coefficients vanish. This selector strength is fixed, not optimized on held-out data.
2. Full+PulseOx and full without missingness flags for those same four model families.
3. Separate context+missingness-only and context+quality-only controls for Ridge and HGB. Quality is not
   physiology. Use the existing quality allowlist from Notebook 13, never newly derived raw measures.
4. Single-window expanded HR at the seven other approved lookbacks for Ridge and HGB (HR60 is already
   primary). This tests scale on identical cohorts, not a post-hoc optimal window.
5. Original 60-minute five-feature complete-case cohort: refit context, original and full for Ridge and
   HGB; include matching mean/median dummies and fixed-alpha=1 Ridge context/original references. Compare
   saved all-valid-model predictions on exactly these validation responses separately from retraining.
   Reproduce Notebook 05 fixed Ridge on this cohort against saved predictions; compare Notebook 09 saved
   tuned Ridge/HGB only on matching EMA keys, targets and outer folds. Do not compare cross-cohort headline
   metrics. Full complete-case feasibility is reported; do not exclude nearly everybody to force it.

## Evaluation and interpretation

MAE primary; RMSE, R², Spearman, participant-averaged MAE, fold stability, prediction SD/range, and fraction
outside 3–15 secondary. Report pooled OOF and in-sample training errors distinctly. Report errors and mean
signed residual (prediction−actual) separately for scores 3–4, 5–9, 10–15 and survey category; these groups
are diagnostics, not alternative targets or new fitting weights. Spearman is undefined for constant
predictions. Actual-versus-predicted plots must be aggregate hexbin/count summaries; never participant
rows or actual timestamps. Include prediction histograms, residual summaries and high-score underprediction.

Use 2,000 fixed-seed participant-cluster bootstrap draws for absolute MAE/RMSE/R²/participant-MAE intervals
and paired MAE/RMSE/participant-MAE differences on identical OOF keys. Principal contrasts: each model
versus matching median/mean and same-family context; full versus original and HR-all; selected versus full;
nonlinear full versus Ridge full; sensitivity versus corresponding full. Negative differences favor the
first model. Intervals condition on fixed OOF predictions and do not include refitting/search uncertainty,
multiple-comparison correction, independent replication or final-test evidence. Missing/infeasible folds
must never silently enter a complete-case model ranking.

Interpret sparse readings, missingness-only controls, history requirements and recall mismatch as possible
limitations, not identified causal explanations. Reuse prior aggregate recording diagnostics. Do not
claim that preprocessing errors explain weak results without a concrete demonstrated bug. Stop automatic
model expansion after this planned benchmark regardless of performance.

## Reproducibility and sources

Pin configuration, source, environment, specification, feature-manifest and frozen-assignment hashes before
fitting. Save immutable resumable per-fold jobs, private OOF predictions/estimators/memberships; aggregate
public metrics and selection stability only. Notebook 14 is explicitly authorized to retain aggregate
execution output. No prior artifacts overwritten, no commits or pushes.

Installed versions: scikit-learn 1.9.1, XGBoost 3.0.5, LightGBM 4.6.0. Parameter meanings checked against
installed interfaces and official documentation:
[Huber](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html),
[HGB](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html),
[XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html),
[LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
