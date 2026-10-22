# Notebook 15 — fixed exploratory binary logistic experiment

Specified 2026-10-19, after inspection of Notebooks 12–14 and before new performance
calculations. This follows repeated development exploration, not independent confirmation.
Primary question: does a single six-hour physiological window add discrimination beyond
context and improve on 60 minutes? The 120-minute comparison is secondary. No regression,
boosting, clinical cutoff claim or final-test evaluation is authorized in this checkpoint.

## Outcomes, inputs and cohorts

Reuse the sealed Notebook 13 development feature matrix and original assignments. Account for
all 3,706 EMAs. Binary A: 0=3–4, 1=5–15; Binary B: 0=3–9, 1=10–15. Recheck the saved targets
against the unchanged valid composite; never impute outcomes. All main models use the same
valid-outcome/context responses (expected 3,689), preserving the five participant folds.

Use the 438 supported candidate universe; exclude the 240 provisional PulseOx columns, HRV,
Garmin stress, participant identifiers and current-survey answers from predictors. Context
is actual-local time sine/cosine plus survey category. Every physiological comparator also
contains context, so its increment is interpretable on identical rows.

For each 60/120/360-minute window use exactly:

- Original: four HR summaries (mean, sample SD, latest, slope).
- Expanded HR: the existing 30 within-window HR summaries.
- HR + activity: expanded HR plus seven supported steps and seven floors summaries (44).
- Full single-window: those 44 plus three within-window co-observation contrasts (47).
- Sparse selection: the same 47, restricting logistic regularization to L1/Elastic Net.

Earlier-history references and adjacent-window differences are excluded from single-window
blocks even when their names end in that window size. Their time support extends beyond
the stated interval. The complete 438-feature multiscale model is a separate unweighted
sensitivity only, not evidence specifically for six hours. Frozen cutoff buffers, support
requirements, accepted readings, counter rules and corrected [start,end) logic are unchanged.

## Modeling fixed before execution

Logistic regression only, plus a training-majority dummy. Main blocks and context each run
unweighted and with class_weight=balanced, estimated from that training split only. Balanced
and unweighted are separate comparisons, not an outer-test-selected choice.

Nested tuning: three StratifiedGroupKFold inner splits, shuffle=True and seed=20261019,
inside each frozen outer training set; same cohort/task receives the same inner memberships.
Select maximum unweighted mean inner AUROC over C={.001,.01,.1,1} and l1_ratio={0,.5,1}
(L2, Elastic Net, L1). Deterministic ties choose first grid entry. L2 uses lbfgs; sparse
penalties use saga. max_iter=10000, tolerance=1e-4. No grid expansion, probability calibration,
threshold search or row-level early stopping. Classification threshold is fixed at p>=0.5.
Reject nonconverged candidates; if none is valid or outer refit fails, report the job as
infeasible/failed, without redrawing folds or silently replacing predictions.

Reuse Notebook 13's train-only transformation: remove training-all-missing/constant columns,
fit training medians, standardize numerical values, train survey encoder, and append only
training-variable missingness indicators. Indicators are recording-process covariates, not
physiology. Medians fill feature summaries, never targets or raw readings.

Separately, for each window, fit these unweighted full-block sensitivities:

1. Correlation filtering: drop later physiological numeric columns with absolute training
   imputed Pearson correlation >.95 in dictionary order, retaining context.
2. Availability filtering: retain physiological features observed in >=50% of training rows,
   with the same filter refitted inside each inner split. This is a provisional engineering
   feature-availability rule, not a validated recording-quality standard or row exclusion.
3. No missingness flags: training-median imputation still used.
4. Missingness-only control plus context: no physiological values.

Add a separate unweighted 360-minute quality-only context control: accepted count/fraction,
occupied-minute fraction, span, maximum gap and latest age. Quality columns never enter
physiological blocks. Assess whether apparent discrimination may reflect recording patterns;
these comparisons cannot causally identify why data are missing.

Complete-case sensitivity: intersection with all original four HR features available at all
three windows. Refit context and each window's full block on identical eligible training and
validation rows (other full-block features retain training-only imputation). Report native
original/full complete-case coverage separately, and evaluate all-valid fitted models on the
same common validation subset separately from these refits. Do not compare differing cohorts
as if a performance change isolated a physiological-window effect.

## Metrics, uncertainty and diagnostics

AUROC primary; average precision (noninterpolated PR area), balanced accuracy, both-class
precision/recall/F1/support, sensitivity, specificity, accuracy, Brier score, confusion matrices,
ROC/PR curves and prediction spread. Report pooled OOF and fold-level metrics, training versus
validation discrimination, exact train/validation class and participant support per fold,
and survey-specific errors. For Binary A, positive recall is not high-score-only detection.
For the majority dummy, use constant hard-class probabilities: AUROC=.5 when both classes
are present and AP equals prevalence. Do not treat the dummy's accuracy as discrimination.

Use 2,000 deterministic paired participant-cluster bootstrap draws with identical resampled
participants for matched models; all responses for a sampled participant receive the same
multiplicity. Report absolute and difference intervals for AUROC, AP and balanced accuracy.
Primary contrasts: full360 versus full60, and full360 versus context, unweighted, separately
for each task. Also report prespecified feature, weighting and handling contrasts. Intervals
condition on fixed OOF predictions; they omit retuning uncertainty and multiplicity correction.
Report undefined/one-class draws rather than replacing them with favorable values. Report
fold AUROC alongside pooled AUROC because fold-specific calibration can affect pooled ranking.
AP is prevalence-sensitive: do not rank the two target definitions by raw AP alone.

Quality diagnostics reuse saved quality and lineage tables: counts, occupied minutes, span,
edge/internal/maximum gaps, freshness, per-feature and class-specific missingness, retained/
excluded cohort composition. Nested windows with identical end/source and contained lineage
ranges allow accepted-count differences to quantify genuinely additional accepted readings.
Check those identities explicitly; do not invent continuous coverage from point samples or
equate the nominal six-hour interval with six hours of monitoring. No sensor re-extraction.

Coefficient magnitudes/selection frequencies are training-fit descriptive summaries, not
causal importance or validation-selected features. Publish aggregate tables and figures only;
keep rows, keys, actual clocks, memberships and models private. Preserve all old artifacts and
seal the new version. Verify replay, grouped splitting, preprocessing, target definitions,
single-window allowlists, numerical metrics, privacy and raw-file metadata integrity.

## Documentation

The installed scikit-learn version is recorded with each run. Its current
[LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
defines l1_ratio, solver compatibility and training-frequency balanced weights.
[Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
uses recall increments without trapezoidal interpolation; it is reported as AP, not mislabeled
as trapezoidal PR area. [AUROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
uses continuous positive-class probabilities, not thresholded labels.
