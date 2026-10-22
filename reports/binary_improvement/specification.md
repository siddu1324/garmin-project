# Notebook 16 — fixed exploratory experiment specification

This follow-up follows inspection of Notebooks 00–15. It is not independent preregistration.
Freeze this document and `config/binary_improvement.json` before new held-out performance.
The aspiration of AUROC 0.70–0.75 is not a selection or stopping target.

## Questions and comparisons

Binary A is composite ≥5; Binary B is ≥10. Require all three original 1–5 items:
`PSS_NERVOUS + (6-PSS_COPE) + (6-PSS_CONTROL)`. No target imputation or clinical claim.
All 3,706 development EMAs remain accounted for; main comparisons use all valid-score/context
responses. Keep the 125 development participants and their five frozen folds; final test stays sealed.

Primary paired AUROC contrasts, separately by task: full 720 minus full 360; full 1440 minus
full 360; numeric compact SMOTE minus numeric compact unbalanced; contextual conditional
SMOTE minus contextual compact unbalanced; observed-minute versus reading-weighted HR4
at 360 minutes. Also compare each physiology model with its matching context and dummy.
All other comparisons, including the largest point AUROC, are descriptive exploratory sensitivities.
Intervals are conditional participant bootstraps, not multiplicity-adjusted confirmation.

## Feature experiments

Use unchanged supported feature definitions at 60/120/360/720/1440 minutes: original HR4,
expanded HR30, HR+steps+floors44, full single-window47, and sparse logistic selection of47.
Every main feature block includes cyclic local time and survey category. Context is shared across
lookbacks because its inputs and cohorts are identical. No PulseOx, IBI/HRV, Garmin stress,
calories, identifiers, current answers or quality columns in the physiological blocks.

Additional fixed representations reuse saved columns: observed-minute mean/SD instead of
reading mean/SD (360 and1440); HR4 plus half-window change, observed-minute SD and rolling-block
variation; HR4 plus earlier-reference mean and current mean/latest deviations (360 and1440);
HR4 plus adjacent nonoverlapping mean contrasts (60/360/720 scales); HR4 plus three coobserved
step-stratum summaries. Existing activity overlap is a descriptive stratum, not causal adjustment.
Saved slopes retain ≥3 readings and ≥60-second span; do not change support after performance.
The separate compact multiscale block contains HR4 at60/360/1440, past-only60-minute reference
mean and disjoint contrasts360/720. No selection of scales from outer results.

Separate controls: context+missingness only; context+five HR recording-quality columns;
full360+quality; full360 without missingness flags. A fixed supported cohort requires existing
HR4, ≥80% accepted occupied minute bins and latest accepted reading ≤60seconds old at360.
Refit context, HR4 and full360 on identical supported training/validation rows; also restrict saved
all-row predictions to those validation rows. No inference that occupied bins imply continuous recording.

## Rebalancing and interpolation

Use an explicit equivalent leakage-safe sampler, with donor lineage saved privately, rather than
altering the pinned dependency environment. Numeric compact bank in fixed order: HR mean, SD,
latest, slope, supported step rate, observed-minute SD, zero-step-stratum HR, half-window change.
Within each training split keep columns with ≥50% observed values and nonzero variance, greedily
remove absolute correlations >0.95, then cap at six. No validation or outcome-based selection.
Fit median imputation and scaling on original training rows before augmentation.

Numeric-only compact comparison: none, balanced class weights, random oversampling,
random undersampling, standard k≤5 nearest-neighbor SMOTE. No context or missingness indicators
are interpolated in this numeric-only comparison. It is not the primary context-adjusted model.
Random balancing targets equal class counts. SMOTE uses uniform minority bases, uniform eligible
neighbors, and uniform interpolation fractions; all donors come from the current training split.

Contextual compact comparison: none, balanced weights, random over/undersampling, and a
**custom conditional SMOTE variant**, not a claim of stock SMOTENC. It interpolates only selected
physiological coordinates between same-participant, same-survey-category, same-missing-pattern
minority observations. Context comes unchanged from the base donor. If no eligible neighbor exists,
duplicate the base and count it as a duplicate, not a synthetic interpolation. This tests whether
constraining cross-person mixing changes feasibility/performance. No complete physiological
plausibility claim follows from convex numeric bounds. Report cross-person/category mixing,
imputed-donor use, fallback duplication and original/resampled counts in every inner/outer fit.
Rows are training augmentation, never additional independent EMAs or manufactured sensor readings.

## Tuning, thresholds and calibration

Five frozen participant outer folds; three shuffled StratifiedGroupKFold inner folds, seed20261019.
For ordinary logistic fits use C={.001,.01,.1,1}, l1_ratio={0,.5,1}; sparse restricts ratio>.0.
L2 uses lbfgs, sparse uses saga, max_iter10000, tol1e-4. Select by mean inner-fold AUROC only,
first-grid deterministic ties. Exclude nonconverged candidates; do not expand the grid.
Class weighting and samplers are fixed sensitivity arms, not outer-selected hyperparameters.
Reuse Notebook15 selected configurations/trials where inputs, transforms, grid and cohort match;
refit and verify exact saved predictions. The 60-row experiment registry caps work at600 outer
fits and at most12 candidates per new ordinary fit. No secondary boosting search: earlier studies
already tested it, and new measurement/representation evidence is the focus of this checkpoint.

For every model, derive a balanced-accuracy decision threshold from its selected configuration's
inner OOF probabilities only; ties favor the threshold closest to0.5. Report both0.5 and this
threshold. Reusing inner predictions for tuning and thresholding can overfit training decisions;
outer validation remains untouched. Do not select a threshold from the test or outer responses.
Report Brier score, mean probability/prevalence and fixed probability-bin calibration curves.
For rebalanced fits also report analytic prior-odds correction using only original/effective
training prevalence, as a sensitivity, not validated calibration. It cannot improve within-fold
ranking; it can affect pooled ranking through fold-specific corrections.

## Timing, diagnostics and controls

Audit score/crosswalk/fold identity, original actual-local minus row offset, precision buffer,
half-open windows and source-row lineage with independent datetime arithmetic. Trace a deterministic
private sample to accepted readings; show aggregates and synthetic examples only. Preserve old
artifacts; document/test any genuine implementation correction before new fits.

Report counts, occupied bins, freshness, largest gaps and per-feature/class missingness for every
lookback. Descriptive class contrasts include context-residualized and participant-equal summaries,
leave-one-participant-out ranges and correlation/constant diagnostics. They never enter predictors.
Three fixed training-label permutations (seeds7/19/31, C=.01 L2) are computational negative controls,
not exchangeability-valid repeated-measures permutation significance tests. Outer labels are intact.

Morning-only retrospective sensitivity reuses Notebook08's two documented wake scenarios, with
context/HR4fixed60/HR4wake trained/evaluated on each scenario's identical complete intersection.
Wake is learned from the current survey; it is never a predictor or pre-response information.
Same-day and constant-offset assumptions remain explicit. No previous completed EMA substitutes
for a missing previous-prompt log in Midday/Endday. Do not invent aligned prospective windows.

Report pooled and fold AUROC, AP with prevalence, balanced accuracy, both-class precision/recall/F1,
sensitivity/specificity, confusion matrices, ROC/PR, Brier and calibration. Use2,000 paired participant
bootstrap draws for primary ranking/BA differences. Report fold calibration and leave-one-person-out
influence; no outer-fold winner is called validated. Save all private memberships/predictions/models
and aggregate tables with immutable version hashes. No raw changes, commit, push or provider message.

## Source documentation

[Original SMOTE paper](https://arxiv.org/abs/1106.1813) motivates minority-neighbor interpolation;
it does not establish physiological validity for this study.
[imbalanced-learn leakage guidance](https://imbalanced-learn.org/stable/common_pitfalls.html)
requires resampling inside training splits. Its [oversampling guide](https://imbalanced-learn.org/stable/over_sampling.html)
distinguishes numeric interpolation from mixed categorical handling.
[scikit-learn threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html)
separates probability ranking from action thresholds and cautions against tuning on evaluation data.
