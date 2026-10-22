# Notebook 09 — frozen exploratory information audit and benchmark

Specified after inspecting Notebooks 04–08, before new performance. This is repeated exploratory
development analysis, not independent preregistration, a search for positive findings, a best-window
selection or validated real-time prediction. All 32 final-test participants remain sealed.

## Information audit (descriptive, no new predictor)

Use Notebook 08's primary matched Morning wake-versus-60 cohort and its all-four-window common
cohort, separately for reported-minute start and +60-second wake sensitivity. Compare all six pairs
among wake, 5, 60 and 120 minutes. Preserve the retrospective-only status of reported wake time.
Reading identity is the original source-content hash plus source row, within verified participant.
Use accepted logical readings from the frozen disposition ledger; retain legitimate repeated values
at different timestamps. Report shared and each-window-only counts, Jaccard overlap, identical sets,
latest source identity/value, signed and absolute feature differences, Pearson/Spearman correlations.
Complete-vector denominators exclude missing features explicitly. Numerical identity uses absolute
1e-10 tolerance (zero relative tolerance). Descriptive near identity means all mean/SD/latest
changes ≤1 bpm and slope change ≤0.1 bpm/min; this is not an eligibility/clinical threshold.

Within each window, minute bins are anchored at its left boundary, including a possibly partial last
bin. Leave empty bins missing. Describe counts per occupied minute, count CV, and the share of
readings and weighted-mean contribution in the densest ceil(10% of occupied bins), ties by earliest
bin. Compare the existing reading mean with the equally weighted average of observed minute means;
this alternative is diagnostic only, not continuous-time coverage or a benchmark input.

Independently trace two distinct participants per frozen fold: ascending SHA256 of stable EMA key,
first eligible key per participant on nominal common cohort. Verify all four windows in both wake
scenarios against original development-only HR source rows and normalized EMA clocks/identity.
Use Python datetime arithmetic, Boolean interval masks and scalar statistics independently of the
vectorized generator; verify accepted-source sets, saved lineage slices, keys, folds and raw values.
Do not display participant keys or rows. Add an end-to-end synthetic positive control through actual
clock parsing, HR acceptance/window summaries, strict outcome linkage and held-out model fitting;
synthetic performance is not evidence of real physiological validity.

## Fixed benchmark and matching

Use Notebook 05's unchanged native 60-minute complete HR/outcome/context cohort across all surveys.
Secondary requires existing supported step additions; refit every comparator on that exact cohort.
Use original five outer participant folds. Inputs: actual-start local-time sin/cos and survey
category; add HR mean, sample SD, latest and slope. On secondary additionally compare context+HR+steps
with context+HR on matching rows. No new features, quality/missingness columns, answers, identifiers,
HRV, Garmin stress, imputation, outcome recoding or binary targets. Retain original support rules.
Sixty minutes is a fixed benchmark, not an optimal or uniformly recall-aligned window.

Models: training-mean/median dummies; fixed Ridge alpha=1; Ridge tuning in ordered [0.1,1,10,100];
HistGradientBoostingRegressor absolute-error loss, learning_rate=.05, min_samples_leaf=20,
l2_regularization=1, early_stopping=False, categorical_features=None, random_state=20261006.
HGB grid ordered (depth,iterations) = (1,50),(1,150),(2,50),(2,150); other installed defaults fixed
and recorded. Compare context and context+HR for each family; secondary also context+HR+steps.
No random row-level early-stopping holdout. Require complete finite predictors even though HGB can
support missing values. Do not clip predictions.

Inside each outer training set use GroupKFold(3,shuffle=False), groups=participant, input sorted by
stable EMA key. Use identical inner splits for all models/configurations. Select by arithmetic mean
of three validation-fold MAEs only; exact ties choose first grid entry. Each split fits its own
StandardScaler (population variance) and dense OneHotEncoder(handle_unknown='ignore'), remainder
'drop'. Refit selected settings on the entire outer training cohort. Outer validation outcomes never
enter tuning. Skip a cohort-fold for all comparators if validation is empty or fewer than three
training participants, or any inner split lacks two training participants; never redraw/relax.
Ridge squared-error/L2 and HGB absolute-error/tree fitting differ in loss as well as function class;
any gain cannot be attributed solely to nonlinearity. Single-thread numerical fitting is reproducible.

## Evaluation and safeguards

Report train and OOF MAE/RMSE/equal-participant MAE, fold results, category errors, cohort/fold counts,
selected settings/inner MAE, prediction SD/range and training-transform checks. Training results are
in-sample repeated outer-training memberships, not independent observations. Pair each learned model
against mean/median and its matching family context; secondary steps also against that family's
context+HR. Compare tuned families with fixed Ridge on the same inputs. Use 2,000 paired participant
bootstrap draws, seed 20261002, percentile 95% intervals, no refitting. Intervals condition on this
exploratory workflow and fixed development predictions; they omit tuning/refitting uncertainty,
external validation and multiplicity adjustment. No automatic expanded search or winning window.

Preserve all prior inputs/results. Verify corrected provenance, source integrity, original assignments,
training-only transformations, nested separation, existing fixed predictions, metrics, tests, Ruff,
notebook and privacy. Source notebook has no outputs; private row-level datasets remain ignored.
No commit, push, provider messages or final-test evaluation.

Implementation references inspected for scikit-learn 1.9.1:
[HGB parameters and absolute-error loss](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html),
[GroupKFold participant separation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
