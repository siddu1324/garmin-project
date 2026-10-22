# Numerical definitions and interpretation limits

This appendix documents the implementation of the frozen specification; it does not change
targets, cohorts, support rules, model settings or comparisons after seeing results.

- **Binary metrics:** AUROC uses predicted positive probabilities; AUPRC is sklearn's
  noninterpolated average precision (AP), not trapezoidal PR area. Sensitivity/specificity
  and balanced accuracy use the fixed probability threshold0.5. The constant baseline uses
  that training fold's observed prevalence, never validation prevalence. Brier score and
  fixed-width probability-bin calibration summaries are descriptive; no calibration fit
  is applied using held-out labels. Always interpret AP beside prevalence.
- **Ordinal model:** five ordered categories, shared slope vector, four increasing logistic
  cutpoints. Likelihood is the difference of consecutive logistic CDFs; objective is summed
  negative log likelihood plus ||beta||²/(2C), C=1. Cutpoints are unpenalized; no predictor
  intercept or redundant full survey dummy set. The proportional-odds assumption is a
  bounded working model, not demonstrated truth. Ranked probability score is the mean
  squared CDF error over four cutpoints (normalized by4); lower is better. Expected-category
  MAE additionally treats adjacent codes as equally spaced, an approximation for ordinal items.
  Optimizer convergence/gradients are recorded. Analytic gradients pass finite-difference
  checks; independent known probabilities and held-out synthetic recovery are tested.
- **Continuous composite:** unchanged1–5 item coding and3–15 sum; fixed Ridge alpha1,
  mean/median training constants. No binary target replaces this composite. Numeric predictors
  are scaled only in the corresponding training split; all compared responses have observed
  features. The experiment does not repeat a feature-imputation search.
- **Descriptive correlations:** pooled ranks, within-person centered ranks, and correlations
  of participant mean ranks are distinct quantities. Nuisance adjustment includes survey/time,
  observed step rate and support diagnostics; within-person adjusted estimates center those
  terms as well. These are partial rank associations, not a causal effect or individual-level
  forecast. Whole-record person means never enter prediction.
- **Random intercept:** continuous composite Gaussian REML with within/between HR mean terms,
  time/survey/activity/support covariates. The profile implementation uses
  V=sigma²(I+rZZ′); a dense covariance calculation independently reproduces coefficients and
  covariance in a synthetic test. Reported coefficient SEs are model-based, not robust
  participant-bootstrap SEs. Rank analyses supply separate clustered descriptive uncertainty.
- **Variance:** observed response-weighted sums of squares partition into within-person and
  between-person terms; contributions divided byN−1 sum to the observed sample variance.
  These fractions are not predictive R², reliability estimates, or a physiological information
  ceiling. The random-intercept ICC is a different, conditional model quantity.
- **Bootstrap:**2,000 whole-participant draws for fixed prediction comparisons;500 for
  descriptive associations. Duplicate sampled participants get distinct bootstrap cluster IDs
  in descriptive centering. Matched prediction contrasts reuse exactly the same participant
  draws and response order. Activity-probe draws missing either class are omitted and counted
  (1,993 valid ranking draws). Four people with both activity labels cannot yield a stable
  physical-validation conclusion. No correction for the repeated exploratory workflow is claimed.
- **Pooling:** pooled OOF AUROC includes cross-fold positive/negative pairs from different fitted
  models; within-fold AUROC weights each feasible fold by its positive×negative pair count.
  Folds with one class have undefined AUROC and zero pair weight. No probabilities are adjusted
  using validation outcomes to make pooled and within-fold results agree.
- **Recorded support:** the activity probe deduplicates source-time bins. Observation duration
  is300s; counter support is the union of wholly contained accepted original pairs, at least240s.
  HR occupied bins, edge/internal gaps and freshness are not continuously observed duration.
  Zero supported counter additions do not prove absence of movement. Neither a positive probe
  nor numerical pipeline agreement establishes the source's physiological validity.

Official method references: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[ordinal logistic likelihood](https://www.statsmodels.org/stable/examples/notebooks/generated/ordinal_regression.html),
[linear mixed-model implementation reference](https://www.statsmodels.org/stable/_modules/statsmodels/regression/mixed_linear_model.html).
The local SciPy implementation is independently tested; statsmodels is not installed or claimed
as the fitted package. Exact installed package versions and source snapshots are recorded privately.
