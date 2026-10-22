# Notebook 18 — bounded root-cause experiment

Frozen before new performance, following inspection of Notebooks 00–17. This is a
prospectively specified exploratory follow-up, not independent preregistration.
All 3,706 development responses remain accounted for. Use only original participant
folds 0–4; the 32 final-test participants remain sealed. Previous artifacts are immutable.

## Questions and decision limits

Audit the recorded-data implementation independently; distinguish confirmed defects,
unresolved source assumptions, plausible scientific explanations and verified behavior.
Correct only demonstrated defects, with regression tests and versioned replacements.
No reinterpretation of sensor clocks, HRV, interpolation or new model search.

## Targets and comparisons fixed before fitting

Use stored integer 1–5 items; nervousness increases with distress, while coping and
control are also described in their original direction and reverse-oriented as 6−item
for like-direction comparisons. Composite stays nervous+(6−cope)+(6−control).
Never impute labels. Binary targets: nervousness ≥3; reversed coping difficulty ≥3;
reversed control difficulty ≥3; unchanged composite ≥5 and ≥10. The item midpoint
thresholds are engineering contrasts (displayed item ≥2), not clinical diagnoses;
no threshold search. Primary new contrast is elevated nervousness context+HR versus
context; coping/control and existing composite cutoffs are explicitly secondary.

Primary matched cohort: valid composite/all three items, valid context, all four
observed 60-minute HR features AND observed supported step rate. All five input blocks
use this identical cohort for each target, with no feature imputation. This isolates
measured inputs at the cost of selection. No second imputed modeling search.
Blocks: training-prevalence constant; context only; HR only; context+HR;
context+HR+steps. HR mean, sample SD, latest, elapsed-minute slope; context is actual
start cyclic local time and survey category; activity is existing supported step rate.
No quality, identity, current answers, retrospective participant means or history labels
in predictors. Each outer train/validation cohort is identical across blocks/targets.
Logistic L2 C∈{0.1,1}, three participant-grouped inner folds, mean inner AUROC selection,
five frozen outer folds; training-only scaling/encoding, no weights, threshold .5.
Skip infeasible classes/splits without changing membership or thresholds.

Also preserve the ordinal information in each of the three oriented items using a
five-category proportional-odds logistic model: P(Y≤k)=sigmoid(theta_k−X beta), ordered
thresholds, sum negative log likelihood + ||beta||²/(2C), fixed C=1, no intercept in X
(thresholds supply location). These are ordinal analyses, not assumed equal-spaced
linear outcomes. Report ranked probability score, log loss and expected-category MAE;
category distances for MAE remain an approximation. Compare training empirical class
distribution and the same four feature blocks. No ordinal hyperparameter search.
Continuous composite remains a fixed Ridge alpha=1 comparison, with training-mean
and median constants, on exactly the same cohort. This is a reference, not a new target.

## Conditional physical-consistency probe

Construct unique, nonoverlapping UTC-aligned five-minute bins wholly contained in at
least one development EMA's existing preceding 60 minutes; deduplicate repeated EMA
views, never consult PSS to select bins. Accepted HR ≥4 occupied minutes and largest
edge/internal gap ≤90s; all existing HR feature support rules still apply. Counter
pairs must be originally adjacent, accepted and wholly contained. Require union step
support ≥240s and maximum unsupported edge/internal gap ≤60s. Active: ≥20 added steps
per supported minute; inactive: exactly zero additions over supported pairs. Intermediate
rates excluded and counted. These fixed engineering labels are not validated activity
ground truth: unchanged/reset/batched/stale counters and missing 60s remain limitations.
Use HR-only logistic versus training prevalence, same frozen participant folds and
small C grid. No activity input, context or quality input to this probe. Report both
classes, participant variation, support and recording summaries. A successful probe
would establish conditional cross-stream consistency only, not sensor accuracy.

## Descriptive associations and information limits

Reproduce exact within/between sums of squares and retrospective lag correlations;
keep full-record means only in descriptive private tables. Describe scores/items,
survey differences, class mix and availability. For the matched observed cohort use
HR mean only in a prespecified within/between random-intercept descriptive model:
outcome ~ within-person HR mean + person mean HR + time/survey context + supported
step rate + occupied fraction + freshness + maximum gap. Fit Gaussian random-intercept
REML for the continuous composite; for ordinal items use rank-based within/between
descriptive correlations with participant bootstrap, avoiding Gaussian item claims.
Report raw and context/activity/support-adjusted associations, not causal effects.
Uncertainty is conditional on the observed development sample and fixed predictions;
2000 participant-bootstrap prediction draws, 500 descriptive draws. No multiple-search
claim of confirmation or inferred information-theoretic ceiling.

## Recall gate and verification

Reuse exact questionnaire/PDF evidence, existing retrospective Morning audit and
matched Notebook 08 results; do not refit current-answer-defined windows as prospective.
Only a verified boundary and boundary-availability clock preceding cutoff, complete
prompt provenance (where relevant), verified UTC conversion and sensor event-time
contract can open a new aligned prospective extraction. Missing evidence blocks it;
current-response wake answers and previous completed surveys are not substitutes.
Provide a tested gated extractor with synthetic boundaries and a field-level request.

Verify source hashes, raw selected development records, row provenance, crosswalk,
offsets/buffers, all eight lookback boundaries and feature formulas on deterministic
samples independent of outcomes; replay saved model predictions and check training-only
transforms. Preserve raw bytes, previous hashes, frozen folds and private outputs.
No commit, push, provider contact or final-test evaluation.

Method references: [scikit-learn logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[ordinal logistic parameterization](https://www.statsmodels.org/stable/examples/notebooks/generated/ordinal_regression.html),
[random-intercept mixed models](https://www.statsmodels.org/stable/_modules/statsmodels/regression/mixed_linear_model.html).
The existing environment supplies SciPy/sklearn; custom ordinal and REML arithmetic
must pass independent numerical tests. No new dependency is required.
