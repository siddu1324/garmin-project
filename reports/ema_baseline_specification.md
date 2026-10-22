# Notebook 05 frozen exploratory baseline specification

Fixed before calculating development scores or fitting models. The scope is exploratory prediction
from recorded pre-EMA data, not validated real-time prediction. Preserve every development EMA and
its frozen fold. No final-test response or sensor data may enter this checkpoint.

## Outcome and context

Raw export fields are PSS_NERVOUS, PSS_COPE, PSS_CONTROL in all three survey categories. Questionnaire
names have suffix _1 for Morning, no suffix for Midday and _4 for Endday. All show stored codes 1–5
paired with displayed labels 0–4. Require all three finite integer codes in 1–5; compute
`pss_combo = nervous + (6 - cope) + (6 - control)` (range 3–15). Preserve original strings, individual
item status and an explicit missing/invalid reason. This is the study's three-item perceived-stress
composite, not the full PSS or a clinically validated diagnostic cutoff. No binary target.

The blank survey PDFs define five response codes but do not document exporter missing-value
sentinels. Empty/whitespace and null cells are missing; the existing inventory tokens NA, N/A, NaN,
nan, NULL, null, -99 and -999 are conservatively excluded as missing-token candidates, with their
literal spelling preserved. No unconfirmed sentinel becomes a score. Any other malformed,
nonfinite, fractional or out-of-range value is invalid. Report observed token counts, and distinguish
document-confirmed codes from missing-token assumptions. Never infer a missing answer from another.

Stable EMA key joins must be one-to-one, cover all 3,706 rows, and match participant, partition,
category, cutoff and original fold. Actual local start supplies sin/cos(2π × seconds-of-day/86400),
including seconds; never use scheduled start or buffered cutoff for context. Survey category is
one-hot encoded. Context is an explicitly authorized comparison, separate from wearable predictors.

## Fixed models and matching

Use 5, 15, 30, 60, 120, 360, 720 and 1440 minutes separately, with unchanged Notebook 04 features
and support rules. HR block: mean, sample SD, latest and slope. Secondary block adds supported step
additions. Floors are deferred. Exclude all other features, including quality, identifiers, answers,
Garmin stress, IBI/HRV, calories and intensity. Keep quality in a separate diagnostic table.

For each lookback, primary native cohort requires observed outcome, valid context and all four HR
values. Secondary native cohort additionally requires observed supported steps. On the primary
cohort compare training-mean dummy, training-median dummy, context-only Ridge, HR-only Ridge and
context+HR Ridge. On the secondary cohort refit all five on that exact cohort, and add context+HR+steps
Ridge. Every model within a comparison uses identical training and validation responses. Do not
compare performance across different native cohorts as if it measured an incremental feature gain.

Also form two common cohorts: the intersection across all eight lookbacks within each block. Refit
all models using the common cohort for BOTH training and validation. Report all lookbacks without
choosing a winner. Common results are flagged as too limited for comparative interpretation if
there are fewer than 100 responses, 30 participants or five validation participants in any fold.
These descriptive flags never change membership or support rules. Mathematically fit a fold only
with at least two training responses and two training participants and at least one validation
response; otherwise retain a shared skipped-fold reason for every model.

Use existing five participant-grouped folds, no resplitting. Fit numerical StandardScaler and
OneHotEncoder(drop=None, handle_unknown='ignore', sparse_output=False) on training rows only within
a Pipeline/ColumnTransformer, remainder='drop'. Scale all continuous columns, including sin/cos;
do not scale dummy variables. StandardScaler uses population variance; source HR SD uses N−1.
Ridge alpha=1, fit_intercept=True, solver='svd', no tuning. Dummies use matching training outcomes.
No prediction clipping, imputation, missingness predictors, participant weighting during fitting,
feature selection, new thresholds, extra models, or final lookback selection.

## Metrics and uncertainty

MAE is primary; RMSE secondary. Report each fold and pooled fixed out-of-fold predictions; also
average each participant's MAE equally. Cohort/fold tables report training and validation response
and participant counts, including skipped folds. Report score distribution and feature-available
versus unavailable survey composition/PSS descriptively (no selection based on these associations).
HR reading count, occupied-minute fraction, freshness, span and gaps and supported step duration
remain quality diagnostics only. Two readings do not establish representative coverage.

For every Ridge model, compare paired absolute errors against matching mean, median and context
baselines (omit comparison with itself); additionally compare steps model with context+HR on the
same secondary cohort. Use 2,000 participant-cluster bootstrap resamples, fixed seed 20261002,
resampling participants with all their available fixed predictions. Report percentile 95% intervals
for model-minus-baseline pooled response MAE and equally weighted participant MAE; negative means
lower error. Pair the same sampled participants for both models. These intervals describe uncertainty
in fixed development predictions, not model-refitting uncertainty, independent EMA observations,
external validation, multiplicity-adjusted significance or causal effects. No winning lookback.

## Provenance and verification

Use current corrected Notebook 04 feature bundle and its corrected eligibility parent, verifying
artifact hashes against the explicit supersession record. Preserve old artifacts. Freeze this file
and config hashes in a private pre-fit record. Version new linked data and predictions by source,
configuration, specification, code and package hashes; never overwrite a validated bundle. Store
all EMA/participant/prediction rows privately and expose aggregates or clearly synthetic examples.
Verify coding, joins, exclusions, fold separation, training-only transforms, deterministic bootstrap,
input integrity, notebook execution, privacy and prior-file preservation.

Implementation references (official documentation, checked for installed scikit-learn 1.9.1):
[Pipeline and ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html),
[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html),
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).
