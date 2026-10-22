# Notebook 12 — prospective specification of an exploratory classification follow-up

Written before classification performance is computed, after examining regression checkpoints 04–11.
This is a new exploratory direction, not an independent preregistered confirmation. Repeated development
exploration and nested tuning do not establish external validity. The final 32 participants remain sealed.
No pregnancy/postpartum eligibility or clinical interpretation is inferred from the model inputs.

## Outcomes and unchanged measurements

Reuse the verified three-item composite: nervous + (6−cope) + (6−control), each stored item 1–5,
requiring all three. Retain all 3,706 development rows; missing scores remain missing. Verify existing
item-value/score consistency before mapping classes. Primary: 3–4→0, 5–9→1, 10–15→2. Secondary binary
questions: 3–4 versus 5–15, and 3–9 versus 10–15. These are researcher-specified exploratory categories,
not PSS-10/PHQ-9 scores or validated clinical diagnoses. No rescaling, outcome imputation or cutoff search.

Use only current corrected feature bundles and the original five participant-grouped development folds.
Existing HR mean, sample SD, latest and slope; supported step additions; local time sine/cosine and survey
category. Existing historical reference/deviations are a separate 60-minute sensitivity. No IBI/HRV,
Garmin stress, floors, calorie/intensity, current answers, identifiers, raw sensor reconstruction or
quality counts/gaps/freshness enter physiological predictor sets. Timing/counter semantics remain provisional.
Morning asks since waking; Midday/Endday since last prompt, not necessarily the fixed lookback.

## Fixed experiment matrix and fair cohorts

Three-class task: all eight lookbacks 5,15,30,60,120,360,720,1440 minutes. At each lookback compare
context, context+HR, context+HR+steps on (i) joint complete cases of all five wearable summaries and
(ii) all valid-outcome/context responses with training-only median feature imputation plus missingness
indicators. Thus every feature/model comparison within a cohort has identical train/validation membership.
The **primary benchmark** is three-class, 60-minute, joint complete-case, unweighted fitting; weighted
and imputed results are explicitly labeled sensitivities rather than selected after seeing OOF results.
Also run a 60-minute HR-native complete-case sensitivity without requiring steps (context versus context+HR).

Both binary tasks use the same 60-minute joint complete and all-valid imputed cohorts only. Historical HR
reference sensitivity is three-class/60-minute: compare context, +HR, +HR+steps and +HR+steps+the existing
reference mean and mean/latest deviations, requiring the union of these features for complete cases;
imputed history uses all valid-outcome/context rows. These redundant deviation columns are deterministic
functions, not extra independent measurements. Do not regenerate or extend history to other windows.

Unweighted and balanced-class fitting are separate prespecified versions of each learned model. Dummy
always predicts the most frequent **unweighted training** class (ties resolve to the smaller class).
Weights n/(K*n_class) are recomputed within each actual training split, never globally or from validation.

At 60 minutes add a context+missingness-only diagnostic on the identical all-valid cohort, without
physiological values, to assess whether availability itself explains apparent gains. Missingness flags
are the only newly authorized recording-process predictors; source quality measures remain excluded.

Imputation is feature-level only; no raw reading or outcome is filled. Train-only median/scaler/encoder
are fitted separately in every inner/outer split. Features wholly missing in a training split are dropped
for that split, rather than assigned physiological zero. Preserve indicators for each permitted wearable
feature. Unknown validation survey categories use all-zero one-hot encoding. Report native-cohort results;
compare complete-case and imputed predictions on the same complete validation subset separately from
all-valid deployment coverage. This matched subset still compares different training populations.
Also report saved native predictions on the common complete-case intersection across lookbacks; label
this an evaluation-only restriction, not common-training refits. The all-valid imputed cohorts provide
identical training and validation memberships across lookbacks. No optimal lookback will be selected.

## Models, nested tuning and budget

Majority dummy; LogisticRegression with C in {0.01,0.1,1,10}, L2/lbfgs, max_iter=2000; XGBoost and LightGBM.
Each outer training set has three StratifiedGroupKFold splits (shuffle=True, fixed seed), grouped by
participant. Outer folds are unchanged. Fit/tune every comparator on the same cohort and inner split
membership. Inner mean macro F1 over the fixed task labels chooses settings; deterministic candidate-order
breaks ties. No outer-validation early stopping, threshold optimization or calibration.

For each boosting family, use 12 fixed candidate configurations (one simple anchor plus 11 deterministic
random samples of the documented parameter space), frozen in config before performance. This is a bounded
random-search equivalent to the requested optimizer, not exhaustive optimization. Search learning rate,
number of trees, tree complexity, child support, row/column sampling and L1/L2 regularization. Fixed CPU
histogram XGBoost and deterministic LightGBM, one thread per fit; no row-level early-stopping split.
The same candidate lists are used for every view/cohort/weight/target. No search expansion after results.

Models require every target class in each inner training/validation split and outer training set; if a
shared cohort/fold is infeasible, mark all its models skipped without redrawing folds. Report class counts
and fold feasibility; absence of a validation class is not evidence of perfect performance. Only report
pooled results as full OOF when all five outer folds are present.

## Evaluation, interpretation and outputs

Primary macro F1, with fixed task-label set and zero-division=0; per-class precision/recall/F1/support,
balanced accuracy, accuracy, weighted F1, macro one-versus-rest ROC-AUC where all classes are present,
and elevated-class recall/precision/PR-AUC. Predictions use argmax; binary uses the default 0.5 rule.
Report train versus OOF results, fold/category counts and errors, confusion matrices, prediction variation,
and an equal-participant-weight confusion-matrix metric to limit domination by frequent responders.

Paired participant-cluster bootstrap (2,000 fixed-seed draws) compares fixed OOF macro F1 with matching
dummy, logistic and same-family context; steps versus same-family HR; history versus same-family steps;
weighted versus unweighted and imputed versus complete on identical observed validation rows. These are
conditional intervals, not retuning uncertainty, multiplicity-adjusted tests or independent repeated
responses. Do not claim established benefit from point estimates alone. Weighted probabilities are not
assumed calibrated. ROC/PR-AUC describe ranking, not deployable risk estimates.

At the primary 60-minute benchmark report outer-validation permutation importance (five repeats) for
original feature groups using the fitted model; repeat for imputed results, distinguishing missingness
from values. It is descriptive, may be unstable/attenuated by correlated features, and never selects
features or feeds tuning. Permutation destroys dependence and is not a causal intervention.

Store all membership, targets, predictions, split/transform audits, fitted estimators and trial logs in
ignored versioned private bundles. Publish only aggregates, configuration, figures, report and abstract.
Notebook 12 will be fully executed as explicitly requested, with aggregate outputs only; prior notebooks
remain unchanged. Preserve raw sources and prior artifacts; no commit, push or final-test evaluation.

Official implementation references (accessed 2026-10-15):
[XGBoost classifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier),
[LightGBM classifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html),
[grouped stratification](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
[feature imputation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).
Pinned installed versions/source hashes, not a moving documentation page, determine executed behavior.
