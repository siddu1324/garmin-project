# Exploratory classification of a three-item perceived-stress composite from Garmin–EMA data

**Objective.** Assess whether recorded pre-EMA heart rate and activity distinguish exploratory perceived-stress
categories beyond survey and time-of-day context after earlier regression benchmarks showed weak performance.

**Methods.** We analyzed 3,706 responses from 125 development participants; 3,689 had valid three-item
composites (range 3–15), classified as low (3–4; n=1,033), middle (5–9; n=2,141), or high (10–15; n=515).
Five frozen participant-held-out folds contained three-fold grouped inner tuning for logistic regression,
XGBoost and LightGBM. Majority-class baselines and context/wearable models used matched cohorts.
The primary benchmark used unweighted 60-minute complete cases (1,355 responses; 113 participants).
Prespecified sensitivities examined eight lookbacks, class weighting, feature-level imputation, earlier
HR references and two binary cutoffs. Macro F1 was primary; intervals used paired participant bootstrapping.
All 32 final-test participants remained sealed.

**Results.** Primary context-plus-HR-plus-steps XGBoost and LightGBM achieved macro F1 of 0.306 and
0.311, versus 0.249 for the majority baseline and 0.251 for logistic regression. Increments over matching
context-only models were 0.015 (95% conditional interval −0.010 to 0.037) and 0.025 (−0.003 to 0.051).
Balanced accuracy was 0.335/0.338; high-class recall was 1.8%/1.2%. Class weighting increased high
recall to 17.2%/14.7%, with precision 10.0%/9.3%. Some longer-window and imputed sensitivities showed
small positive context increments, but gains were inconsistent across weighting choices and did not
establish reliable high-score identification. Earlier HR references added no stable boosting benefit.

**Conclusion.** Classification exposed severe high-score detection failure despite macro-F1 gains over
a majority baseline. The primary experiment did not establish incremental wearable value beyond context.
Measurement timing, recall alignment, recording selection, limited independent participants and repeated
exploratory comparisons constrain interpretation. These categories are not clinical diagnoses; synchronized
measurement evidence and targeted independent validation are needed before any deployment claim.
