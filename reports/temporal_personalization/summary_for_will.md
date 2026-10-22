# Temporal alignment and personalization — summary for Will

**Earlier wearable references did not establish a physiological prediction gain. EMA-history prediction remains untested because answer availability is not timestamped.**

All 3,706 development responses remain accounted for; 3,689 valid-score/context responses from 125 people form the main matched cohort. The five original participant folds and sealed final test are unchanged.

## What we learned

- **Labels:** pooled internal-consistency alpha 0.719; person-centered alpha 0.666. These are descriptive covariance measures, not validated reliability or proof of clinical meaning. Morning refers to since waking; Midday/Endday to the last prompt.
- **Temporal EMA structure:** 3,564 consecutive valid-response pairs have retrospective lag-one correlation 0.584; after full-record person centering it is 0.221. These are associations, not available-at-cutoff predictions. Centering never enters a model.
- **Online-history feasibility:** 3,064 later responses across 125 people follow five valid starts, but **zero** have verified historical-answer availability. Earlier starts do not prove submission/receipt. Previous-EMA, persistence and personalized-label predictions are therefore blocked, not scored as zero performance.
- **Raw physiology versus context:** ΔAUROC A -0.004 [-0.023, +0.015]; B -0.001 [-0.014, +0.012].
- **Personalized wearable representations versus raw:** A +0.011 [-0.012, +0.035]; B -0.008 [-0.029, +0.014]. Versus reference alone: A +0.012 [-0.008, +0.030]; B +0.003 [-0.022, +0.026].
- **Clock-matched/activity extension:** versus personal block A +0.006 [-0.009, +0.020]; B +0.002 [-0.006, +0.010]. No primary interval establishes a gain.
- **Secondary boosting:** context+personal pooled AUROC A 0.432, B 0.529; the matching context increments are A -0.011 [-0.045, +0.023], B +0.002 [-0.032, +0.036].
- **Recording constraints:** raw/reference-observed cohort 1,312 responses/112 people; strict 60-minute support 1,115/104. Sensor coverage, complete observed rows and all-row imputed comparisons are not interchangeable. Empty periods remain unobserved.
- **Pooling:** a pooled OOF AUROC mixes probabilities from different fitted models. Notebook 16 A context oversampling has within-fold ranking near chance despite pooled AUROC≈.542. We decomposed within-/cross-fold pairs without normalizing predictions using validation labels.

## What is not resolved

This does not show that physiology is fundamentally uninformative. It shows no established gain from these fixed representations under the recorded-clock pipeline. Its increment **beyond EMA history cannot yet be tested**. Associations after adjusting for survey/time/activity remain small with participant-bootstrap intervals spanning zero. Missingness controls and feature-level imputation do not establish clean separation of physiological and recording information.

No real personalized chronological benchmark or change-state predictor is claimed. The ≥2-point increase threshold is exploratory, not a validated meaningful-change threshold. Sleep/HRV/respiration/PulseOx are not introduced as validated features. Garmin acquisition/receipt, source quality and prompt semantics remain unresolved.

## Recommended next experiment

Obtain response-version submission/receipt clocks and complete prompt logs. Then run the specified matched previous-EMA versus previous-EMA+physiology comparison, followed by prequential personalization after five verified available answers, with permitted population training labels also available before fitting. No guessed completion delays, automatic new model search or final-test unsealing.

The history constructor and chronological baseline are implemented and synthetically tested. Raw files, previous checkpoints and frozen assignments are preserved; no messages, commit or push.
