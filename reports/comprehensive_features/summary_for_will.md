# Comprehensive physiological features — summary for Will

The full supported feature set shows a positive conditional macro-F1 difference beyond context. Its three-class macro AUROC is 0.499 and balanced accuracy is 0.338; neither matched increment over context excludes zero. Improved macro F1 alone does not establish useful physiological discrimination.

Full versus context: macro F1 +0.091 (95% interval +0.071 to +0.110); AUROC +0.027 (95% interval -0.006 to +0.062); balanced accuracy +0.004 (95% interval -0.012 to +0.020). Full versus original HR 4+steps: macro F1 +0.091 (95% interval +0.072 to +0.110). Full-model high-class recall is 10.9%, precision 13.0%. High-class prevalence is 515/3,689 (14.0%). Full-model log loss is 1.774 versus 0.951 for context, and accuracy is 47.0% versus 58.0%. The expanded model predicts minority classes more often, but probability quality worsens.

| view | macro_f1 | roc_auc | balanced_accuracy | elevated_precision | elevated_recall |
| --- | --- | --- | --- | --- | --- |
| Majority baseline | 0.245 | 0.500 | 0.333 | 0.000 | 0.000 |
| Context only | 0.245 | 0.472 | 0.333 | 0.000 | 0.000 |
| Original HR 4 + steps · 60m | 0.245 | 0.483 | 0.333 | 0.000 | 0.000 |
| Expanded HR · 60m | 0.251 | 0.489 | 0.333 | 0.000 | 0.000 |
| Expanded HR · 8 windows | 0.323 | 0.508 | 0.342 | 0.168 | 0.058 |
| HR + activity · 60m | 0.264 | 0.489 | 0.334 | 0.167 | 0.008 |
| Full supported · 8 windows | 0.335 | 0.499 | 0.338 | 0.130 | 0.109 |
| Full + correlation filter | 0.319 | 0.496 | 0.326 | 0.120 | 0.080 |
| Full · L1 selection | 0.315 | 0.492 | 0.334 | 0.173 | 0.056 |
| Full · Elastic Net | 0.313 | 0.493 | 0.330 | 0.153 | 0.056 |
| Inner-selected feature group | 0.335 | 0.501 | 0.338 | 0.134 | 0.105 |

## All three tasks: descriptive point maxima

| task | view | macro_f1 | roc_auc | balanced_accuracy | elevated_precision | elevated_recall |
| --- | --- | --- | --- | --- | --- | --- |
| three_class | full | 0.335 | 0.499 | 0.338 | 0.130 | 0.109 |
| at_least_5 | full | 0.502 | 0.525 | 0.509 | 0.725 | 0.839 |
| at_least_10 | full | 0.501 | 0.477 | 0.505 | 0.155 | 0.085 |

## Feature counts

| Family | Columns |
| --- | --- |
| floors | 56 |
| history | 48 |
| hr | 240 |
| interaction | 24 |
| multiscale | 14 |
| pulseox | 240 |
| steps | 56 |

Each lookback has 85 candidates except 24 hours (83): adjacent-window contrasts are omitted there. The supported full model offers 438 physiological columns before training-only removal. Its three-class folds use 880–881 transformed inputs including encoded context and missingness flags; without flags, 442–443. The original five physiological features use 15 transformed inputs with context and flags. Correlation filtering retains 765–782 transformed inputs; L1 retains 788–815 nonzero coefficients. These larger numbers are model columns, not additional physiological measurements.

All supported physiological modalities at 60 minutes are HR and activity; that requested comparator is the HR + activity 60-minute row, not a duplicate experiment. PulseOx remains a separate sensitivity.


These maxima are exploratory, not selected or validated winners. Read paired_differences.json for uncertainty.

Original workbook:54 family rows, with 33 windowed IBI measure variants and 1,518 conditional IBI summaries; no exact overall scalar total is specified. Prior supported wearable columns 91; prior modeled distinct wearable columns 43. Now 678 candidates (438 supported-summary branch,240 provisional PulseOx), all 3,706 development rows retained;3,689 scored/context-valid rows modeled. Only one response is complete across all supported columns.

No new real HRV, final-test evaluation, raw changes, commit, push or provider message. Missingness controls are distinct from physiology. Existing timing/recall and source-quality uncertainties limit interpretation.

Next: resolve measurement provenance and recall boundaries; do not automatically expand model complexity.
