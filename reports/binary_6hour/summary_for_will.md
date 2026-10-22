# Binary PSS: six-hour logistic experiment for Will

**Six hours did not establish better discrimination than one hour or context.**

Binary A: full six-hour AUROC 0.508 (0.459–0.557), AP 0.726, balanced accuracy 0.498; versus 60 minutes: ΔAUROC +0.002 (95% interval -0.065 to +0.070), no clear positive difference; versus context: ΔAUROC +0.029 (95% interval -0.043 to +0.101), no clear positive difference.

Binary B: full six-hour AUROC 0.510 (0.472–0.546), AP 0.151, balanced accuracy 0.501; versus 60 minutes: ΔAUROC +0.014 (95% interval -0.052 to +0.079), no clear positive difference; versus context: ΔAUROC -0.003 (95% interval -0.046 to +0.038), no clear positive difference.

| task | model | auroc | fold_auroc_mean | ap | balanced_accuracy | sensitivity | specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Majority dummy | 0.500 | 0.500 | 0.720 | 0.500 | 1.000 | 0.000 |
| A | Context | 0.479 | 0.523 | 0.712 | 0.500 | 1.000 | 0.000 |
| A | Full single-window · 60m | 0.505 | 0.506 | 0.720 | 0.500 | 0.994 | 0.007 |
| A | Full single-window · 120m | 0.518 | 0.523 | 0.735 | 0.498 | 0.993 | 0.003 |
| A | Full single-window · 360m | 0.508 | 0.498 | 0.726 | 0.498 | 0.993 | 0.003 |
| B | Majority dummy | 0.500 | 0.500 | 0.140 | 0.500 | 0.000 | 1.000 |
| B | Context | 0.513 | 0.526 | 0.143 | 0.500 | 0.000 | 1.000 |
| B | Full single-window · 60m | 0.496 | 0.502 | 0.142 | 0.500 | 0.002 | 0.999 |
| B | Full single-window · 120m | 0.489 | 0.492 | 0.139 | 0.500 | 0.000 | 0.999 |
| B | Full single-window · 360m | 0.510 | 0.508 | 0.151 | 0.501 | 0.004 | 0.999 |

All main comparisons use identical 3,689 responses from 125 development participants; all 3,706 EMAs remain accounted for. Binary A is ≥5; Binary B is ≥10. Thresholds are exploratory, not clinical. Main six-hour models use 47 strictly within-window candidates plus context, excluding historical and adjacent-window columns. The complete 438-feature multiscale experiment is separate.

Pooled AUROC compares probabilities across different fold models and can reflect fold calibration differences. Mean fold AUROC is a stability diagnostic: Binary A context is 0.523 versus 0.498 for six-hour full features, despite their opposite pooled point ordering. It does not replace the specified pooled endpoint. Neither supports a reliable physiological gain.

**Which cutoff discriminates better?** At the prespecified unweighted full six-hour model, Binary A AUROC is 0.508 and Binary B is 0.510. These are different outcomes, not a clinical cutoff selection. Interpret uncertainty, class support and prevalence alongside the point difference; raw AP is not comparable across prevalence.

**Do expanded features add beyond context?** Binary A: ΔAUROC +0.029 (95% interval -0.043 to +0.101). Binary B: ΔAUROC -0.003 (95% interval -0.046 to +0.038). The full model includes recording-missingness flags. A context increment alone does not isolate physiological-value signal; interpret it with no-flags and missingness-only controls, not just F1 or accuracy.

**Does feature selection help?** Binary A: sparse versus full: ΔAUROC +0.000 (95% interval +0.000 to +0.000); correlation filter versus full: ΔAUROC +0.003 (95% interval -0.013 to +0.019); availability filter versus full: ΔAUROC +0.000 (95% interval -0.014 to +0.015). Binary B: sparse versus full: ΔAUROC -0.019 (95% interval -0.068 to +0.037); correlation filter versus full: ΔAUROC -0.014 (95% interval -0.063 to +0.041); availability filter versus full: ΔAUROC -0.010 (95% interval -0.061 to +0.047). These are multiple exploratory comparisons; do not select a winning feature set from the outer folds.

**Does weighting improve minority detection?** Binary A: balanced sensitivity 0.504 versus 0.993 unweighted; specificity 0.520 versus 0.003; precision 0.730 versus 0.719. ΔAUROC +0.001 (95% interval -0.041 to +0.041). Binary B: balanced sensitivity 0.402 versus 0.004 unweighted; specificity 0.635 versus 0.999; precision 0.152 versus 0.333. ΔAUROC +0.002 (95% interval -0.012 to +0.017). Sensitivity here is for class 1. Binary A's minority is class 0, whose recall is specificity; Binary B's minority is class 1. A changed operating point does not necessarily improve AUROC.

**Are recording gaps and missingness affecting performance?** Binary A: missingness-only AUROC 0.447; quality-only 0.467; context 0.479. Removing flags versus full: ΔAUROC +0.007 (95% interval -0.001 to +0.017). Binary B: missingness-only AUROC 0.508; quality-only 0.530; context 0.513. Removing flags versus full: ΔAUROC -0.030 (95% interval -0.082 to +0.030). These controls assess associations with the recording process and handling choices; they cannot establish that gaps caused weak prediction. See per-class missingness and complete-case composition.

**What should change next?** Retain strict time-support allowlists, explicit recording indicators and matched-cohort evaluation. Resolve timestamp/batching, sensor quality and prompt/recall evidence. Prespecify confirmation of any promising effect; do not expand the search or open the final test automatically.

Five frozen participant folds and three grouped inner folds tune L2/L1/Elastic Net by AUROC. Preprocessing and feature filtering are training-only. Balanced training, sparse selection, correlation/availability filters, no flags, missingness-only and quality-only controls are separately evaluated. AP is prevalence-sensitive; fixed-threshold recall is not interchangeable with ranking performance.

Six hours adds accepted readings for 2,518/3,689 responses; 1,171 gain none. The median additional count is 1536. This does not establish continuous six-hour coverage. Complete-case sensitivity refits matched cohorts; native coverage and class-specific missingness are reported separately.

Interpret the paired intervals, not isolated point maxima or F1 gains. Intervals resample participants conditional on saved predictions and do not include retuning uncertainty or multiplicity correction. This follows earlier development exploration; no final-test evidence is used. Coefficients are training associations, not causal or validated feature importance.

Next: retain strict single-window feature definitions and separate recording-process controls. Resolve exporter timing/batching, sensor quality and actual prompt/recall boundaries before further automatic searches. A promising difference requires a separately specified confirmation. See the standalone HTML for all results and verification.md for checks.
