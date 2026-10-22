# Comprehensive continuous PSS regression — summary for Will

The lowest point MAE among the prespecified full-feature models is Elastic Net at 2.245, compared with 2.209 for the matching median dummy. Its paired MAE difference from the median is +0.036 (95% interval -0.009 to +0.080); versus its matching context model, +0.006 (+0.001 to +0.010). RMSE is 2.700, R² -0.004, and Spearman correlation -0.095. For actual scores 10–15, MAE is 4.754 and mean prediction−actual is -4.754.

These comparisons do not establish a reliable improvement over both the matching median and context baselines. Point minima are descriptive, not a new selected model or final-test result. Repeated development exploration remains a limitation.

Elastic Net selects zero coefficients in every full-feature outer fold and reproduces the matching training-mean dummy. Its small pooled prediction variation comes only from different training-fold means; it has no within-fold physiological discrimination. The median dummy predicts 6 throughout and has undefined Spearman correlation, not zero correlation.

| Model | MAE | 95% low | 95% high | RMSE | R² | Spearman | Participant MAE | Prediction SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Median dummy | 2.209 | 2.073 | 2.345 | 2.726 | -0.023 | nan | 2.222 | 0.000 |
| Mean dummy | 2.245 | 2.116 | 2.368 | 2.700 | -0.004 | -0.095 | 2.258 | 0.056 |
| Ridge | 2.278 | 2.156 | 2.400 | 2.774 | -0.059 | 0.023 | 2.291 | 0.761 |
| Lasso | 2.247 | 2.120 | 2.374 | 2.704 | -0.006 | -0.072 | 2.260 | 0.220 |
| Elastic Net | 2.245 | 2.116 | 2.368 | 2.700 | -0.004 | -0.095 | 2.258 | 0.056 |
| Huber | 2.314 | 2.188 | 2.440 | 2.873 | -0.136 | 0.019 | 2.326 | 1.072 |
| Histogram boosting | 2.255 | 2.119 | 2.393 | 2.760 | -0.048 | -0.053 | 2.269 | 0.423 |
| XGBoost | 2.246 | 2.114 | 2.373 | 2.718 | -0.017 | 0.001 | 2.259 | 0.441 |
| LightGBM | 2.260 | 2.131 | 2.390 | 2.742 | -0.035 | 0.012 | 2.274 | 0.568 |

| question | answer |
| --- | --- |
| Do expanded features improve continuous prediction? | No full-feature model has a lower point MAE than the median dummy (2.209). Even the lowest learned main comparison, Original HR4 + steps · 60m / Histogram boosting, has MAE 2.211. |
| Do physiological values add beyond context? | 0 of seven full-versus-context paired intervals supports lower MAE. The matched full models often worsen error; the quality/missingness controls do not establish a physiological signal. |
| Do selection or longer windows solve this? | The lowest selected-feature point MAE is 2.244, still above the median baseline. Single-window and multiwindow results do not justify an optimal scale. PulseOx stays provisional and does not rescue the full benchmark. |
| Do nonlinear models outperform Ridge? | Full XGBoost minus full Ridge MAE is -0.032 (95% interval -0.063 to -0.001). That modest conditional advantage does not beat the matching median/context baseline. The compared losses also differ; this is not isolated evidence for nonlinearity. |
| Can the models identify higher continuous scores? | For the 515 responses scoring 10–15, actual mean PSS is 11.16. Full-model mean predictions are about 6.1–6.5, with MAE about 4.7–5.1. Higher-score underprediction remains severe. |
| Is the problem implementation, sparsity or weak physiology? | Synthetic controls, exact legacy replays and estimator replays passed. This rules out the tested computational errors, not all possible errors. Sparse/stale readings, recall mismatch and source semantics remain plausible limitations; this benchmark cannot identify their causal contribution. |

All main models use the same 3,689 valid-score/context responses from 125 participants. All 3,706 EMA rows are retained in accounting. These are newly tuned regressions, not conversions of classification probabilities.

438 supported summaries are reused; provisional PulseOx stays separate. Eight main blocks, seven model families and prespecified secondary checks produce 98 comparisons across five frozen outer folds. Every transform/selector and hyperparameter decision stays inside grouped training splits. MAE is the selection objective; boosting losses are explicitly compared.

The full tables report all feature groups, score-band errors, calibration of continuous predictions, prediction spread, convergence, missingness/quality controls and matched legacy comparisons. Point minima do not identify an optimal window, model or physiological mechanism. High-score residuals and weak correlation must not be hidden by pooled MAE.

Intervals condition on fixed development predictions and omit retuning uncertainty and multiple-comparison correction. Timing, recall alignment, source quality, sparse observations and participant heterogeneity remain unresolved. No new HRV or proprietary Garmin stress is included.

Next: resolve measurement timing/quality and prompt/recall boundaries, then specify a narrow confirmation of any supported matched effect. Do not automatically expand the search or open the final test. See verification.md and README.md for checks and exact output paths.
