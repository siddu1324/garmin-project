# Professor summary — retrospective Morning comparison

**Result:** reported wake-to-cutoff HR does not show a useful advantage over the specified fixed
windows. This is an exploratory development experiment using current-survey wake reports
retrospectively; it is not a deployable prediction model.

The primary wake-versus-60-minute comparison uses the same 227 responses from 86 participants,
with identical training and validation memberships in the original five participant folds. OOF
MAE is 2.2506 for context+wake HR and 2.2510 for context+60m HR. Paired difference: −0.0003 composite
points (conditional 95% participant-bootstrap interval −0.0275 to +0.0263). Matching median dummy
and context-only MAE are lower by point estimate: 2.1410 and 2.1858. No interval establishes a gain.

Secondary 5- and 120-minute comparisons also fail to establish a wake advantage. Later-minute wake
sensitivity leaves comparison membership unchanged and has primary difference −0.0093
(−0.0495 to +0.0270). The common cohort is the same 191 responses / 74 participants as the
5-minute pairwise cohort; these are not independent replications.

Only 24.3% of all 934 Morning responses enter the primary comparison. All exclusions remain
accounted for, including three missing scores and invalid wake boundaries. Nominal wake has
253 responses with any HR, but only 237 with all four required features. Among primary included
responses, median wake-window minute occupancy is 48.1% and median largest edge-inclusive gap
39.4 minutes. A fresh final reading does not establish full recording coverage. Quality, wake
clock/duration and current answers never enter predictors.

**Recommendation:** do not search for an optimal window or tune models from these results.
Prioritize provider clarification of acquisition versus receipt/batching, start-event definitions,
wake date/timezone reporting and device-quality metadata. Any revised measurement protocol needs
a separately specified evaluation. The observed differences are too small/uncertain to support
window selection; they do not rule out physiological signal under better measurement.

All 32 final-test people remain sealed. No HRV, stress predictor, imputation, nonlinear model,
commit or push. Questions are not sent. See [report](report.md), [full matched errors](pooled_metrics.md)
and [conditional paired intervals](paired_intervals.json).
