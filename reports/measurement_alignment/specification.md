# Notebook 19 — measurement recovery and alignment specification

Frozen before new diagnostics and any new performance. This follows inspection of Notebooks 00–18;
it is exploratory, not independent preregistration. Preserve all prior results, 3,706 development
responses, the original 125 development participants/five folds and 32 sealed final-test participants.

Current path: no new provider evidence supplied. Execute timing, source-field, cadence, counter and
coverage diagnostics. No new real predictive fits until the relevant gates pass. Absence of evidence
is represented as BLOCKED, never as poor model performance or a substituted long recall window.

## Fixed diagnostics

Actual minus scheduled start is an observed start delay relative to a schedule, not delivery delay
or answer completion duration. Describe sign, missingness, precision, quantiles and fractions exceeding
5/15/60 minutes by survey. Report 6 four-hour local-start strata, all eight fixed lookbacks, participants
and survey categories. Availability by existing composite≥5/≥10 labels is descriptive; participant-cluster
uncertainty and survey/time adjustment separate composition from crude associations. No target change.

Use cached development HR and counter ledgers and raw-preserving normalized fields, never repeat raw
normalization. Report timestamp repetitions/conflicts, original-order reversals, consecutive recorded
gaps, repeated values and counter resets. Fixed gap diagnostics2/60/300/3600 seconds are descriptions,
not new eligibility rules. Patterns can be consistent with multiple mechanisms and cannot identify
acquisition, batching or upload timing without the collector contract. Empty periods never mean inactivity.

Prespecified timing sensitivity: hypothetical Garmin event-clock displacement−60/0/+60 seconds relative
to stored time. Evaluate recording support only, for each fixed lookback; preserve stored clocks and values,
keep the original EMA cutoff, and do not compute performance or choose a preferred shift. Use only records
with stored timestamps strictly before cutoff in every scenario; reject shifted times that reach cutoff.
This is a bounded perturbation analysis, not a confidence interval for unknown latency or a repair.

## Future metadata contract and recall gates

Import typed development event JSONL plus a separately reviewed contract/approval registry, preserving
original payloads, source hashes and line numbers. Validate explicit UTC offsets, precision intervals,
identity, event/version keys, duplicate conflicts, clock order and source-contract hashes. Require complete
event-log coverage for the relevant prompt/wake horizon, including unanswered/suppressed events. The
contract must explicitly state what “last prompt” means, including current-prompt handling. No previous
completed-response substitute. Any unresolved order or precision overlap blocks the affected response.

Morning: independently recorded verified waking event on a verified waking-date scope → frozen cutoff.
Midday/Endday: last eligible verified delivered prompt → cutoff, according to the approved recall rule.
Only accepted HR with acquisition-time upper bound<cutoff and receipt-time upper bound<cutoff can enter
prospective features; callback/upload clocks stay separate. Boundary availability must also precede cutoff.
Conservative interval containment respects clock precision. Preserve original records and values; do not
average conflicts, interpolate, bridge rejected counter endpoints or infer missing events.

## One conditional matched experiment

If metadata gates open, targets remain composite≥5 and≥10; nervousness≥3 is separately labeled secondary.
Require valid outcomes/context and observed four original60-minute HR features, four aligned HR features
and supported aligned activity rate for the common A–E cohort. Preserve excluded rows/reasons. A context;
B original HR-only60min; C aligned HR-only; D context+aligned HR; E D+supported activity. Add context+original
HR as the explicit matched original reference. Separate context+coverage and D+coverage controls.
No imputation, no quality predictors in the physiology blocks, no threshold search. Reuse L2 logistic
C∈{0.1,1}, three grouped inner folds and five frozen outer folds. All comparators use identical rows.
Report constants, pooled/fold AUROC, AP, balanced accuracy, sensitivity/specificity, Brier/calibration,
class support and paired participant-bootstrap intervals. No inference of physiological gain solely
from coverage-control improvement. Skip infeasible folds without resplitting or relaxing gates.

Previous EMA history is independently gated on exact answer versions and their verified submission/receipt
availability. Select the latest version available strictly before each cutoff; never retrospectively
apply a later edit. Use Notebook17 history/chronological population functions after this version-aware
bridge. Keep chronological personalization separate from unseen-participant prediction. No real activation
is authorized by a fabricated clock, current answer or metadata merely declaring itself verified.

Synthetic integration tests exercise success, failures, identity/precision ambiguity, versions, current
prompt versus prior prompt, delayed receipt, DURING/POST exclusion and injected-signal recovery through
the actual extractor and grouped model path. Synthetic scores never appear as real findings.
