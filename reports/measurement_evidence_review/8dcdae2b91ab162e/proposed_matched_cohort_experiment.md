# Conditional proposal — IBI variability versus observed HR

**Proposal only; not an approved/fitted model or an independent preregistration.** This follows the
previous development experiments. Measurement review and outcome-independent feasibility must finish
first. Do not infer that statistical power is adequate from the counts currently available.

1. Establish the actual clock/interval/unit/quality contract, preserve raw fields and source lineage,
   and version any supported parser/logical-view changes. Re-audit all three frozen policies and
   both hypotheses where still applicable at 5/15/30/60/120/360/720/1440 minutes. Keep the corrected
   five-minute window engine and one-minute steps. Do not choose a policy by predictive performance
   or retention. If source annotations require a new policy, document its rationale before results.
2. Proposed initial comparison: the existing Screen policy plus the existing dual-240-second single-run
   restriction, **only if compatible with reviewed evidence**. No duration relaxation. Keep the
   immediate five-minute question separate from earlier-segment questions. The current immediate
   cohort is infeasible. A 60-minute retrieval horizon would connect to the previous fixed HR benchmark
   as a separately declared exploratory question, not as an optimal window. Other lookbacks remain
   feasibility tables until a scientific comparison is explicitly specified.
3. For each EMA/hypothesis, propose choosing the latest qualifying five-minute observation window,
   then its longest qualifying original run (supported duration, span, original row order as fixed
   tie-breakers), without consulting outcomes or feature values. Record age to cutoff and actual
   support. This is a new proposed deterministic selection rule to freeze before any expanded feature
   or performance run; it does not replace the old ten-run pilot selection or any earlier result.
4. HR comparator inputs must refer to the **same selected observation window**, using the existing
   mean/SD/latest/slope definitions, not unrelated 60-minute HR summaries. HR readings and IBI support
   can still cover different portions of that observation; expose this difference diagnostically.
   A stricter same-run HR comparison would be a separately specified sensitivity, not an automatic search.
   Do not aggregate overlapping IBI runs or treat each window as a separate EMA outcome.
5. Intersect valid existing composite scores, the five IBI measures and the required HR/context values.
   Preserve all development EMA rows in accounting. Refit every comparator on identical training
   responses and evaluate identical validation responses within the unchanged five participant folds.
   Suggested fixed set: training-mean/median dummies; context-only; context+HR; context+IBI; and
   context+HR+IBI, using standardized Ridge alpha=1 with training-only encoders/scalers. No tuning,
   imputation, participant IDs, quality/missingness predictors, Garmin stress or current answers in X.
6. Before fitting, report exact matched responses, participants, per-fold training/validation counts,
   unique source episodes and unique supported minutes, repeated-use counts, survey representation,
   missing-outcome/HR/IBI exclusions and age/support distributions. Report native cohorts separately;
   any cross-lookback comparison requires an explicit common cohort. Require every original fold to
   have nonempty training/validation cohorts and variation sufficient for the stated analysis; this
   is only numerical feasibility, not evidence of adequate precision. Skip infeasible fits without
   redrawing folds. Set precision/sample-size expectations separately from desired positive results.
7. If authorized and feasible, report pooled/fold MAE, RMSE, participant-averaged MAE, prediction
   variation, and paired participant-bootstrap differences on the same OOF responses. The key added
   value comparison is context+HR+IBI minus context+HR; context+IBI versus context+HR compares modalities.
   Bootstrap intervals are conditional on fixed exploratory predictions; repeated/overlapping sensor
   support is not independent evidence. No causal or validated real-time interpretation.

Current assessment: one immediate qualifying response/person per hypothesis cannot sustain the
five participant folds. Within 60 minutes, the earlier census offers only 20 responses/19 people
(end) or 12/12 (start), even before matching HR/outcomes; expect a very small pilot at best. At 24 hours,
334/51 or 262/40 suggests potential feasibility work, but says nothing about same-window HR overlap,
fold balance, independent episodes, physiological validity or power. No longer horizon is selected
because it retains more records. Earlier segments also change the temporal scientific question.

The first concrete implementation after sufficient evidence is a guarded, versioned development
eligibility rebuild plus run/member/quality ledgers and independent arithmetic checks—not a model
search. Normal-beat status remains explicitly unverified unless separately established.
