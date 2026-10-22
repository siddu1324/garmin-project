# Professor briefing — resolve measurement provenance before further modeling

**We have not established that Garmin `ts` marks acquisition or that exported IBI rows preserve
individual beat timing/order. The exact collection API/SDK and version remain unidentified.**
This checkpoint reuses existing diagnostics; it fits no models and calculates no new real HRV.

The strongest established findings:

- HR/activity prediction has not shown a clear gain beyond matched dummy/context baselines.
  On 1,634 responses/118 people, HGB context+HR MAE is 2.0814 versus matching median 2.0710;
  difference +0.0104, participant-bootstrap interval [−0.0160, 0.0392]. Supported steps also do
  not establish benefit on their own matched 1,355-response cohort. Earlier-reference/deviation
  models do not improve on raw HR or reference-only models within the 1,575-response history cohort.
- The IBI work is separate: only 20 selected run-hypothesis rows received five time-domain
  calculations. Numerical package agreement is not physiological validation. Normal-beat status
  remains unverified; no IBI feature has been tested in the prediction models.
- 53.5% within-person score variation describes variation around retrospective person means.
  It does not say that HR measures those changes or that those means are available at prediction
  time. Centered HR associations were small, with intervals spanning zero.
- The numerical pipeline passes raw HR tracing and a synthetic injected-signal control. The
  int16 window defect was corrected previously. This audit found no new concrete implementation bug.

Local `R61-R33_ResearchStrategy.pdf` page 5 mentions a Vivoactive study; page 7 proposes a Vivoactive
5/SDK integration. It does not identify the implementation that produced these exports. No collector
source or application build lock was found in the two searched project roots. Six exported field
names do not reveal upstream arrays, clock assignment or source quality flags. Existing derived
`q_` flags are our diagnostics, not Garmin beat annotations.

One consequential uncertainty: identical IBI rows could be duplicate delivery or distinct equal
intervals flattened at one batch timestamp. All physical records survive, but logical deduplication
and every start/end support calculation remain conditional on upstream semantics. Batched receipt
could invalidate both anchoring hypotheses simultaneously. We will not repair timestamps or loosen
rules to increase retention.

**Next justified work:** obtain a minimal provenance packet: app/build and exact SDK/API method/version;
device/firmware and BBI mode; timestamp assignment and array-flattening/export code; a small
redacted payload-to-CSV trace retaining order, clocks, units and quality/missing-beat information.
Request complete prompt-event logs separately to resolve “since last prompt.” The attached
[provider draft](provider_request_draft.md) and [decision table](decision_table.md) make this actionable.
Nothing has been sent.

If acquisition clocks and interval identity are documented, validate a controlled trace before
another physiological pilot. If only receipt clocks survive, seek original event timestamps/re-export;
if unavailable, leave beat-timing HRV and precise recall alignment blocked. HR/step numbers remain
computable as provisional summaries of exported observations, not validated real-time predictors.
All 32 final-test participants, frozen folds, raw files and previous bundles remain untouched.
