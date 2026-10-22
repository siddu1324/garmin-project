# Decision table

| provider_answer | what_changes | next_action |
| --- | --- | --- |
| ts is acquisition/end timestamp, one ordered interval per event | End hypothesis becomes documented; start becomes a historical sensitivity | Check tiny event trace, latency and missing-beat encoding; revise contract/version before use |
| ts is acquisition/start timestamp | Start hypothesis becomes documented; end remains historical sensitivity | Check trace; current NeuroKit start adapter remains blocked unless separately validated |
| ts is callback/receipt/upload but original event clocks exist | Current membership/support describes exported clocks, not acquisition | Request re-export with event and availability clocks; version and re-audit, never shift to fit |
| Batch carries ordered intervals plus documented anchor/offsets | Original payload may support a proper decoder; CSV alone may have lost identity | Validate release-specific decoder and missing-beat handling before authorizing new normalized version |
| Only receipt clock and scalar values survive; order/offsets unknown | Exact physiological timing and NN adjacency cannot be established | Stop timestamp reconstruction and expanded HRV; retain explicitly provisional export-clock summaries |
| Identical rows share one immutable event ID (retry) | Evidence supports transport deduplication | Validate event-ID mapping against exact-row rule; preserve raw copies |
| Identical rows represent different batch indices/events | Current logical collapse conflates real interval records | Create focused regression and new logical-view version after evidence; preserve old bundles |
| Enhanced BBI confidence or other quality flags available | New provenance may distinguish withheld/low-confidence measurements | Recover original flags and define separate policy; confidence alone is not normal-beat proof |
| Quality flags absent or irretrievably dropped | Normal-beat and upstream rejection status remain unresolved | Do not label current IBI calculations physiological NN HRV; request validation/collection evidence |
| HR is cached/smoothed or activity timestamps are update clocks | Per-row weighting/freshness and counter-interval attribution need review | Document semantics; specify and version any revised summaries before performance evaluation |
| Full delivered-prompt logs exist | Midday/Endday recall boundaries may be observable | Check intended prompt event and prediction-time availability; do not infer from completions |
| No prompt logs or actual acquisition clock recoverable | Recall alignment cannot be made exact with existing data | Finish feasibility limitation; plan improved collection rather than more window search |
