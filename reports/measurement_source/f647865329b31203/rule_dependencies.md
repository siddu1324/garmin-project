# Rule dependencies

| rule | unverified_assumption | consequence | current_action |
| --- | --- | --- | --- |
| Clock window membership and precision buffer | ts locates relevant event; local start/offset is the intended boundary | Receipt clocks can misdate observations; live availability needs its own receipt log | Preserve cutoff; label all timing provisional |
| IBI plausibility and interval-change flags | val is milliseconds; ordered neighbors have appropriate biological meaning | Wrong units, missed beats or upstream processing can invalidate artifact interpretation | Keep ranges and >20% flag provisional; no threshold changes |
| Pair consistency and gap/expected ratio | Adjacent exported clocks correspond to adjacent interval endpoints/starts | Callback jitter/batches can sever valid physiological runs or mimic close observations | Keep both frozen hypotheses; no reconstructed timestamps |
| Edge gaps, supported-time union, contiguous span and duration/elapsed ratio | ts anchors each interval under the chosen start/end hypothesis | Both hypotheses fail if ts is receipt; union is then geometry, not acquired duration | Report conditional eligibility; Screen can admit only a 120-second run |
| Exact duplicates and same-time conflicts | Identical full rows are redundant events; timestamps distinguish interval identity | Batch flattening could merge distinct equal intervals or create artificial conflicts | Preserve physical copies and lineage; require original batch/event/index identity |
| HR latest/slope/occupancy and earlier reference | Clocks locate observations; values are new readings rather than cached repeats | Freshness and occupied bins may describe delivery cadence; dense batches affect means | Observed-reading summaries only; quality separate; no continuous-coverage claim |
| Supported activity deltas | Counter increments occur within the two recorded endpoint clocks; reset/update behavior understood | Delayed values, hidden resets and boundary-spanning measurement intervals misattribute additions | Keep supported-only deltas provisional; unsupported remains missing |
| Package-adapter gating | RRI units, actual adjacent-pair identity and permitted anchor are correct | Numerical agreement cannot establish normal beats or sensor validity | Keep current restricted adapters; no additional real HRV |
