# Measurement evidence review — awaiting the supplied source

**No new measurement evidence has been located for review. No Notebook 11 uncertainty can therefore
be marked resolved by this request.** This is an evidence-availability checkpoint, not a finding
that an unseen document is insufficient. The user has been asked for its local path, name or link.
No revised eligibility, expanded real IBI calculation or model is run.

Git status was inspected; pre-existing changes are preserved. No applicable AGENTS.md was found in
the repository or its ancestors. The project, project parent, earlier documentation tree and local
attachment metadata were checked. The only project file newer than the Notebook 11 verification
JSON was its companion verification Markdown. The already-reviewed study document remains available;
a proposed SDK integration cannot be reclassified as proof of the exported stream's provenance.
A file with preserved timestamps or evidence elsewhere could still exist: an exact location is needed.

## Uncertainty resolution against Notebook 11

| Question | Resolved by new evidence? | Precise remaining evidence needed |
| --- | --- | --- |
| Actual Garmin interface/version, device/firmware and BBI mode | Not assessed: new source unavailable | Collection-build dependency/API method and device provenance tied to these exports |
| Acquisition versus callback/receipt/upload clock | Not assessed; remains unresolved | Code/contract assigning `ts` and `dte_tme`, plus event/receipt timing trace |
| Units, interval start/end anchoring, timestamp precision | Not assessed; remains unresolved | Release-specific units/anchor/rounding definition and payload-to-row example |
| Original beat order and arrays/batching | Not assessed; remains unresolved | Ordered original payload, per-event/batch/index identity and flattening code |
| Exact-row duplicates versus distinct equal intervals | Not assessed; remains unresolved | Event identity across retries/batches; exact scalar equality is insufficient |
| Rejected/missed/withheld beats and original adjacency | Not assessed; remains unresolved | Missing/rejection encoding and upstream filtering/repair behavior |
| Source quality and normal-beat status | Not assessed; remains unresolved | Original confidence/motion/contact/rhythm annotations and their meaning; normal-beat status remains unverified |
| Live availability and survey recall boundaries | Not assessed; remains unresolved | Separate receipt/availability clocks and full prompt-event logs |

Notebook 11's verified facts are unchanged: original strings/provenance survive; analysis clock
conversion, corrected windows and frozen folds are traceable; derived `q_` flags are not source beat
annotations; existing package comparisons establish arithmetic only. Both anchoring hypotheses remain
conditional and can both be wrong under batch-receipt clocks.

**No scientific pipeline change is supported at present.** This version adds documentation and a
repackaging of previously published aggregate coverage only. Existing normalizers, adapters, policies,
input hashes and bundles are unchanged. It does not enable a new calculation entrypoint or repurpose
the historical pilot configuration, which references superseded dependencies. A future implementation
must use the current corrected index and create new manifests with evidence hashes and explicit
supersession links; it must never overwrite or silently relabel earlier results.

## Deliverables supported now

- [Feature dictionary](feature_dictionary.md): precise definitions for the five requested measures,
  with physiological interpretation and quality/provenance separated.
- [Existing coverage comparison](coverage_comparison.md): all three frozen policies, both hypotheses
  and all eight lookbacks; separately, the earlier dual-240-second pilot feasibility census.
  These are **before-evidence** values, not newly rebuilt or validated coverage. No after-evidence
  count or coverage change can be calculated until the source is reviewed.
- [Proposed matched-cohort experiment](proposed_matched_cohort_experiment.md): a conditional design;
  no fitting, outcome-based threshold selection or tuning is launched.

The immediate dual-240-second census has only one response/participant per hypothesis and cannot
support the five-fold comparison. The 24-hour search has 334 responses/51 participants (end) and
262/40 (start) before outcome/HR intersection and fold checks. These are upper bounds, not proven
matched sample sizes or a reason to choose 24 hours. Available aggregate tables cannot establish
whether the final matched cohort spans every fold or has enough independent recordings. That remains
undetermined; a small arithmetic pilot is not evidence of adequate predictive-study power.

All 32 final-test participants remain sealed. No new sensor or outcome values are opened, no features
are recomputed, and no raw data, prior artifact or participant assignment is changed. No provider
message, commit or push occurs. The next dependent action is to inspect the exact newly supplied source.
