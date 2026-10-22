# Focused provider / PI request — draft only

No message has been sent. Notebook 18 found no new verified processing defect and no
established physiological prediction gain. Additional model searching cannot identify
what the exported clocks or missing activity reports mean.

| Needed evidence | Smallest useful deliverable | Affected development responses | What it changes |
|---|---|---:|---|
| Sensor event time | Collector app/build + exact Garmin API/SDK/version/device/firmware; timestamp-assignment and array-flattening code; one redacted original payload → exported row trace retaining event, callback, receipt and upload clocks | 3,706 | Determines whether recorded pre-cutoff readings were acquired before cutoff, batch order and freshness; any correction must produce a new version |
| Counter contract | Define steps units, cumulative/reset behavior, emission-on-change versus periodic reporting, unchanged/stale values, counter measurement intervals and rejected/quality flags | 3,706 potentially; 41,024 candidate activity bins in this audit | Only 5 supported inactive bins versus 591 active bins survive. Clarifies whether unchanged/no reports can identify inactivity; missing never becomes zero automatically |
| Complete prompt history | All delivered/suppressed/expired/unanswered prompts with stable keys, UTC event time and the protocol meaning of “last prompt” | 1,838 Midday + 934 Endday = 2,772 | Opens a verified since-last-prompt window; previous completed EMA cannot stand in for a prompt |
| Waking evidence | A separately observed waking event and its availability time, or explicit permission to remain retrospective; clarify date/offset and late-response interpretation | 934 Morning | Current-survey waking is retrospective only: 836 nominal candidates, 826 later-minute candidates; no verified prospective boundary |
| Exact answer availability | Per-answer-version item-save, submission, server-receipt and edit clocks tied to stable EMA keys, with precision/timezone definitions | 3,706 | Permits past-answer prediction only after those exact values were available; start/completion flag cannot substitute |
| Measurement quality / IBI | Wear/contact/quality annotations, original IBI payload ordering, start/end anchoring, missed/rejected beat encoding | All physiological interpretations; IBI remains blocked | Determines which values can be physiological inputs; normal-beat status remains unverified, no new HRV calculated |

The existing exported fields are `actual_start_local`, `timezone_offset`,
`scheduled_start_local`, `instance_id`, `SLEEP_WAKETIME`, and a completion/instruction
field. They do not supply the missing event clocks. We require provenance, not guessed
constant delays or timestamps shifted until performance improves.

Public entrypoint: `src/garmin_root_cause/alignment.py::align_verified_metadata` first
requires matching development participant/EMA keys, the frozen cutoff, a matching
development sensor ledger and approved source-contract/payload evidence hashes.
Its underlying `src/garmin_root_cause/data.py::recall_gate` requires verified sensor
acquisition, UTC conversion and boundary meaning; verified wake-event or delivered-prompt
source; boundary occurrence ≤ availability < cutoff; complete prompt log when applicable.
`aligned_features` then uses only accepted HR in [boundary, cutoff). Every real development
response currently fails the gate; the implementation is tested with synthetic metadata.

If acquisition clocks and prompt boundaries are verified, freeze a matched aligned-versus-60-minute
comparison before fitting. If timestamps are receipt-only, preserve them and request acquisition
clocks; do not reconstruct beats or compensate with a guessed delay. If activity is change-triggered,
obtain a validated periodic unchanged-state stream or a separate activity reference before repeating
the physical-consistency probe. If provider evidence remains unavailable, prioritize prospective
measurement improvements and preserve the current results as exploratory recorded-data analyses.

Local sources: [exact questionnaires and PDF/page citations](../recall_alignment/f518537c0a3a6f14/questionnaire_evidence.md),
[clock inventory](../recall_alignment/f518537c0a3a6f14/report.md),
[measurement evidence ledger](../measurement_source/f647865329b31203/evidence_ledger.md),
[Notebook 17 request](../temporal_personalization/data_quality_audit.md).
