# Draft for Orson / data provider — not sent

We can locate actual survey starts to minute/second precision, but cannot yet establish what physiological
period the sensor rows represent or when prior answers became available. Please supply the following for
the **existing development participants only**, retaining stable keys and original records.

1. **EMA delivery and protocol:** the implemented definition of “since the last prompt,” including whether
   that refers to the current prompt or the preceding notification. Export complete scheduled/delivered,
   suppressed/missed/dismissed/unanswered logs, stable prompt IDs and their linkage to response/instance IDs.
   Include coverage intervals and explanations of logging gaps; logs of completed responses alone are insufficient.
2. **EMA clocks and versions:** actual start, every answer save/version, submission and server-receipt clocks,
   original timezone/offset, precision/truncation/rounding, clock source/synchronization, and editing semantics.
   Receipt must identify the exact answer version. Is a prior answer editable after submission?
3. **Waking:** any independently recorded waking event known before Morning response, its verified local date,
   event time, receipt time and collection method. If only `SLEEP_WAKETIME` exists, confirm that; we will retain
   the retrospective label and will not manufacture prospective waking events.
4. **Garmin lineage:** actual API/SDK name/version, collector/exporter commit/build, device model/firmware,
   original payload examples and the code mapping them to `ts`, `dte_tme`, `data_type`, `val`. Distinguish
   acquisition, SDK callback, phone receipt and server upload; document batch ordering, stale/cache values,
   clock precision/drift, retries, deduplication and export sorting. Generic Garmin API documentation cannot
   identify this export's meaning.
5. **Quality and activity:** original beat/measurement quality, wear/motion/signal flags and rejection policy;
   step counter units, reset/rollover/timezone rules, cumulative versus interval totals, predecessor linkage
   and whether unchanged values are emitted. Are missing counter rows equivalent to no report, not zero steps?
6. **IBI clarification for a later checkpoint:** individual interval versus batch/array, ordering, start/end
   anchor, missed/rejected beats, and normal-beat/quality status. No new HRV calculation depends on an assumed answer.

Please retain original payload hashes and export source-row mappings. The proposed typed interface is
[source_contract.md](source_contract.md); a separately reviewed hash registry is required before import.
No outcome-based timestamp shifting or threshold selection will be used.

| Provider answer | What changes | What stays blocked |
|---|---|---|
| Acquisition time and pre-cutoff receipt are verified | Stored readings can be evaluated as prospective measurements | Recall windows still need prompt/wake boundaries |
| `ts` is callback/receipt/upload with unknown delay | Existing results remain recorded-time exploratory | Prospective containment and recall-aligned prediction |
| Complete delivered-prompt logs plus recall rule supplied | Midday/Endday boundaries can be reconstructed | Any response lacking sensor availability/quality evidence |
| Only completed-survey schedule supplied | No change to recall gate | Last delivered prompt cannot be inferred |
| Independent waking and availability supplied | Morning prospective boundary can be tested | Current-response waking alone remains retrospective |
| Exact submitted versions and receipts supplied | Past-only history bridge can activate with reviewed chronology | Future edits/current answers cannot enter earlier predictions |
| Counter units/reset/adjacency confirmed | Supported increments inside aligned windows can enter matched activity comparator | Missing/partial/rejected intervals remain missing |
| Beat timing and quality unresolved | HR/activity audit remains useful | General HRV expansion and normal-beat claims |

Next action: review one small de-identified development payload/log slice and exporter function against this
contract before requesting a full export. This is a draft only; no messages have been sent.
