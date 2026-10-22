# Notebook 11 — what do the Garmin export clocks and IBI rows mean?

**The current files establish our export-to-analysis transformations, but do not establish the
sensor timestamp's event meaning or identify the collection API/SDK and version.** The strongest
next step is a short provider-assisted provenance trace, not another model search. Passing our
rules means conditional eligibility under a clock/interval hypothesis, not verified physiological
coverage or normal beats. No new real-data calculations or policy changes occur in this checkpoint.

## What has actually been tested

All predictive experiments concern the study's continuous three-item perceived-stress composite
and observed HR/activity summaries, with 125 development participants and five frozen participant
folds. There are 3,706 development responses, 3,689 complete scores. The 32 final-test participants
remain sealed. The limited IBI pilot is a separate numerical feasibility exercise: it has not been
linked to outcomes or tested as a prediction feature. This distinction also matters when reading
older study proposals or legacy analyses, which are not current evidence.

| Checkpoint | Question tested | Established result and limit |
| --- | --- | --- |
| 00–01, corrected in 04 | Source identity, clocks, raw preservation, provisional IBI eligibility | Verified crosswalk and raw lineage; clock semantics remain unknown. Corrected window engine replaces overflow-affected 12/24-hour audits. |
| 02–03 | Input continuity and five time-domain IBI calculations | Severe immediate loss; independent arithmetic agrees with permitted package calls. Normal-beat status unverified. |
| 04–05 | Observed HR/step summaries and fixed Ridge/dummy/context prediction across eight lookbacks | No consistent wearable advantage on matched native/common cohorts; sparse observed readings are not representative continuous coverage. |
| 06–07 | Denser/recent HR; median loss; recall and constant predictions | No consistent support-policy gain. Most median fits are constant; training-only alpha diagnostics do not establish generalization. Morning recall is since waking; Midday/Endday since last prompt. |
| 08–09 | Retrospective wake alignment, distinct information, source tracing, bounded nested Ridge/HGB | Many window inputs coincide; sampled features trace correctly; synthetic signal is recoverable. Nonconstant HGB predictions still do not establish wearable benefit. |
| 10 | Earlier unlabeled HR reference and deviations | Does not establish improvement beyond matched raw HR or reference-only models. |

The following are **within-row matched comparisons** of existing pooled OOF MAE, in composite
points. Do not compare errors across rows to infer model improvement: their populations differ.
Each fitted comparator used the same training/validation memberships within its experiment.

| Existing comparison | Responses / people | Matching results | Paired MAE difference, conditional 95% interval |
| --- | --- | --- | --- |
| 08 Morning wake versus 60 min, nominal wake | 227 / 86 | Median 2.1410; context 2.1858; wake+context HR 2.2506; fixed+context HR 2.2510 | Wake−fixed −0.00033 [−0.02754, 0.02630] |
| 09 60-min HR, all categories | 1,634 / 118 | Median 2.0710; HGB context 2.0741; HGB context+HR 2.0814 | HR−median +0.01038 [−0.01600, 0.03918]; HR−context +0.00732 [−0.01986, 0.03622] |
| 09 supported-step subset | 1,355 / 113 | Median 2.0613; HGB context+HR 2.0690; plus steps 2.0698 | Steps−same HR comparator +0.00078 [−0.00392, 0.00588] |
| 10 earlier-history subset | 1,575 / 116 | Median 2.0673; HGB raw-HR C 2.0646; reference-only D 2.0854; deviations E 2.0949 | E−C +0.03033 [−0.02532, 0.08401]; E−D +0.00954 [−0.01716, 0.03432] |

Sources: [08 report](../../morning_recall/b1fd2e61d20cdbe4/report.md),
[09 report](../../window_benchmark/e3e34a2ea014f38f/report.md),
[10 report](../../past_hr_reference/d021427887616868/report.md).
Intervals resample participants with fixed saved predictions. They do not account for all repeated
exploration, future collection conditions or model-refitting uncertainty. Neither nested tuning nor
an interval containing zero establishes equivalence. No optimal window/model is declared.

The **53.5% within-person score variation** is a retrospective decomposition of squared deviations
around full-record person means. It says where scores vary, not which information predicts those
changes. Unmeasured context, reporting variation and measurement noise can contribute. It is not
an ICC, reliability estimate, available wearable signal or a ceiling for HR prediction. On the
native 60-minute HR cohort the fraction is 52.3%; person-centered mean-HR/score correlation is
−0.032 [−0.094, 0.036]. Full-record means use later observations and never enter predictors.

## Evidence about the collection source

The bounded search found **zero application-language source candidates and zero application build
or SDK-lock candidates** in the current repository and supplied earlier project tree, excluding
data, outputs and dependency folders. Existing Python/notebook code begins with exported CSVs;
it is analysis code, not a callback handler. The private discovery ledger records exact roots and
paths. This is absence in searched locations, not a claim that no collector exists elsewhere.
Prior keyword inspection also reviewed analysis/documentation files and available PDF documents;
no collection-time SDK release or timestamp contract was established. No new sensor sample was
needed: the missing evidence is upstream of the rows already traced.

**Local study document:** `R61-R33_ResearchStrategy.pdf`, PDF page 5, section B2.2, describes a
current study wearing a Garmin Vivoactive smartwatch during waking hours. PDF page 7, B3.4, proposes
a STARRS integration with Vivoactive 5 and a Garmin SDK, referencing a collaboration letter. Its
future-tense plan does **not** identify which exact device generation, app build, SDK edition/version
or callback produced these CSVs. The locally inspected copy has SHA-256
`cad3443ed2dc340830b20563d36deeb685ec126a4249881e5b34ebbbf66462ff`.
The exact private path and page extracts are in the evidence bundle. The document's earlier HRV
availability/performance statements have not been reproduced by our current limited pilot; their
cohorts, transformations and measurement provenance cannot be assumed equivalent. No old binary
results or proposed features are imported as evidence for the current continuous benchmark.

**Relevant official evidence, accessed 2026-10-07:** Garmin's November 2023 Enhanced BBI paper
(PDF pages 3–5; printed pages 1–3) describes PPG-derived BBI, access through Standard SDK and Health
API, differences between legacy and Enhanced processing, per-beat confidence in Enhanced BBI, and
possible withheld measurements. These are reasons to ask about mode, firmware and quality metadata;
they do not identify our stream. [Garmin Enhanced BBI](https://www8.garmin.com/garminhealth/news/Garmin-Enhanced-BBI_Final.pdf).
Garmin's current Health API overview describes JSON delivered after device sync to Garmin Connect
and lists Enhanced BBI among its offerings. API access/upload timing therefore cannot simply be
substituted for the event time of a measurement inside a payload.
[Official Health API overview](https://developer.garmin.com/gc-developer-program/health-api/).
The SDK landing page was checked but did not yield a release-specific callback contract in the
accessible page text. [Official SDK entry point](https://developer.garmin.com/health-sdk/).

No generic API schema, FIT timestamp convention, sensor cadence or fractional digits establishes
our source product. Analysis package versions (NeuroKit2 0.2.13, HeartPy 1.2.7) are not collection
SDK versions. Even a high-confidence beat annotation, if recovered, would not by itself prove
normal-to-normal status. That is a separate physiological question.

## Trace from original export to analysis

The missing portion is **device → collection callback/payload → storage → CSV exporter**. The
verified portion is **CSV strings → preserved physical rows → logical view → conditional windows**.
The [evidence ledger](evidence_ledger.md) gives exact source files, line anchors and hashes.

| Stage or field | Verified locally | What it cannot establish |
| --- | --- | --- |
| Corrected input container | Three top-level archives include three EMA tables, six blank questionnaires and nested Garmin CSV exports; already inventoried | Original transport payload, API method or app version |
| Raw sensor schema | Three schema variants, union `ts,dte_tme,rsp_id,mbl_cod,data_type,val`; identity fields may be absent | Device/firmware, batch/event/index fields, separate acquisition/receipt/upload clocks or beat-quality annotations are not explicitly exported |
| `data_type`, `val` | Long-form sensor category and scalar numeric parsing; originals preserved; observed HR/IBI categories | One CSV scalar is not proof that one callback contained one interval; arrays may have been flattened upstream. Numeric magnitude does not verify units |
| `ts` | Parsed as Unix seconds directly into UTC; no extra timezone shift | Acquisition, callback, receipt, upload, interval-start/end anchoring |
| `dte_tme` | Parsed separately; inferred offset/residual produces diagnostic flags, not a clock correction | An independent acquisition clock or known origin; it may derive from the same clock |
| Raw preservation | Original strings/schema, source/archive hashes, row number, participant mapping method and fold retained | Row number identifies export order, not necessarily beat order |
| Logical view | Only exact full-raw-record/schema duplicates collapse; sorted by timestamp/source/row. Equal IBI values at different times remain | Sorting cannot recover acquisition order after batching/reordering or identify distinct equal beats at a shared timestamp |
| Derived `q_` flags | Numeric/clock/tie/conflict/jump flags and explicit normal-beat-unverified flag | Device confidence, motion, wear/contact quality, ectopy or upstream rejection/correction |

Exact code: `src/garmin_inventory/sources.py:62`;
`src/garmin_preprocessing/normalize.py:51,132,158,168,261`.
Notebook 09 independently traced 320 linked HR feature values through 80 views to 18,153 original
accepted HR rows from ten development sources, with no mismatch. Its synthetic positive control
also recovered an injected relationship. Those checks support numerical plumbing, not the missing
upstream measurement contract. We reused them without opening another participant sample.

**Duplicates and stale values:** transport retries and legitimate equal intervals flattened under
one timestamp can produce identical exported strings. Current logical deduplication assumes exact
copies are redundant; without event/batch identity that assumption is unverified. Physical copies
remain available for a later evidence-based, versioned revision. A synthetic regression test verifies
this reversibility and that equal values at distinct times survive. Conflicts are rejected, never
averaged. Repeated HR values may be stable physiology, quantization or cached/stale values; the
CSV cannot decide. A timestamped zero/absence is not a device-quality label. Missing/rejected beats
could be withheld or lost upstream; nothing locally identifies their count or positions.

## What existing timing diagnostics do and do not resolve

Reused development IBI evidence: 34,427,546 physical rows; most fractional clocks have six/seven
nonzero decimal places. There are 504,202 compressed positive-time pairs (timestamp separation
less than one quarter of both neighboring IBI durations), 3,190 ties and 33,766 gaps over five
seconds. High display precision rules out universal whole-second representation, not callback
jitter or batching. Compressed pairs are compatible with batching, buffering or exporter behavior;
they do not uniquely diagnose it. Monotonic export clocks do not prove original interval order.
[Unaffected stream diagnostics](../../hrv_input_validation/30a0bbe188d56b40/report.md#precision-batching-and-clocks).

Consecutive time differences fit current and preceding durations similarly under broad tolerances;
this does not choose end versus start anchoring. In immediate five-minute windows, 2,196/3,706 have
no recorded IBI; 1,510 have some. Screen passes only four end/five start windows. Short uninterrupted
runs, internal gaps and insufficient support dominate overlapping failures. Removing the provisional
>20% interval-change gate alone did not improve Screen retention. That flag is not proof of artifact.

The pilot additionally required BOTH timestamp span and supported duration ≥240 seconds, leaving
one immediate window per hypothesis. Earlier searches allowed ten deterministic runs per hypothesis
(cap one participant each); samples shared eight people. HeartPy's permitted low-level arithmetic
agreed for all 20; NeuroKit calls were blocked for all 20 under the input contract. Start anchoring
is unsupported by that adapter; end inputs failed its native pair conditions. This does not establish
that real intervals are unusable or normal. These are **IBI variability—normal-beat status unverified**,
not complete five-minute HRV, stress predictors or a representative sample.
[Limited pilot](../../ibi_pilot/78af22e7f0ead75b/report.md).

## Which rules depend on unverified measurement assumptions?

See [rule dependency table](rule_dependencies.md). All three frozen policies and both anchoring
hypotheses remain unchanged. Their comparisons of elapsed timestamps to IBI duration, edge gaps,
supported-time unions and contiguous spans assume timestamps locate the relevant intervals. Both
start/end hypotheses can be wrong if clocks instead denote batch receipt. Their disagreement is
not a bound on total timing uncertainty. A recorded gap is not automatically a physiological gap;
a burst is not automatically a burst of actual beats. Never shift clocks, cumulatively reconstruct
beat times or concatenate disconnected rows to make these checks pass.

Whole-interval containment and the pre-EMA buffer are implemented correctly on exported clocks;
they do not establish acquisition-time or live-availability containment. HR freshness, occupied
minutes, slopes and historical references are likewise conditional. Two HR readings or 80% occupied
bins do not prove representative continuous observation. Step differences remain provisional until
counter/event-interval semantics are confirmed; reset handling cannot detect an unobserved reset.

No new concrete implementation defect was identified in this review. The old int16 overflow was
already corrected and fully reissued in Notebook 04; the current loader rejects known-invalid
versions. [Supersession record](../../window_supersession.json) is authoritative. Notebook 02 is
cited only for unaffected immediate/stream diagnostics, and the pilot's historical statement that
full reissue remained outstanding is now obsolete. No old bundle is overwritten or silently repaired.

## Decision and next step

[Feature status](feature_status.md) separates computable numbers from physiological validation.
The [decision table](decision_table.md) specifies how each possible provider answer changes the
next action. The [short provider request](provider_request_draft.md) asks for the minimum source
chain evidence, not a new participant export. It is a local draft and has not been sent.

Request the collection build and release-specific SDK/API contract plus a de-identified, ordered
payload→callback→storage→CSV example covering a normal interval, a batch and a gap. Include original
clocks, indices and quality flags. If it confirms acquisition timestamps and preserved intervals,
verify a small controlled trace and revise the input specification before considering another HRV
pilot. If the CSV uses receipt times but original event times survive, plan a separately versioned
re-export and re-audit. If original clocks/order are irrecoverable, physiological beat-timing HRV
and recall-aligned timing claims remain blocked; retain only clearly labeled export-clock summaries.
No further model search is justified by this checkpoint.

The private evidence bundle contains discovery paths and document extracts only, with input hashes;
public outputs contain aggregate facts, schema/code references and synthetic examples. Tests, notebook
execution, privacy and input-preservation results are recorded in the adjacent verification report.
