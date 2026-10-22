# Small development IBI variability pilot — selection fixed before calculation

The user explicitly authorized this limited real-data checkpoint after Notebook 02. This protocol
adds a separate guarded entrypoint; it does not change the historical input specification,
synthetic-only adapters, Screen policy or earlier checkpoint artifacts.

## Frozen selection

1. Use only the 125 frozen development participants and existing five development folds. Read
   EMA identity/clock metadata, not survey answers or outcomes. Search existing Screen-eligible
   five-minute windows, stepped backward by one minute, through the eight approved lookbacks
   up to 24 hours. Keep all 3,706 development EMAs in feasibility denominators.
2. Hydrate the unchanged input contract separately for interval end and interval start. A candidate
   must be one original uninterrupted run with **timestamp span ≥240 seconds AND union of
   acceptable interval support ≥240 seconds**, inside its Screen-eligible window. No relaxed rules.
3. Identify an exact repeated view by `(participant, hypothesis, first logical IBI row, last logical
   IBI row)`. Further group qualifying views that share at least one logical IBI record into
   transitive overlap components within participant/hypothesis. This conservative deduplication
   prevents slightly different window clips of the same recording being treated as independent.
   Adjacent ranges without a shared record are not automatically combined.
4. For each component choose one existing candidate by earliest first timestamp, then earliest
   last timestamp, then earliest EMA cutoff, window end, EMA key, private catalog key and row bounds.
   **Never merge intervals or extend the selected run across windows.** Component union bounds are
   only a deduplication key, never a new calculation input.
5. Choose the earliest representative per participant, then the earliest ten representatives
   globally under the same ordering, separately for each hypothesis. Retain fewer if necessary.
   At most one run per participant per hypothesis also prevents within-person temporal overlap
   in a calculation sample. Chronological sampling is a small convenience sample, not representative
   cohort estimation. Outcomes, feature values and package acceptance do not enter selection.
6. Publish the ignored, versioned selection and its hash **before** any real feature calculation.
   A separate calculation command verifies that manifest and requires the explicit pilot flag.

## Calculations and safeguards

Label every result **“IBI variability—normal-beat status unverified.”** A ≥240-second run is
not complete five-minute HRV. Preserve original absolute timestamps, millisecond IBI values,
physical/source references, quality flags, rejected window members and original valid adjacency.
The calculation input is one selected run only; do not concatenate, interpolate, impute, clean,
reconstruct a waveform, shift clocks, or manufacture pairs after removal.

Calculate only the five approved outputs: mean IBI; median IBI; sample SDNN (N−1); RMSSD over
original valid adjacent pairs; and pNN50 as 100 × count(|difference| >50 ms)/valid-pair count.
Undefined values remain missing. Reuse independently validated arithmetic and original adjacency
validation. The new real adapter uses the previously approved HeartPy lower-level time-domain
interface, with its sample-SD and percent conversions. Its unavoidable ancillary time-domain
internals are not retained as outputs. Median IBI has no HeartPy equivalent.

NeuroKit comparisons are permitted only for interval-end input with every original pair satisfying
its native 10 ms criterion and unchanged input sanitization. Record other calls as blocked without
invoking the calculation or repairing timestamps. No high-level HeartPy RR pipeline, spectral,
breathing, additional nonlinear or Garmin-stress predictor is invoked. Preserve the old synthetic
adapters unchanged and test equivalent synthetic cases against the new separate entrypoint.

## Feasibility, ages and privacy

Count every EMA/participant with ≥1 qualifying run independently of whether selected for the small
calculation sample. A run counts for a lookback only through an approved observation window fully
inside that lookback. Compute unique supported time by interval unions within participant, never
window count × five minutes. Count overlapping EMA views as correlated views, not independent runs.
Report exact-bound and overlap-component counts to make deduplication visible.

For each selected run preserve span, supported duration, interval/pair counts, and age from its
representative EMA cutoff to the last actual timestamp and to the last hypothesized support endpoint.
Also retain all exact repeated views and their ages in an ignored ledger. Public summaries report
ranges/medians across the calculation sample, never participant rows, IDs, absolute dates or exact
private source manifests. Summarize age across all qualifying views separately, explicitly noting
those views overlap. Report participant overlap and shared source records between hypotheses;
do not treat the two samples as independent cohorts.

Real run-level outputs, selection ledgers, source-member records and mappings stay ignored.
The raw archives, original bundles, frozen assignments/policies and folds remain unchanged.
No PSS targets, EMA-level feature aggregation, models, commit or push are part of this checkpoint.
Numerical agreement is not physiological validation; acquisition/receipt semantics, anchoring,
IBI units, missing-beat representation, sensor modality and normal-beat annotations remain unresolved.

## Cutoff-grid correction before feature calculation

The pilot invariant check discovered integer overflow in the historical audit generator:
`60 * step` used a signed 16-bit array, overflowing from step 547 (9 h 7 min).
Some stored windows consequently lay after the EMA cutoff; those windows must not enter this pilot.
This was found before any real feature calculation. The generating source now uses int64.
Previous immutable bundles and reports are retained as historical artifacts; their 12/24-hour
counts are superseded for pilot feasibility. Immediate through six-hour windows are unaffected.

The pilot creates an ignored corrected Screen window index in its selection bundle: derive
every end as `cutoff - 60 * int64(step)`, retain unchanged Screen results for matching windows,
and evaluate the exact same frozen Screen policy on corrected windows. No sensor timestamps,
EMA cutoffs, policies, buffers, selection ordering or thresholds change. The complete corrected
census and selected-run manifest are frozen before features. Full multi-policy reissue of the
old eligibility report is deferred; do not reuse its long-lookback counts as valid evidence.
