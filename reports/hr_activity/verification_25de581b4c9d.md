# Notebook 04 verification

Completed the full window correction and provisional development-only HR/activity checkpoint.
No expanded real-data HRV, Garmin stress predictors, PSS targets, imputation, feature selection,
models, commit or push. All 32 final-test participants remain sealed.

## Corrected audit and dependent artifacts

- Reissued **5,321,816 windows**, all three unchanged policies and both hypotheses, with **177,888**
  EMA/policy/hypothesis/lookback accounting rows. All **192** aggregate coverage rows reconcile to
  individual diagnostic accounting; supported-time unions are bounded by eligible/requested unions.
- Independently verified **every window** using Python integer and datetime/timedelta arithmetic,
  exact 300-second duration, consecutive 60-second steps, each applicable lookback and EMA cutoff.
  Tests cover steps 546/547, 1092/1093 and the final 24-hour boundary at step 1435.
- **3,294,634** stored clock locations change. All approved 5/15/30/60/120-minute and six-hour
  results agree exactly with the obsolete version: **1,319,336** diagnostic rows, all coverage
  measures and all rejection counts. No eligibility threshold changed.
- Checked **60,407,178** contained interval memberships in eligible windows against independently
  derived boundaries; none extends beyond the EMA cutoff. Counts include overlapping views.
- Reissued the full dependent Notebook 02 index: **5,321,816** indexed slices match corrected
  windows and independent searches in the original IBI timestamps. Its **363 immediate member/run/case
  files are byte-identical** to their historical versions. Immediate rejection, variation, run,
  timestamp-precision and whole-stream timing aggregates are unchanged; long-lookback continuity
  changes are recorded in the supersession ledger.
- Default bundle validation, the existing indexed-window loader, new eligibility loader and report
  entrypoints reject known-invalid window versions. A renamed synthetic fixture is rejected before
  data access. Explicit historical checksum-only inspection remains separate from current use.
- Original raw/normalized/pilot bundles are unchanged. The old audit/reports/index remain preserved
  with an explicit replacement graph in `reports/window_supersession.json`; README identifies the
  current versions and marks old 12-/24-hour results obsolete. Notebook 03's corrected pilot is valid.

## HR/activity verification

- **3,706 feature rows**, preserving each original EMA key, category, cutoff/precision and frozen
  participant/fold values. All **125 development participants and five folds** are retained.
- **104 measurement columns**, with **88 explicitly allowed provisional predictors**. The 16
  calorie/intensity placeholders are missing and excluded. Quality and identifiers are excluded.
- Separate quality and lineage tables each contain **148,240 rows** (EMA × five streams × eight
  lookbacks). Every feature value and feature-specific availability reason reproduces from its
  original source window: **148,240 full window replays**.
- Independently recomputed **1,960** counter cases using a scalar walk over original timestamp
  groups, checking values, accepted-pair counts and duration. It cannot join across rejected groups,
  resets, long gaps, date/source/offset transitions or a left-boundary crossing.
- **17,364,780** logical HR/counter readings have reproducible row dispositions and source references.
  Supported counter time unions reproduce for both counters at all eight lookbacks; overlapping
  EMA views count each accepted original pair once. Missing observations never become zero activity.
- Actual EMA cutoffs were independently recomputed using Python datetime and each row's offset,
  with the precision-specific buffer. No clocks shifted to improve alignment.

The following counts identify EMAs encountering particular excluded counter pairs/readings.
Reasons overlap, may occur alongside supported pairs, and **must not be added**. All-EMA denominator
is 3,706. The JSON verification includes all eight lookbacks and survey categories.

| Sensor | Minutes | Reset/decrease | Gap >300 s | Day transition | Rejected endpoint | Left crossing | Conflict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| steps | 5 | 7 | 457 | 12 | 0 | 827 | 0 |
| steps | 1440 | 2944 | 3526 | 3245 | 256 | 3304 | 7 |
| floorsClimbed | 5 | 3 | 206 | 14 | 3 | 231 | 0 |
| floorsClimbed | 1440 | 1592 | 3288 | 2989 | 250 | 3067 | 0 |

## Tests, execution, privacy and reproducibility

- **89 tests passed**. They cover timestamps/offsets, both IBI hypotheses and boundary crossing,
  overflow/revocation, duplicates/conflicts, broken adjacency, resets, missing versus observed zero,
  irregular-time slopes, count/span requirements, participant/fold separation and unique-time coverage.
- Ruff passes on `src tests notebooks/04_hr_activity_features.ipynb`. Historical Notebook 02 retains
  its pre-existing unused `Markdown` import (F401); it is preserved unchanged. The source files and
  current notebook have no Ruff failures. `git diff --check` and offline lock verification pass.
- Notebook 04 executed all **5 code cells** without errors. Source outputs are empty; the executed
  copy is ignored at `outputs/04_hr_activity_features.7fdba29ad8d6f0fe.executed.ipynb`.
  Jupyter used authorized local loopback sockets outside the sandbox; no remote service was used.
- Known identifier/private-filename scanning passed for **141 public files** before this verification
  record. Executed notebook rendered text passed the same scan. Credential/local-home-path scans
  passed. All **1,903** checked private/output files are ignored. Figures display aggregate data only
  and were visually reviewed.
- Two numeric attrition-delta counts coincided with known identifier tokens. Review confirmed they
  are computed counts. The narrow privacy exception validates the exact count schema, integer types,
  nonnegative old/current counts and delta arithmetic before masking **only** those numeric count
  fields for token scanning. Strings, keys, other fields and malformed schemas remain checked;
  regression tests reject malformed/string-valued exceptions. No identifiers were published.
- Raw/reference tree hashes, sizes and mtimes match the original baseline; dependency files and
  frozen assignments are unchanged. **82 pre-existing source/document/notebook files** match the
  start-of-task hashes. Eight existing files contain the documented loader/configuration-resolution,
  independent-verification, privacy and README edits. Original synthetic adapters remain untouched.
- Deterministic reruns returned the same corrected audit, dependent diagnostics and feature versions
  with every artifact hash unchanged. No earlier validated bundle was overwritten. Git index is empty.

## Exact outputs

- `private/hr_activity/features/76f5885b3955d5b0/features.parquet` — 3,706 rows.
- `private/hr_activity/features/76f5885b3955d5b0/quality.parquet` — 148,240 rows.
- `private/hr_activity/features/76f5885b3955d5b0/window_lineage.parquet` — 148,240 rows.
- `reading_dispositions/` and `counter_pairs/` beneath the same bundle preserve original-row and
  adjacency provenance, including exclusions. The complete predictor list is in `feature_dictionary.json`.
- `private/preprocessing/eligibility/974e98729579967b/` — corrected all-policy audit.
- `private/hrv_input_validation/diagnostics/55eb9ba094e893b7/` — corrected downstream input diagnostics.
- [Current report and figures](7fdba29ad8d6f0fe/report.md),
  [feature dictionary](../hr_activity_feature_dictionary.md),
  [supersession record](../window_supersession.json), and
  [provider-question draft](../provider_questions_draft.md).

## Remaining limitations

HR values describe observed readings, not every moment in a lookback. Steps/floor sums are
supported additions, not complete-window activity. Acquisition/receipt, batching, invisible resets,
device quality and exporter counter semantics remain unresolved. Calorie/intensity features are
therefore missing. Neither lookback nor validity rules were selected to maximize retention.
The provider list is drafted only and was not sent. Resolve those semantics before any newly
authorized outcome linkage or modeling; HR availability does not validate expanded HRV.
