# Provider-event contract — version 19-v1

This is an implemented future interface, **not evidence that our original Garmin `ts` is acquisition time**.
No real metadata have been supplied or approved. Existing exports, timestamps, values, assignments and
frozen feature bundles remain unchanged. Public examples are synthetic; original filenames and line
provenance are in the ignored `exact_source_catalog.json` within this checkpoint's private run.

## Source evidence today

- EMA fields: `actual_start_local`, `scheduled_start_local`, `timezone_offset`, `instance_id`, `rsp_id`,
  `mbl_cod`, `MORNING_SURVEY_COMPLETED`, `SLEEP_WAKETIME`, `Location Timestamp`. Availability and precision
  are counted in `field_audit.json`. Exact raw source/archive/member mapping is private because filenames
  may identify participants. `n_source_id`, `n_source_sha256`, `n_source_row`, `n_source_columns` retain lineage.
- Garmin original fields: `ts`, `dte_tme`, `rsp_id`, `mbl_cod`, `data_type`, `val`. `ts` is parsed as Unix
  seconds unchanged, `dte_tme` is a separate textual clock. Normalization preserves both and flags clock
  discrepancies. No source-defined acquisition/receipt/batch/quality/wear metadata are established.
- Normalized source: `private/preprocessing/normalized/31e51c934bcdeffa/`;
  corrected HR/counter ledgers: `private/hr_activity/features/76f5885b3955d5b0/`;
  cached features/quality: `private/comprehensive_features/features/eecf1da995c39b27/`.
- Code: `garmin_preprocessing/normalize.py::parse_ema_clock,normalize_chunk,annotate_duplicates`,
  `garmin_hr_activity/features.py`, `garmin_comprehensive/features.py`, and the new
  `garmin_measurement_alignment/{metadata,alignment,provider,experiment}.py`.
- [Exact supplied questionnaire wording and PDF/page references](../recall_alignment/824551e40b8e301d/questionnaire_evidence.md):
  Morning blank form pp.2–3 Q9–11 / `PSS_*_1`; Midday p.2 Q5–7 / `PSS_*`;
  Endday p.2 Q5–7 / `PSS_*_4`. Morning wake Q7 is an answer in the current response.

## Files and review

Supply `contract.json` plus one or more event `.jsonl` files to a private inbox. A separate review JSON
must name `review_status: approved`, a reviewer, the exact approved `contract_sha256` list and approved
`event_file_sha256` list. Merely placing files in the inbox or calling a field “verified” does not open
any gate. A human must substantiate these definitions from collector/export code and original payloads;
a hash establishes file identity, not truth. No review registry is generated automatically.

Contract fields: `schema_version: 1`; `precision_convention: lower_bound_plus_resolution`;
`recall_prompt_rule: last_delivered_before_cutoff | exclude_current_prompt` (provider must resolve what
“last prompt” means, including whether the current delivered prompt counts); `clock_definitions`
with `acquired: sensor_acquisition_time`, `received: server_received_this_exact_event_version`;
`collector` with `api_sdk`, `api_sdk_version`, `collector_version`, `device_model`, `firmware`;
`hr_acceptance` with `accepted_quality: [accepted]`, `accepted_wear: [worn]`; `steps` with
`units: steps`, `mode: cumulative`, `reset_rule: explicit_epoch_and_negative_change_break`,
`maximum_pair_gap_seconds: 300`. This version deliberately rejects a different counter interpretation;
review and version code before accepting interval totals or undocumented reset behavior.

## Event schema

Every event: stable `event_id`, canonical development `participant_id`, `event_type`,
`payload_sha256`, `source_contract_sha256`, and `clocks`. An optional `ema_id` must map one-to-one to the
same participant. All EMA saves/submissions/starts require it. Prompt events require `prompt_id`.

Supported event types and required event clock:

| Event | Clock / additional fields |
|---|---|
| `prompt_scheduled` / `prompt_delivered` | `scheduled` / `delivered`; stable `prompt_id` |
| `prompt_suppressed`, `prompt_missed`, `prompt_dismissed`, `prompt_unanswered` | `event`; `prompt_id` |
| `survey_started` | `start`; optional verified `provider_prompt_id`, `verified_local_date` binding |
| `answer_saved`, `answer_submitted` | `saved` / `submitted`; integer `answer_version`, stored 1–5 `items` (missing keys/null retained; incomplete versions cannot supply a composite) (`PSS_NERVOUS`, `PSS_COPE`, `PSS_CONTROL`) |
| `wake_observed` | `wake`; independent verified local date, never the current-response wake answer |
| `sensor_measurement` | `acquired`; `sensor: hr | steps`, numeric `value`, `quality`, `wear`, `source_sha256`, positive original `source_row` |
| `coverage` | `coverage_start`, `coverage_end`; `scope: prompt | wake`, `complete: true` |

Every event also needs its exact-version `received` clock. Optional `callback` and `uploaded` remain
separate. Counters additionally need `counter_epoch` and `previous_event_id` (null only for the original
first measurement). This prevents creating artificial adjacency from an incomplete export.

Each clock is `{value, encoding, precision_seconds, utc_offset_minutes?}`. Encoding is `iso8601`
or `unix_seconds`. Naive ISO requires its explicit local-minus-UTC offset in **minutes**. Unix may not
receive another offset. Precision is an explicit truncation uncertainty width, not inferred acquisition
accuracy: possible time lies between lower and lower+resolution; zero is an explicitly exact time.
This version permits widths 0–60 s. If source timestamps are rounded, drifted, or have other uncertainty,
review a new contract rather than force them into this one. ISO values alone do not prove true precision.

## Validation and operational behavior

Unknown/final-test identities fail before interpreting their measurements. Wrong EMA ownership, unknown
keys, hash mismatch, conflicting event IDs/answer versions/export-row joins, invalid types/ranges,
ambiguous clock order and unsupported contracts fail. Exact retries remain in physical JSONL and are
collapsed only in the logical view. Input raw JSON, hashes and line numbers remain in each import.
Current implementation imports JSONL in memory; large provider exports should be partitioned or streamed
in a later version before production-scale ingestion. It never reprocesses the original raw sensor archive.

Complete prompt/wake log coverage is required to establish the last relevant event. A later-delivered
proof of log completeness may document historical completeness, but the boundary event itself must have
been received before cutoff. A delayed latest event blocks alignment; it does not trigger fallback to an
older event. Provider start bindings must agree with the preserved actual start; original cutoffs never move.

Use boundary upper bound as the conservative left endpoint. Only acquisition and receipt upper bounds
strictly before cutoff enter features. All original values/clocks survive. Conflicts are rejected;
repeated equal HR at distinct timestamps survive. HR 30–240 bpm, quality accepted and worn, with the
existing mean/SD/slope support rules. Unresolved overlapping acquisition-time intervals block ordered HR.
Steps require both exact endpoints inside the interval, original predecessor, same source/reset epoch,
0 < duration ≤300 s and nonnegative increment. Rejected endpoints, resets and missing predecessors break
adjacency; no partial increments, zero-fill or interpolation. Activity-rate support is counted once.

Features and quality stay separate. Physiological columns: four `aligned_hr__*__60m` names are compatibility
aliases for **variable recall intervals**, not claims of 60-minute duration; the explicit aligned boundary
and duration govern their meaning. `aligned_steps_rate` is steps per supported minute. Coverage columns
are controls/diagnostics only. There is no HRV/stress predictor or reconstruction of beats.

Use `provider.build(root, inbox, approval, development, cfg, execute_fixed_experiment=False)` to import,
validate and save versioned aligned records. An explicit `True` executes only the frozen A–E logistic
comparison on its common eligible rows, plus coverage controls. Exact-version history features are saved
separately, with no automatic history-model activation. Notebook 17 chronology still needs a reviewed
fit-time/evaluation schedule when real version clocks become available; unknown availability remains blocked.

All outputs are private immutable bundles. `source_contract.md` and the synthetic tests describe supported
interfaces, not permission to replace earlier scientific results.
