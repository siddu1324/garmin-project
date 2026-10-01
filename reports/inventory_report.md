# Corrected Garmin / EMA inventory

Recomputed from the local raw archives. The inventory computation performs no final linkage, PSS score calculation, HRV, imputation, feature selection or modeling, and never uploads participant data. Only structural accounting uses the whole cohort. All value, clock, recording-pattern and pre-EMA history summaries below use the frozen development participants. Final-test outcomes and exploratory sensor summaries remain untouched.

## Sources and cohort

The local inputs contain **4,680 EMA responses from 157 participants**, and **154 Garmin CSV tables with 72,165,655 physical rows**. The Garmin tables map to 154 participants; 3 EMA participants have no Garmin source. There are three top-level archives: three corrected EMA CSVs, six questionnaire PDFs, and 154 nested Garmin ZIPs. Members are read in memory without extraction.

| category | rows | participants | fields |
| --- | --- | --- | --- |
| Endday | 1183 | 156 | 137 |
| Morning | 1177 | 157 | 120 |
| Midday | 2320 | 157 | 115 |

Local duplicate-export files: 0; logical rows: 72,165,655. 154 local Garmin CSV content hashes match the corrected-v2 source catalog. 3 EMA CSV content hashes also match the corrected-v2 EMA source catalog. The reference catalog has 1 extra physical source and 416808 duplicate-export rows. That extra copy is not part of the current local input. Legacy All/INS exports, old features, models and prior results are not inputs to this inspection.

## Identity and split verification

Frozen assignments available: **True**. Development: **125**; final test: **32**. The original five development folds and assignment bytes are preserved in ignored private storage. No new random split is generated.

Assignment SHA-256: `93e8da449a56c911c2326492ef678bf71d609f88c935f22918fecdb50a283853`.

EMA participant/instance duplicate excess: 1; participant/instance/survey-category duplicate excess: 0. Across Garmin and EMA, 1 respondent key maps to multiple participant keys. Current/reference crosswalk pairs: 181/181; current-only/reference-only pairs: 0/0.

| method | rows |
| --- | --- |
| content_rsp_id | 8026518 |
| content_mbl_cod | 63801014 |
| reference_corroborated_filename | 338123 |

Use exact `mbl_cod` where present and an unambiguous, content-verified `rsp_id` crosswalk. Respondent keys alone are not universally unique. Missing-key sources require exact known filename tokens corroborated against the reference normalized provenance by CSV content hash and participant. This checks consistency with the prior verified mapping; it is not independent identity evidence inside a keyless raw CSV. Do not generalize filename-only mapping to new exports. Preserve source hash and row number in future preprocessing.

## Questionnaire coding

All three corrected tables use `PSS_NERVOUS`, `PSS_COPE`, and `PSS_CONTROL`. The questionnaire PDFs map exported codes 1–5 to displayed labels 0–4. The agreed score is `nervous + (6 - cope) + (6 - control)`, range 3–15; all three items must be integer codes in 1–5. Missing is not zero. A recoded 0–4 version would differ by a constant of three; do not mix score conventions. This is an adapted three-item measure, not the standard full PSS instrument. No response scores or outcome distributions were calculated.

| category | development_rows | complete_valid_items | rows_missing_any_item | nonmissing_invalid_cells |
| --- | --- | --- | --- | --- |
| Endday | 934 | 932 | 2 | 0 |
| Morning | 934 | 931 | 3 | 0 |
| Midday | 1838 | 1826 | 12 | 0 |

## Clocks — development only

All 3,706 development EMA rows remain in history denominators. Unparsed actual/scheduled clocks: 0/0. Scheduled local time minus the row offset agrees with `instance_id` for 3,706 rows. Offsets are minutes local-minus-UTC: [-480, -420, -360, -300]. Unix timestamps receive no additional timezone shift.

Actual-start precision: {'minute': 1868, 'second': 1838}. The median start delay is 420 seconds; 945 delays exceed 30 minutes and 46 exceed two hours. Scheduled time is not actual start. No dedicated submission/end clock exists in these CSV schemas. Survey-completed flags and location timestamps do not establish submission time.

The prior corrected-v2 implementation (`window_quality.py:boundary_epoch`) subtracts **60 seconds for minute precision and zero seconds for second precision**. This policy is preserved provisionally. Intervals are left-inclusive and right-exclusive: `[cutoff - lookback, cutoff)`. Rounding versus truncation remains undocumented.

Garmin development rows: 57,552,038; invalid epochs/text clocks: 0/0. 294 clock pairs differ by more than 1.1 seconds after allowing an inferred minute offset; 66 imply offsets absent from EMA. These are overlapping flags, not independent exclusion counts. Naive Garmin text is predominantly local time. Offset inference is a diagnostic, not a clock repair or proof of acquisition timing.

Development Garmin UTC date span: 2023-01-31–2026-01-14; development EMA span: 2023-01-31–2026-01-14. No study-date exclusions are silently applied. `ts` has fractional-second precision, `dte_tme` has mixed second/fractional precision. The exports do not expose a separate upload/acquisition clock; physiological measurement versus buffered receipt time remains unresolved.

## Sensor values and recording patterns — development only

| sensor | participants | rows | missing_or_nonnumeric_values | zero_values | min | max | outside_provisional_range | duplicate_timestamp_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hr | 123 | 15618530 | 0 | 4468 | 0.000 | 203.000 | 4468 | 270 |
| ibi | 123 | 34427546 | 0 | 0 | 250.000 | 2,463.000 | 9 | 3662 |
| stress | 123 | 4287943 | 0 | 846026 | 0.000 | 99.000 | 0 | 69 |
| pulseOx | 123 | 1471476 | 0 | 11992 | 0.000 | 100.000 | 11992 | 15 |
| steps | 123 | 1063305 | 0 | 1630 | 0.000 | 18,942.000 | 0 | 14 |
| calories | 123 | 650790 | 0 | 654 | 0.000 | 5,501.000 | 0 | 33 |
| floorsClimbed | 122 | 15176 | 0 | 8533 | 0.000 | 123.000 | 0 | 1 |
| intensityMinutes | 119 | 17272 | 0 | 9562 | 0.000 | 1,026.000 | 0 | 2 |

Range flags are diagnostic sensitivities only: HR 30–240 bpm, IBI 250–2000 ms, PulseOx 70–100%, stress 0–100, counters ≥0. Units are plausible interpretations of the schema and values, not verified exporter specifications. HR/PulseOx zeros are unusable as physiological measurements, but remain preserved. Stress zero may be a valid resting score or an exporter sentinel; no positive-only filtering is applied.

| sensor | positive_distinct_interval_p05_seconds | positive_distinct_interval_median_seconds | positive_distinct_interval_p95_seconds | gaps_above_60s | gaps_above_3600s | run_gap_threshold_seconds | continuity_run_median_seconds | continuity_run_p95_seconds | longest_continuity_run_seconds | runs_at_least_5min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hr | 0.838 | 1.050 | 3.150 | 7130 | 2675 | 2 | 2.866 | 15.961 | 181.005 | 0 |
| ibi | 0.416 | 0.630 | 0.909 | 5202 | 2219 | 2 | 43.694 | 845.871 | 7,921.866 | 21995 |
| stress | 1.008 | 3.990 | 12.172 | 50790 | 2593 | 60 | 55.962 | 1,926.742 | 31,977.901 | 12622 |
| pulseOx | 0.840 | 1.889 | 27.307 | 47995 | 2875 | 60 | 10.198 | 338.952 | 29,745.892 | 2790 |
| steps | 0.825 | 1.049 | 50.179 | 48834 | 3283 | 60 | 8.986 | 181.005 | 5,775.276 | 931 |
| calories | 59.008 | 60.065 | 60.484 | 459760 | 1965 | 60 | 0.000 | 59.990 | 598.289 | 66 |
| floorsClimbed | 20.013 | 651.840 | 30,138.202 | 12465 | 3857 | 60 | 0.000 | 45.150 | 824.807 | 9 |
| intensityMinutes | 28.898 | 305.349 | 28,826.606 | 13998 | 3824 | 60 | 0.000 | 59.429 | 371.090 | 4 |

Cadence quantiles pool positive distinct-timestamp differences within participant/sensor, rounded to milliseconds; no intervals cross participants. Continuous runs split at gaps >2 seconds for HR/IBI and >60 seconds for other streams. These descriptive thresholds are descriptive diagnostics, not validated eligibility rules. Singleton runs have duration zero. Calorie intervals slightly above 60 seconds (observed median 60.065 seconds) can split runs under the strict >60-second rule even at an approximately minute cadence. The HR result of no runs lasting five minutes under the 2-second rule does not establish absence of usable HR data; HR usability requires a separately justified, sensor-specific policy. No observations are removed by these run diagnostics. Even a long run does not establish clean NN intervals; a 24-hour first-to-last span does not imply 24 hours of continuous data.

| sensor | within_day_pairs | within_day_decreases | date_transition_pairs | date_transition_decreases |
| --- | --- | --- | --- | --- |
| steps | 1062251 | 33 | 931 | 851 |
| calories | 649734 | 39 | 933 | 884 |
| floorsClimbed | 14196 | 2 | 858 | 457 |
| intensityMinutes | 16265 | 2 | 888 | 275 |

Counter comparisons use adjacent sorted records and exported local dates. Decreases at date changes support cumulative daily counters; within-day decreases may be resets or inconsistencies. Tied timestamps remain a caveat. Do not sum raw counter readings. Exact-row duplicates, timestamp/value duplicates, conflicting timestamps, export-order reversals, more gap thresholds and an IBI clock-consistency diagnostic are in `inventory.json`.

## Available preceding history — development only

All development EMA responses are the denominator, including participants without Garmin. Counts use recorded timestamps without range cleaning, interpolation or imputation. “Any” requires one observation. The stronger column requires every internal and boundary gap ≤60 seconds; it is a diagnostic of recording density, not complete or usable coverage. IBI additionally has a ≤2-second version in the JSON. Missing history is never a zero value.

| sensor | lookback_minutes | denominator_development_ema | responses_with_any_observation | responses_with_all_edge_and_internal_gaps_le_60s |
| --- | --- | --- | --- | --- |
| hr | 5 | 3706 | 1520 | 1312 |
| hr | 15 | 3706 | 1598 | 1208 |
| hr | 30 | 3706 | 1663 | 1084 |
| hr | 60 | 3706 | 1797 | 955 |
| hr | 120 | 3706 | 1979 | 748 |
| hr | 360 | 3706 | 2790 | 227 |
| hr | 720 | 3706 | 3219 | 33 |
| hr | 1440 | 3706 | 3539 | 1 |
| ibi | 5 | 3706 | 1510 | 1314 |
| ibi | 15 | 3706 | 1570 | 1204 |
| ibi | 30 | 3706 | 1632 | 1083 |
| ibi | 60 | 3706 | 1749 | 953 |
| ibi | 120 | 3706 | 1920 | 740 |
| ibi | 360 | 3706 | 2729 | 217 |
| ibi | 720 | 3706 | 3124 | 30 |
| ibi | 1440 | 3706 | 3498 | 1 |
| stress | 5 | 3706 | 1536 | 854 |
| stress | 15 | 3706 | 1664 | 503 |
| stress | 30 | 3706 | 1757 | 268 |
| stress | 60 | 3706 | 1898 | 117 |
| stress | 120 | 3706 | 2069 | 20 |
| stress | 360 | 3706 | 2893 | 0 |
| stress | 720 | 3706 | 3241 | 0 |
| stress | 1440 | 3706 | 3539 | 0 |
| pulseOx | 5 | 3706 | 835 | 109 |
| pulseOx | 15 | 3706 | 1162 | 34 |
| pulseOx | 30 | 3706 | 1360 | 14 |
| pulseOx | 60 | 3706 | 1570 | 4 |
| pulseOx | 120 | 3706 | 1816 | 0 |
| pulseOx | 360 | 3706 | 2706 | 0 |
| pulseOx | 720 | 3706 | 3133 | 0 |
| pulseOx | 1440 | 3706 | 3527 | 0 |
| steps | 5 | 3706 | 827 | 25 |
| steps | 15 | 3706 | 1184 | 3 |
| steps | 30 | 3706 | 1422 | 1 |
| steps | 60 | 3706 | 1650 | 0 |
| steps | 120 | 3706 | 1887 | 0 |
| steps | 360 | 3706 | 2753 | 0 |
| steps | 720 | 3706 | 3191 | 0 |
| steps | 1440 | 3706 | 3540 | 0 |
| calories | 5 | 3706 | 1675 | 1 |
| calories | 15 | 3706 | 1742 | 0 |
| calories | 30 | 3706 | 1801 | 0 |
| calories | 60 | 3706 | 1910 | 0 |
| calories | 120 | 3706 | 2078 | 0 |
| calories | 360 | 3706 | 2919 | 0 |
| calories | 720 | 3706 | 3269 | 0 |
| calories | 1440 | 3706 | 3544 | 0 |
| floorsClimbed | 5 | 3706 | 234 | 0 |
| floorsClimbed | 15 | 3706 | 443 | 0 |
| floorsClimbed | 30 | 3706 | 654 | 0 |
| floorsClimbed | 60 | 3706 | 953 | 0 |
| floorsClimbed | 120 | 3706 | 1357 | 0 |
| floorsClimbed | 360 | 3706 | 2383 | 0 |
| floorsClimbed | 720 | 3706 | 2856 | 0 |
| floorsClimbed | 1440 | 3706 | 3317 | 0 |
| intensityMinutes | 5 | 3706 | 254 | 0 |
| intensityMinutes | 15 | 3706 | 466 | 0 |
| intensityMinutes | 30 | 3706 | 668 | 0 |
| intensityMinutes | 60 | 3706 | 970 | 0 |
| intensityMinutes | 120 | 3706 | 1351 | 0 |
| intensityMinutes | 360 | 3706 | 2459 | 0 |
| intensityMinutes | 720 | 3706 | 2955 | 0 |
| intensityMinutes | 1440 | 3706 | 3394 | 0 |

## Integrity and limits

Source hashes and modification times were rechecked for 3 local input files and 7,867 reference-project files (excluding environments, Git internals and caches). All matched. Archive CRC checking is performed by complete ZIP reads. The private manifest retains identifying paths; public reports do not. No participant-level inspection tables are written; the original frozen assignment file is copied unchanged into ignored private storage.

Open issues: acquisition versus upload timing, the origin of clock anomalies, actual-start rounding, exporter-specific units/sentinels, keyless-source identity provenance, counter reset semantics, and validated IBI artifact/continuity eligibility. Some can be characterized further in development data; exporter/protocol semantics require external documentation. See `feature_pipeline_plan.md` for the next concrete step and the package review.
