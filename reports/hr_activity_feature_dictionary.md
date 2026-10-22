# Provisional HR/activity feature dictionary

The complete [machine-readable dictionary](hr_activity/7fdba29ad8d6f0fe/feature_dictionary.json)
lists 104 measurement columns and the exact 88-column predictor allowlist. This table explains
the column patterns. `{L}` is 5, 15, 30, 60, 120, 360, 720 or 1440 minutes before the actual EMA cutoff.
See the [prespecified rules](hr_activity_feature_specification.md) for source validity and exclusions.

| Column pattern | Meaning/units | Minimum valid observations and span | Predictor |
| --- | --- | --- | --- |
| `hr_mean_bpm__{L}m` | Arithmetic mean of observed HR readings, bpm | 2 distinct times, ≥1 s | yes |
| `hr_median_bpm__{L}m` | Median observed HR, bpm | 2 distinct times, ≥1 s | yes |
| `hr_sd_bpm__{L}m` | Sample SD, divisor N−1, bpm | 2 distinct times, ≥1 s | yes |
| `hr_minimum_bpm__{L}m` | Lowest observed HR, bpm | 2 distinct times, ≥1 s | yes |
| `hr_maximum_bpm__{L}m` | Highest observed HR, bpm | 2 distinct times, ≥1 s | yes |
| `hr_q25_bpm__{L}m` | 25th percentile, linear order-statistic estimator, bpm | 2 distinct times, ≥1 s | yes |
| `hr_q75_bpm__{L}m` | 75th percentile, same estimator, bpm | 2 distinct times, ≥1 s | yes |
| `hr_latest_bpm__{L}m` | Last valid observed HR before cutoff, bpm | 1 reading, ≥0 s | yes |
| `hr_slope_bpm_per_minute__{L}m` | OLS slope against actual elapsed minutes, bpm/minute | 3 distinct times, ≥60 s | yes |
| `steps_added_supported__{L}m` | Sum of supported nonnegative counter differences, steps | ≥1 intact eligible pair, 0<dt≤300 s | yes, provisional |
| `floorsClimbed_added_supported__{L}m` | Same for floor counter, floors | ≥1 intact eligible pair, 0<dt≤300 s | yes, provisional |
| `calories_added_supported__{L}m` | Missing placeholder; units/counter meaning unresolved | never emitted | no |
| `intensityMinutes_added_supported__{L}m` | Missing placeholder; weighting/reset meaning unresolved | never emitted | no |

Features remain missing when their own support rule fails. No recording and no supported counter
pair are not zero. Supported zero activity requires an actually observed eligible unchanged pair.
Means and slopes summarize recorded readings; they do not fill intervening times.

Metadata `n_ema_id`, `n_participant`, `n_partition`, `n_fold`, `n_survey_category`, `n_cutoff_epoch`,
`n_start_precision`, `n_invalid_clock` are preserved for provenance, not predictor columns.
No survey answers/PSS, Garmin stress, IBI or PulseOx columns enter this feature table.

Quality is a **separate** table keyed by `(n_ema_id, sensor, lookback_minutes)`. None is a predictor:

- `recording_status`, per-measure `status_*`: no recording, all rejected, insufficient observations,
  insufficient timestamp span, no supported pair, unresolved semantics, invalid cutoff or available.
- `recorded_rows`, `distinct_recorded_timestamps`, `valid_readings`, `rejected_rows`, tie/conflict and
  clock/value counts: observation accounting; overlapping flag counts are not additive attrition.
- `observed_span_seconds`, `latest_age_seconds`, left/right and maximum internal/edge gaps,
  `internal_gaps_gt_{60,300,3600}s`: diagnostic timing; no gap rule is silently applied to HR summaries.
- `occupied_minute_bins` / `possible_minute_bins`: recording density, not measured physiological time.
- `supported_pairs`, `supported_pair_seconds`, `supported_pair_fraction`: accepted counter support.
- `pairs_*`: no previous baseline, rejected endpoint, local-day/offset/source transition,
  nonpositive time, long gap, counter decrease/reset, or left-boundary crossing.

The lineage table records original participant partition, source-partition hash, exact time boundaries
and timestamp-group/source-row slices. Reading dispositions retain original logical/physical row
references and source hashes. Counter-pair ledgers retain rejected adjacency as well as accepted pairs.
All are ignored and participant-level; only aggregate tables appear in reports and Notebook 04.
