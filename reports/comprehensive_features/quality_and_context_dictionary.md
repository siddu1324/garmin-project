# Metadata, context and separate quality measures

`development_features.parquet` has one row per development EMA. The physiological column dictionary is in `feature_dictionary.md`. Keys, actual clocks, raw answers, targets and folds are private; they are never predictor candidates. The continuous target requires all three valid items: nervous + (6 − cope) + (6 − control). Three-class values are 0 for 3–4, 1 for 5–9 and 2 for 10–15. Binary targets use ≥5 and ≥10. Missing targets remain missing.

Only sine/cosine of actual local start time and survey category enter the context block. Category encoding is fitted in each training split. Frozen participant and EMA keys are evaluation/linkage metadata. No provider timing uncertainty is resolved by these calculations.

`quality.parquet` is long format: EMA key, sensor/family and lookback. Inapplicable measures remain missing. Quality does not enter physiological models; a separately labeled quality-only comparator uses an explicit allowlist. Missingness indicators in imputed models describe absent engineered values, not invented raw observations.

| Fields | Definition and units |
| --- | --- |
| recorded_groups, accepted_readings, accepted_fraction | Logical timestamp groups before acceptance, accepted count, and accepted/count ratio. Exact/repeated duplicates follow the existing logical ledger; conflicting readings are rejected. |
| occupied_minutes, occupied_fraction | For current HR/PulseOx windows: window-left-aligned one-minute bins containing accepted observations, and count divided by window minutes. Earlier references instead use observed UTC-minute bins divided by 1,440; their two partial edge bins can make the proxy slightly exceed one. Neither measure is continuous coverage. |
| span_seconds | Last minus first accepted timestamp; does not subtract internal gaps. |
| latest_age_seconds, left_gap_seconds | Cutoff minus latest observation, and first observation minus left boundary, in seconds. |
| maximum_internal_gap_seconds | Largest accepted timestamp separation, retaining the fact that intervening rejected observations can break valid adjacency. |
| maximum_gap_seconds | Largest internal or edge gap, seconds. |
| valid_adjacent_pairs | Original accepted neighboring pairs meeting the provisional ≤60-second rule; never rebuilt after rejecting a record. |
| supported_pairs, supported_seconds | Counter pairs with defensible nonnegative increments and both endpoints inside the window; total interval duration in seconds. |
| reset_pairs, boundary_pairs | Diagnostic counts of counter decreases and intervals ending inside the window whose preceding endpoint crosses its left boundary. Pairs ending at/after the right boundary are excluded by the half-open slice, rather than counted in this left-crossing diagnostic. Their increments do not enter supported additions. |
| occupied_hours, history_status | Earlier-reference UTC-hour support and explicit availability/insufficient-history reason. |
| reference_start, reference_end | Private UTC boundaries for the disjoint earlier-recording history; not predictors. |
| reasons_json | Per-feature unavailability reasons: no recording, rejected observations, inadequate count/span/pairs, or constant values where a definition requires variation. |

The separate `zero_quality.parquet` counts zero-valued logical records. A zero fraction is not time at zero, confirmed nonwear or a verified sensor artifact. The original IBI artifact flags remain provisional and are not newly interpreted as beat-quality labels.

Predictor transforms remove training-all-missing and constant columns, then fit medians, scaling and optional correlation filtering on training data only. Validation values never determine these operations. Constant columns and selected feature counts are recorded per fold in `settings.json`; private estimator objects retain the exact transform. Public selection stability includes missingness flags explicitly named as such. No calibrated-probability claim is made: calibration plots, Brier scores and log loss are diagnostics of the original probabilities, without post-hoc recalibration.
