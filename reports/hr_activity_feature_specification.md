# Prespecified provisional HR/activity definitions — Notebook 04

These definitions are recorded before feature calculation/retention comparison. They do not
change IBI eligibility policies and do not authorize expanded HRV, outcomes or models. They use
only the frozen 125-development cohort/five folds and keep all 3,706 EMA records, including
invalid cutoffs/no Garmin/no usable readings. The 32 final-test participants remain sealed.

## Sources, clocks and duplicates

Read only normalized identity/clock/survey-category metadata from EMA, never response items.
Garmin source fields are `data_type`, `val` and Unix-seconds `ts`, with `dte_tme` used only for
clock flags/local-day boundaries. `n_value` and `n_epoch_s` are parsed copies; physical rows retain
raw strings, content/archive hashes, source rows and verified mapping provenance. No re-normalization.
EMA UTC is actual local start minus its own local-minus-UTC offset. Minute-resolution starts
use a 60-second buffer; second-resolution starts use none. Never scheduled EMA time. Each
lookback is [cutoff−60L,cutoff), for L=5/15/30/60/120/360/720/1440 minutes. No clock shifting.
Acquisition versus receipt/batching remains unresolved; every timing claim is provisional.

| Source | Field/value interpretation | Existing development cadence (median positive dt) | Decision |
| --- | --- | --- | --- |
| hr | `val`, presumed bpm | 1.050 s; irregular, gaps exist | observed-reading summaries |
| steps | cumulative daily steps, count | 1.049 s; likely change-dependent updates | supported nonnegative increments only |
| floorsClimbed | cumulative floor count, floors | 651.840 s; sparse | supported nonnegative increments only |
| calories | likely daily cumulative energy, kcal not exporter-confirmed | 60.065 s | missing until units/counter meaning confirmed |
| intensityMinutes | cumulative minutes, weighting/reset period unconfirmed | 305.349 s; sparse | missing until semantics confirmed |

Cadences are inventory diagnostics, not new eligibility thresholds. Garmin's [Health API overview](https://developer.garmin.com/gc-developer-program/health-api/)
identifies HR and activity categories but does not prove this exporter implements a particular
counter/reset schema. Steps/floors interpretation is provisional, supported by units/names and
observed within-day monotonicity/date-boundary decreases. Do not interpret accepted sums as
complete-window activity or silently include unverified calorie/intensity counters.

Physical duplicates stay intact; reuse the logical view that removed only exact full-record
duplicates. For this new analysis view, a same-timestamp/same-value group contributes one reading
(the earliest original logical row) only if all group members have acceptable clock/value flags.
Keep an ignored row-level ledger of every original logical record and its disposition. Conflicting
timestamp values invalidate the entire group; never average. Identical values at different
timestamps remain separate readings. Counter adjacency is between original consecutive timestamp
groups, not a filtered list: rejected/conflicting groups sever pairs. Same-value tie collapse does
not manufacture elapsed time; the complete group is one observed point.

## HR predictors (all eight lookbacks, no retention-selected lookback)

A reading requires finite Unix time, no invalid epoch/text-clock/discrepancy/unusual-offset flags,
and a finite value in **30–240 bpm inclusive**. This broad provisional range is not a validated
physiological diagnosis; zero is rejected. Timestamp conflicts are rejected as above.

- Mean, median, minimum, maximum, Q25, Q75 and sample SD require **≥2 distinct valid timestamps
  with span ≥1 second**. SD uses N−1. Mean/SD are observation-weighted, not duration-weighted.
  Quantiles use NumPy's `method="linear"` order-statistic definition; this is a quantile estimator,
  not interpolation of missing sensor readings. [NumPy SD](https://numpy.org/doc/stable/reference/generated/numpy.std.html)
  and [quantile documentation](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html).
- Latest requires **≥1 valid observed reading**, span ≥0. Its actual age is separate quality data;
  no value outside the window is carried into it.
- Slope requires **≥3 distinct valid timestamps, span ≥60 seconds**; ordinary least squares of
  bpm against elapsed minutes since the first valid timestamp, with intercept. No weighting,
  resampling, inferred readings or extrapolated values. This describes observed points, possibly
  separated by gaps; it is not a continuously measured physiological trend. Gaps stay explicit.

Minimum count/span gates define provisional descriptive support, not proven reliability. A
long span does not imply continuous coverage. Sparse single readings can yield latest while
other statistics remain missing. Do not use the inventory's 2-second run rule as HR eligibility.

## Activity predictors: supported additions, not complete-window totals

Steps/floors values must be finite nonnegative integers, with the same clock/tie rules. For two
consecutive original timestamp groups, require both accepted, same exported local calendar day,
same inferred offset, same source content hash, strictly increasing Unix time, and **dt ≤300 s**.
The 300-second maximum is a prespecified conservative attribution limit matching the shortest
lookback, not a validated wear rule and not chosen from retention. It permits minute-cadence
jitter and does not promote the old 60-second diagnostic into eligibility.

A decrease marks a reset/conflict and gives no increment; never take its absolute value or count
it as negative activity. The accepted post-reset observation may begin the next same-day pair.
All date transitions are excluded, even when nondecreasing; hidden resets cannot be ruled out.
A gap >300 s, rejected endpoint or source/offset transition severs the pair. No bridging.

Attribute delta=current−previous only when the **entire pair interval [previous ts,current ts]**
is inside the lookback and current ts < cutoff. A preceding baseline outside the left boundary
is insufficient: exclude the entire crossing increment, never prorate or assign all of it.
At least one fully supported pair with positive duration is required. Sum only eligible deltas;
zeros are retained only if an actual accepted pair records no counter change. No accepted pair
means missing, not zero. Sum is named `*_added_supported`, not total daily/window activity.
No gap is filled; the quality table reports the union of supported pair intervals and omitted
boundary/reset/gap counts. Unknown hidden reset/update semantics remain limitations.

Calorie/intensity output columns remain missing with explicit semantics reasons and are **not**
in the predictor list. No Garmin stress, PulseOx, IBI or wearable-quality measure is a predictor
in this checkpoint. No physiological-rate cap is invented for counter deltas; large changes
remain inspectable diagnostics until provider specifications are available.

## One EMA becomes one row, with separate quality records

The ignored feature Parquet has exactly one row per original `n_ema_id`, participant and frozen
fold, with eight lookback suffixes. Only the explicit dictionary predictor list may enter any
future modeling task. Identity, category, cutoff/precision, quality and missingness reasons are
not predictors. No PSS target is read or attached. The separate quality Parquet uses the unique key `(EMA, sensor, lookback)` and carries recorded/valid/rejected/tie counts, per-feature status, freshness,
observed span, maximum/internal/edge gaps, occupied one-minute bins and counter-supported time.
Point HR observations have no asserted duration: occupied bins are a recording-density diagnostic,
not measured seconds. Gap thresholds 60/300/3600 s are diagnostics only. An ignored lineage
ledger preserves window slice indices/source hashes without copying sensor readings per EMA.

Availability distinguishes no recording, all rejected readings, insufficient count/span for each
feature, unresolved counter semantics, and no supported counter pair. Report survey-specific and
all-EMA counts, participant reach, HR/activity intersection, supported time and gaps. Do not add
overlapping windows' duration or choose rules using outcomes, retention or model performance.
