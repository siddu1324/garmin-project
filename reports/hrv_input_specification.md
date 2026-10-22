# HRV input contract — development checkpoint

This contract prepares inputs; it authorizes **no real-data HRV calculation**. Keep all 32 final-test
participants sealed, the 125 development assignments/five folds unchanged, and quality measures
separate from predictors. Normal-beat status remains unverified. No spectral or additional nonlinear
feature is ready under this contract.

## Observation window, support and runs are different

An observation window is exactly 300 seconds, `[c−60j−300, c−60j)`, at one-minute steps. Preserve all
approved lookbacks and original cutoffs: EMA local start minus that row's local-minus-UTC offset;
subtract another 60 seconds for minute-resolution starts only. Garmin Unix seconds receive no extra
shift. Do not move a clock to improve alignment.

For each timestamp member `t`, preserve the original IBI value and all quality/provenance fields.
The two unchanged hypotheses are interval end `[t−r,t]` and interval start `[t,t+r]`, with `r` in
seconds. The timestamp must lie in the observation window and the entire hypothesized interval must
fit; an interval endpoint may touch the boundary. Crossing intervals are retained with an exclusion
flag, never clipped, borrowed, interpolated or reassigned to a fabricated beat time.

**Supported duration** is the union of acceptable hypothesized interval supports, with holes and
overlaps handled explicitly. **A provisional run** contains only adjacent original logical records
that pass the frozen policy's interval and pair checks. Its timestamp span and supported duration
are separate fields. Tolerated timestamp jitter does not establish physiological continuity; any
small geometric holes remain excluded from the support union. A recording gap beyond the policy's
expected-interval limit or timing tolerance breaks the run. Undetectable missing beats cannot be
ruled out by these checks.

Every rejected interval, remaining timestamp tie/conflict, clock flag, provisional jump flag,
external rejection or missing original logical row breaks adjacency. A pair failing the scenario's
timing tolerance also breaks it. Keep rejected rows with `run_id = -1`; never filter a vector and
then calculate new adjacent differences. The >20% jump rule is a provisional discontinuity screen,
not a diagnosis of artifact or ectopy. Exact export duplicates already collapsed by preprocessing
are not additional beats; equal-valued intervals at different timestamps remain legitimate records.

**Screen can pass with only a 120-second unbroken span.** A calculation on that run must say
“120-second run within a five-minute observation window,” not “five-minute HRV.” Likewise, a
240-second pilot run is a four-minute-or-longer run, not five-minute HRV. For a future five-minute
estimate require one uninterrupted input covering the five-minute observation under an explicitly
approved edge/boundary rule. The current window eligibility flag is not sufficient. Never average
or concatenate separate run-level HRV values into a five-minute estimate by default.

## Stored representation and access

The ignored diagnostic bundle stores a compact logical slice index for **every** approved window.
Immutable normalized Parquet retains actual timestamps, parsed IBI, source hashes/rows, raw-value
references, participant/fold provenance and quality flags. The combination is a lossless reference
representation; it avoids duplicating overlapping records billions of times.

`garmin_hrv_input.access.load_indexed_window` hydrates any indexed development window and returns
all its records, scenario-specific flags, original adjacency and run boundaries using
`contract.window_input`. It checks source-version identity and slice membership. Verify bundle
hashes once before a sequence of reads. A catalog participant with no IBI returns `None`; that
window remains present in the index and rejection accounting.

Immediate windows have materialized `immediate_members/`, `immediate_runs/` and `immediate_cases/`
Parquet tables for all three frozen policies and both hypotheses. Other lookback windows hydrate
on demand from the complete index; their run tables are not all redundantly materialized.
`n_physical_row` points to original string values in the physical source partition. Never expose
these tables, catalog keys or participant rows in notebooks or public reports.

Each record retains `c_plausible`, `c_contained`, `c_boundary_excluded`, `c_provisional_jump_flag`,
`c_external_rejection`, `c_acceptable`, `c_link_to_previous_original_record`,
`c_neurokit_native_pair_compatible`, `c_run_id`, and both hypothesized interval endpoints, alongside
original normalized fields. Each run retains original index bounds, interval count, allowed-pair
count, timestamp span, supported seconds, package-pair compatibility and unverified beat status.
Mathematical definability is distinct from permission and scientific eligibility.

## Proposed calculation definitions — one run at a time

Let `N` be the run's interval count, `P=N−1` its original adjacent pairs, and `d` the allowed signed
successive differences. All interval values/differences use **milliseconds**; clocks use seconds.

| Quantity | Project definition | Mathematical minimum |
| --- | --- | --- |
| Mean IBI | sum of IBI / N | N ≥ 1 |
| Median IBI | ordinary median of IBI | N ≥ 1 |
| SDNN convention | square root of sum of squared deviations / (N−1) | N ≥ 2 |
| RMSSD | square root of sum of squared allowed differences / P | P ≥ 1 |
| pNN50 | 100 × number of allowed absolute differences strictly >50 ms / P | P ≥ 1 |

Exactly 50 ms does not count. Undefined results remain missing, not zero. These minima only make
arithmetic defined; they do not establish adequate measurement duration. Without verified normal
beats, label a future result “IBI variability (NN status unverified),” not certified NN variability.
Mean/median IBI do not themselves measure variability.

The small proposed pilot additionally requires a frozen Screen-eligible window and a **single run
with timestamp span ≥240 s AND supported duration ≥240 s**, preserving start/end alternatives.
This is a transparent pilot restriction, not a new validated eligibility policy or a claim of
five-minute coverage. Select at most ten unique nonoverlapping runs in deterministic chronological
order within development participants, cap one per participant, and never select using outcomes or
feature values. Do not reuse overlapping EMA views as independent pilot samples.

## Package interfaces and safeguards

The installed versions inspected are **NeuroKit2 0.2.13** and **HeartPy 1.2.7**. Source hashes and
signatures accompany the synthetic report; online documentation can differ from installed releases.

- NeuroKit2 `hrv_time({"RRI": milliseconds, "RRI_Time": seconds})` accepts explicit timestamps.
  Native adjacency compares timestamp differences with **current** IBI at a **10 ms** tolerance.
  Its missing-interval heuristic is not the frozen policy and must not establish input quality.
  The adapter requires a single run, interval-end semantics, every native pair agreeing and unchanged
  sanitized timestamps/values. Otherwise it blocks instead of adjusting clocks. Start-anchored or
  jittered runs can use external adjacency arithmetic; direct NeuroKit pair results are not assumed
  equivalent. Native pNN50 divides by P+1; convert to the declared P denominator only after verifying
  the exact pair set. Sample SDNN already uses N−1. The public function also computes other time-domain
  summaries; only the five specified outputs are validated and retained.
- HeartPy `process_rr` accepts milliseconds but no timestamps. Even with `threshold_rr=False`,
  `clean_rr=False`, `calc_freq=False`, it attempts Poincaré and breathing computations. It is **not
  called**. Use `analysis.calc_ts_measures` with explicit absolute allowed differences and their squares,
  plus fresh dictionaries. This lower-level function performs no cleaning, interpolation, breathing
  or spectral estimation. Its SDNN uses population N: convert by multiplying by sqrt(N/(N−1)). Its
  pNN50 is a fraction with denominator P: multiply by 100. It has no median-IBI output; use independently
  validated median arithmetic rather than claiming a nonexistent package result.

Always supply explicit timestamps to NeuroKit. Its omitted-clock path constructs cumulative times
and may median-fill missing intervals for that purpose; that path is forbidden here. Its sanitization
can remove nonfinite intervals or reinterpret time units, so adapters reject such changes. No waveform,
peak reconstruction, resampling or imputation is used. Spectral, breathing and nonlinear interfaces
are deferred until their distinct input requirements are specified.

Only declared synthetic fixtures can enter this checkpoint's package adapters. No real-data package
calculation entrypoint is enabled. Provider confirmation of units, sensor modality, acquisition versus
receipt timing, start/end meaning, buffering, missed-beat behavior and beat-quality annotations remains
necessary for a physiologically interpretable pilot. Numerically validated arithmetic alone is not
physiological validation.

Sources inspected: [NeuroKit time-domain API and source](https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/hrv/hrv_time.html),
[NeuroKit interval utilities](https://github.com/neuropsychology/NeuroKit/blob/v0.2.13/neurokit2/hrv/intervals_utils.py),
[HeartPy time-domain source](https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/_modules/heartpy/analysis.html),
[HeartPy high-level RR source](https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/_modules/heartpy/heartpy.html).
