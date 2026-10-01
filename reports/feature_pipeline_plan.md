# Proposed feature pipeline — not implemented at this checkpoint

The next concrete implementation is a **raw-preserving, development-only preprocessing and
five-minute IBI eligibility audit**. It must resolve beat timing and define quality rules before
calling an HRV package. This checkpoint has not computed a prediction feature, HRV value,
PSS score, final linked dataset, imputation, model, or a new split.

## Evidence and decisions

Use the verified corrected CSVs and the exact frozen participant assignments in
`private/frozen_participant_assignments.parquet`. Preserve all five development folds; final-test
participants remain locked until the entire analysis is frozen. Prior normalized files are used
only to corroborate identity/provenance for keyless sources. Old features and models are not reused.
See `inventory_report.md` for raw-derived counts and `inventory.json` for detailed diagnostics.

The source contains timestamped IBI, HR, PulseOx, Garmin stress and four activity counters.
There is no raw ECG/PPG waveform, and normal-to-normal beat status is not established.
The absence of numeric missing cells would not imply uninterrupted wear: gaps, zeros,
duplicates and incomplete preceding history are different quality problems.

## Ordered implementation

1. Preserve all original strings with source archive/content SHA-256 and one-based row numbers.
   Retain physical copies while defining a separately documented logical view. Do not discard
   timestamp ties: identical rows, identical timestamp/value pairs and conflicting values are
   different cases. Validate the participant crosswalk against the frozen population and reject
   ambiguous respondent-only joins. Require independent provenance for any new keyless export.
2. Retain raw local clocks, parsed clocks, row offsets and UTC clocks separately. Use Unix seconds
   directly for Garmin time arithmetic. Keep discrepancy flags, original order and out-of-period
   records. Establish whether `ts` represents beat acquisition, buffering or receipt; compare
   within-stream time differences against interval durations without silently rebuilding time.
3. Define EMA cutoff `c = actual_start_utc - 60 s` for minute-resolution starts and
   `c = actual_start_utc` for second-resolution starts, matching the reference implementation.
   The provisional buffer remains until rounding/truncation is documented. Never anchor on
   `instance_id` (scheduled time), embedded EMA sensor fields, or an old PRE/DURING label.
4. Audit five-minute IBI windows with one-minute steps, anchored to each cutoff. For integer
   `j >= 0`, a candidate window is `[c - 60j - 300, c - 60j)`. Right boundaries are excluded.
   A lookback of `L` minutes contains only windows whose full support is inside `[c - 60L, c)`.
   This gives one candidate at 5 minutes and eleven at 15 minutes. Timestamped intervals may
   straddle a boundary: decide whether the timestamp marks beat/interval start or end before
   declaring the support wholly preceding the cutoff. No post-cutoff beat may enter a window.
5. Prespecify development-only quality sensitivities for physiologically plausible intervals,
   timestamp ties, gap sizes, edge coverage, artifact burden and duration/time consistency.
   The current report's 2-second HR/IBI and 60-second other-sensor gap thresholds are descriptive
   diagnostics, not validated eligibility rules. Calorie intervals slightly above 60 seconds
   can split runs even at an approximately minute cadence. No five-minute HR runs under the
   2-second diagnostic does not establish absence of usable HR data; define HR usability
   separately from IBI eligibility. Do not concatenate across a gap, treat missing beats
   as zeros, infer clean NN status from count alone, or silently interpolate/correct artifacts.
   Retain every rejection reason in private quality outputs; publish aggregate attrition.
6. Only after timing/eligibility review, compute approved interval-based HRV with explicit package
   versions and settings. Do not feed bpm or an uneven IBI series to a waveform peak detector.
   Keep rejected windows missing. Verify synthetic intervals and package agreement using matched
   definitions before computing real windows. Pin HRV dependencies in the same uv environment
   at that implementation step; they are deliberately not installed or invoked here.
7. Summarize valid five-minute windows over **5, 15, 30, 60, 120, 360, 720 and 1440 minutes**.
   Preserve the feature-sheet summaries specified for this checkpoint: **mean, SD, minimum,
   maximum, trend and latest**. Trend denotes slope against actual window times; its estimator,
   time units, weighting and minimum support remain to be prespecified. **Median and IQR are
   additional candidates**, not replacements for that planned list. These are custom summaries
   of window measurements, not additional package HRV indices. Overlapping windows
   are correlated and must not be treated as independent observations. Deduplicate equivalent
   computations while retaining each EMA's exact support and source provenance.
8. Later build one private EMA-linked row per participant/instance/survey category, retaining the
   original fold. Attach the complete-item PSS score with the verified 1–5 convention only at
   that authorized stage. Do not drop an EMA silently because all wearable windows are missing.
   Any later imputation, transformations or selection must fit within development training folds.

## Candidate families and exclusions

| Family | Evidence / proposed use | Conditions |
| --- | --- | --- |
| Time-domain HRV: mean NN, RMSSD, SDNN, pNN50 | IBI stream exists; plausible millisecond values | Clean eligible five-minute intervals and verified clock semantics |
| LF/HF band powers and selected frequency summaries | Potentially supported by sufficiently continuous IBI | Fix resampling, detrending, PSD, band and normalization settings; evaluate duration adequacy; no blanket autonomic interpretation |
| Poincaré SD1/SD2 | Potentially available from intervals | Avoid redundant SD1 and RMSSD predictors; prespecify compact set later |
| Entropy/fractal measures | Defer from initial set | Assess sample-length dependence and stability before inclusion |
| HR summaries | Direct timestamped bpm stream | Exclude invalid physiological readings; preceding support only |
| Activity increments | Four largely cumulative counters | Resolve daily rollover, within-day resets and observation gaps; never sum raw counters |
| PulseOx summaries | Timestamped percent-like values | Resolve zeros/sentinels and sparse coverage; include only if justified |
| Garmin stress and anything derived from it | Proprietary measurement | Exploratory only; excluded from initial modeling |
| Coverage, gaps, counts, zeros, artifacts, availability | Quality and attrition diagnostics | Not initial predictors, including missingness indicators |
| VLF, VLF ratios and total-power measures containing VLF | Duration/continuity insufficiently assessed | Deferred; do not import indirectly through package “all metrics” output |
| EMA free text, location, embedded device values | Sensitive or lacking preceding acquisition timing | Excluded from initial wearable predictors |

## Official package review (2026-09-30)

[NeuroKit2 HRV API](https://neuropsychology.github.io/NeuroKit/functions/hrv.html)
accepts interval input through a dictionary containing `RRI` and `RRI_Time`. Intervals are
milliseconds and their time coordinates are seconds. Select approved `hrv_time`, `hrv_frequency`
and `hrv_nonlinear` outputs explicitly; the convenience `hrv()` function returns broader families
than this study initially allows. [Its frequency implementation](https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/hrv/hrv_frequency.html)
resamples/interpolates by default, so settings and handling of gaps need deliberate control.
A package result does not certify source quality or recording continuity.

[HeartPy's official `process_rr` source and documentation](https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/_modules/heartpy/heartpy.html)
provide an interval-based entry point accepting a millisecond tachogram. Threshold and cleaning
options are explicit, with cleaning off by default. It accepts no independent timestamp vector,
so continuous segment eligibility must be established before calling it. The documented source
also warrants checking the installed frequency implementation and Welch-window handling rather
than assuming every argument is propagated. Review cleaning effects on adjacency; never merge
separate recordings merely because a function accepts one concatenated interval list.

Package calculations and custom operations must be labeled separately in a future manifest:
package/version, function, exact parameters, eligible input support, returned metric/units versus
custom window alignment, rejection policy, lookback aggregation and slope definitions.

## Clearly synthetic examples — no real participants or values

- **Sensor row:** `participant=SYNTHETIC_A`, `sensor=ibi`,
  `measurement_utc=2030-01-01T11:57:00.500Z`, `value=800 ms`, plus source/row provenance.
  This is one interval record, not HRV and not proof of its acquisition semantics.
- **HRV window:** a synthetic second-resolution EMA at `12:00:00Z` has a candidate window
  `[11:55:00Z, 12:00:00Z)`. Its eligibility, RMSSD and SDNN remain `not calculated` here.
  A minute-resolution EMA recorded at `12:00` instead uses cutoff `11:59:00Z` and candidate
  window `[11:54:00Z, 11:59:00Z)`.
- **EMA-linked row:** `SYNTHETIC_A`, synthetic event/category, actual start, cutoff, frozen
  partition/fold, `pss_score=not calculated`, `rmssd_15m_mean=not calculated`. A separate
  diagnostic table contains valid-window count, gaps and exclusion reasons. None of those
  diagnostics becomes an initial predictor.

## Uncertainty and acceptance gates

Analysis can quantify tie/conflict rates, interval-duration consistency, resets, gap patterns,
and sensitivity of eligible-window coverage in development participants. It cannot prove whether
a timestamp denotes acquisition versus upload, whether displayed minutes were rounded, the
meaning of undocumented sentinel codes, or an ambiguous identity without exporter/protocol
records. Do not shift clocks to maximize coverage or select quality rules using test outcomes.

The next implementation is ready when source/key integrity checks pass, clock exceptions remain
explicit, synthetic tests demonstrate strict preceding support, and the development-only eligibility
report separates short valid windows from long recording spans. HRV calculation and final linkage
remain later checkpoints.
