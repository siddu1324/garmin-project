# Provisional IBI eligibility policies — specified before the retention comparison

These policies were fixed before running this checkpoint's retention analysis. The canonical
configuration is `config/ibi_eligibility.json`; its hash is stored in each immutable run manifest.
No PSS targets, associations or sample-size objective enter policy definition or selection.
The labels below mean **provisionally eligible for further HRV input review**, not validated HRV
and not verified normal-to-normal beats. None is designated a winning policy based on retention.

| Requirement | conservative | screen | timing_sensitivity |
| --- | --- | --- | --- |
| Plausible IBI, inclusive (ms) | 300–2000 | 250–2500 | 250–2500 |
| Pair time tolerance, max(absolute, relative × expected interval) | max(0.10 s, 10%) | max(0.25 s, 25%) | max(0.50 s, 50%) |
| Consistent consecutive-pair fraction, minimum | 95% | 80% | 80% |
| Maximum consecutive timestamp gap / expected interval | 1.5 | 2.0 | 2.0 |
| Maximum left/right timestamp edge gap | 2.5 s | 5 s | 5 s |
| Union of acceptable interval support in 300 s, minimum | 285 s | 270 s | 270 s |
| Longest unbroken acceptable sequence, minimum | 240 s | 120 s | 120 s |
| Potential-artifact interval fraction, maximum | 1% | 5% | 5% |
| Sum of contained plausible IBI / contained timestamp span | 0.90–1.10 | 0.80–1.20 | 0.80–1.20 |

All policies require at least two contained interval records, no implausible/missing IBI among
timestamp members, no remaining timestamp ties (even equal-valued ties), and no flagged clock
residual or unusual offset among timestamp members. Exact source-record duplicates alone are
collapsed in the logical view, with all copies retained physically. Conflicting IBI values are
never averaged. The same interval value at different timestamps is legitimate repetition and
is never removed merely because its value repeats.

For each EMA cutoff `c`, enumerate `[c - 60j - 300, c - 60j)` for integers `j = 0..1435`.
Only `j <= lookback_minutes - 5` belongs to that lookback. Every development EMA is retained,
including invalid clocks or missing Garmin. A separate EMA accounting table records invalid
cutoffs; no invented timestamps are assigned. Left boundaries are included, right boundaries
excluded. One window record has all six policy/scenario results, avoiding duplicate window rows.

## Timestamp and boundary alternatives

For a record with timestamp `t` and interval duration `r` seconds, the interval-end scenario
assigns hypothetical support `[t-r, t]` and compares `t[i]-t[i-1]` to **current** `r[i]`.
The interval-start scenario assigns `[t, t+r]` and compares the same difference to **preceding**
`r[i-1]`. These support intervals are hypotheses, not reconstructed beat timestamps. For either
scenario the timestamp itself must be in the window, and the entire hypothesized interval must
lie inside it (an interval endpoint may meet the boundary without extending beyond it).
Cross-boundary intervals are excluded only from that scenario's diagnostic subset and counted
explicitly; they remain in the normalized dataset. No values after the window are borrowed.

Potential artifacts flag **both endpoints** of a >20% interval change relative to the preceding
interval, only for positive-time neighbors separated by at most twice their maximum interval
plus 0.25 s. This is an unvalidated discontinuity flag; ectopy and normal beats cannot be
identified without waveform/beat annotations. Acceptable support excludes implausible, tied,
clock-flagged and artifact-flagged intervals. A run breaks at every excluded record and every
pair failing that scenario's timing tolerance or maximum expected-gap ratio. Dropping a record
never creates a new adjacent pair. Supported seconds are the geometric union of acceptable
hypothesized intervals; overlaps count once and actual holes remain holes. Sequence duration
is first-to-last timestamp span; its small tolerated timing errors do not prove continuous physiology.

Coverage reports distinguish timestamp presence, interval plausibility, provisional eligibility,
and normal-beat status (always unverified). Across windows/EMAs, time unions are calculated per
participant before summing: neither overlapping windows nor repeated EMA histories inflate
unique supported minutes. Category totals may overlap and must not be summed. A single eligible
five-minute segment in a 24-hour lookback is not complete 24-hour coverage.

## Rationale and sources

The [1996 ESC/NASPE Task Force](https://www.escardio.org/static-file/Escardio/Guidelines/Scientific-Statements/guidelines-Heart-Rate-Variability-FT-1996.pdf)
establishes the short-term five-minute convention. [Laborde et al. (2017)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2017.00213/full)
emphasize consistent duration, artifact assessment and reporting. These sources motivate the
window length and explicit quality audit; **they do not validate our numerical thresholds**.
The interval limits represent broad rate plausibility screens (30–200 bpm versus 24–240 bpm),
not clinical definitions. The support, edge, artifact and timing thresholds are explicitly chosen
research sensitivities. The third policy isolates sensitivity to clock/interval agreement by
changing only the timing tolerance from the broader screen. No inventory 2 s / 60 s run rule
is being promoted to eligibility; gaps here scale with each hypothesized interval duration.

[NeuroKit2's official interval API](https://neuropsychology.github.io/NeuroKit/functions/hrv.html)
accepts RRI in milliseconds and interval-time coordinates in seconds. Timestamp-aware input
should remain explicit if used later; do not use cumulative sums to fabricate an uninterrupted
clock. [HeartPy's official `process_rr` documentation/source](https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/_modules/heartpy/heartpy.html)
accepts millisecond intervals and exposes optional cleaning; it has no separate timestamp
argument. Segment boundaries therefore need external enforcement. Its filter parameters are
not the policies above. Neither package is called in this checkpoint, and no correction,
interpolation, imputation or synthetic beat insertion is performed.
