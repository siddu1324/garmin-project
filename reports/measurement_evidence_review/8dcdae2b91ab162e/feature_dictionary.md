# Five IBI measures — proposed expanded output dictionary

Status: definitions already validated in the limited pilot; **expanded calculation remains blocked
pending measurement evidence**. Label outputs “IBI variability—normal-beat status unverified.”
Mean and median describe interval level; the other three describe variability. No diagnostic or
normal-to-normal physiological validity is implied by the conventional SDNN/pNN50 names.

For one uninterrupted run, let x be original IBI values in milliseconds, N its accepted interval
count, A the set of permitted original adjacent pairs, P=|A|, and d their signed differences. A
rejected interval or broken pair ends a run; removing records must never create new pairs. For one
complete contiguous run P=N−1. Mathematical minima below are not duration or physiological eligibility.

| Column | Definition | Unit | Mathematical minimum | Role |
| --- | --- | --- | --- | --- |
| mean_ibi_ms | sum(x)/N | ms | N≥1 | candidate physiological summary, semantics provisional |
| median_ibi_ms | ordinary median of x | ms | N≥1 | candidate physiological summary, semantics provisional |
| sdnn_sample_ms | sqrt(sum((x−mean(x))²)/(N−1)) | ms | N≥2 | candidate variability; NN status unverified |
| rmssd_ms | sqrt(sum(d²)/P), using only A | ms | P≥1 | candidate variability; no cross-break differences |
| pnn50_pairs_percent | 100 × count(abs(d)>50 ms)/P | percent | P≥1 | candidate variability; exactly 50 ms does not count |

Undefined values remain missing. Do not concatenate runs, average run-level HRV into a five-minute
estimate by default, interpolate, manufacture beats, fabricate waveforms or clean automatically.
Observation windows stay 300 seconds with 60-second steps, [left,right), inside each approved
lookback, using the existing actual-start cutoff/precision buffer. Entire hypothesized intervals
must fit; crossing intervals are flagged and excluded without clipping their duration. Retain both
start/end hypotheses unless reviewed evidence resolves anchoring.

Screen window eligibility alone is insufficient for a full five-minute estimate: it allows only a
120-second uninterrupted span. The existing pilot additionally requires one run with BOTH span and
supported duration ≥240 seconds. A feature on that run is a ≥240-second run feature within a five-minute
observation, not complete five-minute HRV. No stronger physiological duration rule is inferred here.

## Separate quality and provenance records — never initial predictors

Retain exact window/run bounds, actual timestamps, raw/parsed values, source and archive hashes,
source/physical/logical row numbers, mapping provenance, development fold, hypothesis, policy/evidence
version, N, P, supported-time union, timestamp span, edge/internal gaps, quality/rejection flags,
run boundaries, normal-beat-unverified status and package call/block reasons. Keep these in separate
quality/provenance tables with stable run/window keys. They are not added to the predictor list.
Never use reading counts or overlapping-window counts as independent sample sizes.

## Package agreement to preserve

Existing independently validated arithmetic is authoritative for the declared pair set. HeartPy's
low-level `calc_ts_measures` accepts supplied adjacent changes: convert its population SD by
sqrt(N/(N−1)) and pNN50 fraction to percent. It has no median output. Do not call `process_rr`.
NeuroKit `hrv_time` comparisons are permitted only by the existing guarded adapter, with explicit
unchanged RRI/RRI_Time, end anchoring, compatible native 10-ms pairs and unchanged sanitization.
Convert its pNN50 to the valid-pair denominator only after verifying pair identity. Record blocked
calls; never retime data to obtain agreement. Pin installed source/version hashes before a new run.

This checkpoint makes no new package calls on real data. Garmin stress stays excluded; VLF and
other spectral/nonlinear features remain deferred. A future evidence-supported contract may need
an explicitly versioned duplicate/order correction, not silent reuse of provisional logical rows.
