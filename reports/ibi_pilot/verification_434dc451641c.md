# Notebook 03 pilot verification

Completed the limited real-data pilot. All 32 final-test participants remain sealed; no outcomes,
PSS, imputation, EMA feature aggregation, spectral/nonlinear features, models, commit or push.

- **72 tests passed**, covering timestamps/offsets, boundaries and both hypotheses, duplicates/conflicts,
  broken adjacency, unique-time coverage, cohort/fold separation, deterministic selection, dual duration
  gates, guarded entry, strict >50-ms denominator rules and package agreement. The new clock regression
  reproduces the historical int16 overflow and verifies exact boundaries through 24 hours.
- Ruff passes on `src tests notebooks/03_ibi_variability_pilot.ipynb`; `git diff --check` and
  `UV_CACHE_DIR=/private/tmp/garmin-uv-cache uv lock --check --offline` pass. Full-repository Ruff also
  exposes an existing unused `Markdown` import (F401) in Notebook 02. That historical source notebook
  is preserved unchanged; this is a known lint limitation, not a Notebook 03 failure.
- Notebook 03 executed all **5 code cells**, without errors, using the existing Python 3.11 environment.
  Source outputs remain empty. The executed copy is ignored at
  `outputs/03_ibi_variability_pilot.78af22e7f0ead75b.executed.ipynb`.
  Jupyter required local loopback sockets outside the filesystem sandbox; no remote service was used.
- HeartPy **20/20 comparisons passed** for its four supported outputs after documented conversions;
  median has no HeartPy counterpart. Maximum absolute discrepancy was 2.132e−14.
  NeuroKit2 **20 calls were blocked before calculation**: ten spacing mismatches, ten start-anchoring
  mismatches. Actual timestamps were never altered to satisfy a package.
- All selected source-member fields exactly match the immutable normalized stream, including raw
  values, timestamps, quality flags and provenance. Original logical-row adjacency, whole-interval
  boundaries and Screen eligibility were independently rechecked. Span and supported duration each
  exceed 240 s. Reference arithmetic replays only the selected runs.
- The **5,321,816** corrected-index windows lie wholly within their intended pre-EMA 24-hour grid.
  **3,294,634** clock locations required correction; old bundles remain intact. Complete all-EMA
  accounting contains **59,296** rows (3,706 × two hypotheses × eight lookbacks). Unique supported time
  was recomputed as interval unions, and every feasibility denominator/count reconciled.
- Deterministic selection replays exactly. Re-running both guarded phases returned the same version
  directories with unchanged artifact hashes and no additional feature calculation. The selection
  manifest/hash was published before the calculation bundle and recorded there before features.
- Raw/reference input tree hashes, sizes and modification times match the original baseline. Frozen
  assignments still match the inventory hash: **125 development / 32 final-test, five development folds**.
  **64 prior files** outside the three documented edit locations match the start-of-task hashes,
  including original synthetic adapters, notebooks, configuration, dependency files and prior reports.
  Edits to existing locations are README additions/current-scope wording, privacy allowlist additions,
  and the one-line int64 generator fix. No prior bundle was overwritten.
- Known identifier/private-filename scanning passed for **82 public files** before this verification
  document was added. Credential/local-home-path scanning passed; executed notebook text passed the
  same known-token scan. All **138 private bundle artifacts** are ignored. Figures were visually
  checked and contain aggregate data only. Git's index is empty.

## Exact private outputs

| Ignored dataset | Rows |
| --- | --- |
| `private/ibi_pilot/calculation/59b11baba770afb0/package_comparisons.parquet` | 40 |
| `private/ibi_pilot/calculation/59b11baba770afb0/run_level.parquet` | 20 |
| `private/ibi_pilot/calculation/59b11baba770afb0/selected_window_members.parquet` | 8,482 |
| `private/ibi_pilot/selection/52eb55d251e1d5e1/candidate_views.parquet` | 3,772 |
| `private/ibi_pilot/selection/52eb55d251e1d5e1/ema_availability.parquet` | 59,296 |
| `private/ibi_pilot/selection/52eb55d251e1d5e1/selected_exact_views.parquet` | 33 |
| `private/ibi_pilot/selection/52eb55d251e1d5e1/selected_runs.parquet` | 20 |

Corrected Screen windows are partitioned beneath
`private/ibi_pilot/selection/52eb55d251e1d5e1/corrected_screen_windows/` (5,321,816 total rows).
Public tables/figures and the plain-language report are in `reports/ibi_pilot/78af22e7f0ead75b/`.
The selection protocol is `reports/ibi_pilot_protocol.md`; reusable modules are in `src/garmin_ibi_pilot/`.

## Remaining limitations

The 20 rows describe ten runs per hypothesis, with eight shared participants, four overlapping
source-run pairs and one identical source range. Neither the two hypotheses nor overlapping EMA
views are independent samples. Span ranges are 253.245–285.394 s (end) and 250.933–272.584 s (start);
supported ranges are 241.820–272.599 s and 241.151–264.087 s. These are not complete five-minute HRV.

Earlier runs improve availability to 334 EMAs/51 participants (end) and 262 EMAs/40 participants
(start) within 24 hours, but only one immediate EMA qualifies under each hypothesis. Chronological
selection is a convenience sample; age relative to the representative EMA can be many hours.
Numerical correctness cannot establish normal beats or acquisition semantics. Obtain provider
clarification before expanding physiological HRV use. The complete historical three-policy
long-lookback audit has not been reissued; its 12/24-hour counts must not be reused.
