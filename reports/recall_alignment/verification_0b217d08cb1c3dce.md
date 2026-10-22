# Notebook 07 verification

Current [notebook](../../notebooks/07_recall_alignment_and_model_diagnostics.ipynb),
[report](824551e40b8e301d/report.md), [briefing](824551e40b8e301d/professor_briefing.md),
and [frozen specification](../recall_alignment_specification.md).

- **118 tests passed**, including six new synthetic tests for AM/PM and UTC offsets, invalid/nonpositive
  wake clocks without date repair, minute-boundary uncertainty, cutoff exclusion, partial bins,
  rejected/absent readings, matched missing-recording membership, training-only target reads/scaling,
  held-out-label perturbation and objective arithmetic.
- **Ruff passed** for `src`, `tests` and Notebook 07. Environment and lockfile were preserved.
- **Notebook execution passed**, seven code cells; source notebook remains output-free. The final
  executed copy is `outputs/07_recall_alignment_and_model_diagnostics.0b217d08cb1c3dce.final.executed.ipynb`. Only aggregate/synthetic displays; no participant rows.
- **Independent verification:** all 3,706 development EMAs preserved in each scenario; 1,862 wake
  datetime checks and all 1,662 candidate coverage views checked by Python datetime arithmetic and
  Boolean masks against original accepted-reading ledgers, independently of vectorized window lookup.
  Right boundaries equal the precision-buffered cutoff; no shifted clocks or fabricated prompt boundary.
- **Training checks:** all 45 fixed fits converged; 50,220 training prediction rows replayed by independent
  matrix arithmetic and loss/penalty calculation. Maximum discrepancy 8.9e-16.
  All 15 alpha=.01 fits reproduce saved parameters/scalers/encoders and objectives. Fifteen saved
  prediction groups were checked without reading validation outcomes. No new validation predictions/scores.
- **Determinism:** guarded rerun reused the immutable current bundle. Three fresh builds made during
  implementation have byte-identical boundary, coverage, training prediction, parameter and training
  diagnostic artifacts; later versions strengthen source-log evidence and restrict timing reads to an
  explicit non-outcome column allowlist. Earlier artifacts remain preserved and are not current inputs.
- **Provenance/integrity:** current corrected feature/window parents verified. Raw/reference hashes,
  sizes and modification times match the original baseline; assignments and original five folds unchanged.
  Eleven prior standard-format private bundles passed full checksum verification; two legacy manifests
  lacking the newer `complete` field have exact artifact checksums and are checked as history only.
- **Preservation:** 237 existing source/documentation files are byte-identical. Existing README updated
  with the current checkpoint; existing privacy helper only adds the exact new notebook/config allowlist
  entries. Prior uncommitted work, dependency changes and outputs remain intact. Git index is empty;
  no commit, push or history rewrite.
- **Privacy:** current private bundle files and executed notebook are Git-ignored. Known identifiers and
  source filenames are checked across public artifacts and human-readable executed-notebook output;
  credentials/local absolute paths scanned. Two aggregate figures inspected visually. The machine-readable
  verification record records the final public scan count. No messages were sent to the provider.

## Exact outputs

- `private/recall_diagnostics/runs/0b217d08cb1c3dce/boundaries.parquet`: 7,412 rows (3,706 EMAs × two scenarios).
- `private/recall_diagnostics/runs/0b217d08cb1c3dce/candidate_coverage.parquet`: 1,662 candidate views (836 nominal + 826 later-minute).
- `private/recall_diagnostics/runs/0b217d08cb1c3dce/training_predictions.parquet`: 50,220 training-fit rows; repeated memberships are not independent observations.
- `private/recall_diagnostics/runs/0b217d08cb1c3dce/training_parameters.json`: 45 private coefficient/scaler/encoder audit records.
- `reports/recall_alignment/824551e40b8e301d/`: aggregate JSON tables, two figures, source citations, report and professor briefing.
- `src/garmin_recall_diagnostics/`: reusable guarded pipeline, recall coverage, training-only diagnostic,
  report generator and independent verification; `config/recall_diagnostics.json` is frozen.

## Limits

All 32 final-test participants remain sealed. Schema-only archive inventory does not inspect their
sensor/response values. Self-reported wake answers are retrospective and may be rounded or inaccurate;
the one-minute sensitivity is not a confidence interval for waking. Applying the actual-start row offset
at waking is provisional. No full previous-prompt/delivery/submission log is supplied. Sensor acquisition
versus receipt/batching and actual-start app-event semantics need provider clarification. Accepted bins
are not continuous coverage. Constant solutions and tiny training improvements do not establish
physiological validity, generalization, or an optimal penalty/window. No new target coding, HRV, stress,
imputation, nonlinear model or binary cutoff was introduced.

The sandbox's PyArrow CPU-cache probing notices are environmental; all Parquet operations and numerical
checks completed. Historical artifacts were never silently repaired or untracked.
