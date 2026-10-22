# Notebook 10 verification

- **143 tests passed**, including 11 new tests for exact history boundaries, UTC/partial bins,
  equal-minute weighting, missingness, duplicate/conflict handling, frozen support rules,
  future/current exclusion, reference identity/folds, nested transforms and distinct retrospective associations.
- Ruff passes source, tests and Notebook 10. All **7 notebook code cells** execute without errors;
  source notebook is output-free. Executed text outputs pass privacy checks.
- All **3,706 reference windows** independently replayed using Python datetime boundaries, Boolean
  masks and timestamp-based UTC bins. Counts/support and reference/deviation arithmetic match.
  The **55,966,036 accepted reading/window views** include repeated overlapping views, not unique
  readings or continuous coverage; they must not be interpreted as independent observations.
- All **40 outer learned models** verified: independent closed-form Ridge, deterministic HGB replay.
  All **480 inner split/transform checks** pass, including original outer-participant separation,
  group-only inner validation, selected inner-MAE settings and training-only preprocessing.
- **180 metric rows** independently recomputed; **26 paired participant-bootstrap contrasts** replayed.
  Sixteen retrospective variance/association checks and unchanged composite coding independently pass.
- Explicit predictor schema excludes person means, EMA answers, quality/missingness and IDs from X.
  Full-record means remain in-memory retrospective diagnostics only; no person-level means are saved.
- Raw/reference input integrity and frozen assignments pass. **18 prior private manifests and all their
  artifacts** match; **450 prior public/source files** remain unchanged. README and the narrow privacy
  notebook/config allowlist were intentionally extended; all earlier unrelated work is preserved.
- **468 public source/text artifacts** privacy checked. Small-cell public outcome/quality summaries
  are suppressed, retaining counts; private row-level outputs remain ignored. No participant IDs or
  complete participant rows are displayed. No staged paths, commit, push or provider messages.
- Deterministic rerun returns the same immutable bundle without refitting or overwriting artifacts.
  All 32 final-test participants remain sealed; zero evaluated.

## Current artifacts

- Notebook: `notebooks/10_within_person_signal_and_past_hr_reference.ipynb`.
- Immutable private computation: `private/past_hr_reference/runs/510f1ad96a9e46f3`.
- Current aggregate report: `reports/past_hr_reference/d021427887616868/report.md`.
- Professor summary and feature-status/provider draft tables are in that report directory.
- Executed notebook: `outputs/10_within_person_signal_and_past_hr_reference.510f1ad96a9e46f3.executed.ipynb` (ignored).
- All-response tables: 3,706 reference-feature, quality, lineage, predictive-input and accounting rows.
- OOF predictions: 1,575 response rows; training predictions: 6,300 repeated outer-training memberships.

```sh
PYTHONPATH=src .venv/bin/python -m garmin_past_hr_reference.pipeline --reference "$GARMIN_REFERENCE_PROJECT"
PYTHONPATH=src .venv/bin/pytest -q
.venv/bin/ruff check src tests notebooks/10_within_person_signal_and_past_hr_reference.ipynb
```

An earlier aggregate rendering is preserved; the current pointer uses the lint-clean renderer,
with identical numerical results. No prior validated artifact was overwritten.

## Limitations

The fixed engineering history rule is not a physiological standard. A recent endpoint and 572
median occupied minutes coexist with long gaps; no continuous/daily/resting baseline is established.
Timing availability, acquisition/receipt, batching, sensor-quality flags and IBI semantics remain
provider questions. Retrospective associations are not prediction. Conditional fixed-prediction
bootstrap intervals omit refitting/tuning uncertainty and multiple-comparison correction. Nested
folds do not erase repeated development exploration. The experiment does not support automatic
model expansion or final-test evaluation. Local notebook execution required Jupyter loopback socket
permission and succeeded. Sandbox CPU-cache probe warnings did not affect deterministic replay.
