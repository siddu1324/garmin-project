# Comprehensive PSS regression checkpoint

Open `index.html` locally for the standalone presentation. `summary.md` explains the findings;
`specification.md` fixes the cohorts, model grids, feature controls and evaluation before fitting.
The executed aggregate-only notebook is `notebooks/14_comprehensive_pss_regression.ipynb`.

## Exact artifact paths

- Reused, unchanged feature dataset: `private/comprehensive_features/features/eecf1da995c39b27/development_features.parquet`.
  All 3,706 development EMA rows; 438 main supported candidates and 240 separate provisional PulseOx candidates.
- Original feature definitions: `reports/comprehensive_features/feature_dictionary.md` and
  `reports/comprehensive_features/quality_and_context_dictionary.md`.
- Current regression run: `private/comprehensive_regression/runs/4e769e522b239f43/`.
  `ema_accounting.parquet` retains every EMA and its outcome/context/support status.
  `all_predictions.parquet` contains repeated private OOF experiment views, not independent responses.
  `jobs/` contains immutable fold estimators, training/validation predictions, inner memberships,
  tuning trials, training-only feature selections and model metadata.
- Public aggregates and eight figures: `reports/comprehensive_regression/4e769e522b239f43/`.
  `metrics.json`, `fold_metrics.json`, `strata.json`, `paired_differences.json`, `settings.json`,
  `feature_stability.json`, `legacy_comparison.json` and `restricted_evaluation.json` distinguish
  evaluation cohorts, models, training populations and score-band diagnostics.
- Detailed checks: `verification.md`, the aggregate `verification.json`, and `completion.json`.
- Preserved metadata-export predecessor: `private/comprehensive_regression/runs/b3eedd7230280409/`.
  See `supersession.json`. It is not the completed benchmark and its partial results are not used as a
  separate comparison. Completed compatible jobs were copied after immutable-input and fit-path checks;
  unpublished XGBoost jobs were rerun with strict-JSON-safe metadata.

All participant keys, actual clocks, outcomes attached to keys, memberships, estimators and individual
predictions remain ignored/private. No participant rows are displayed in the notebook or HTML.

## Reproduction

Use the existing `.venv` and `uv.lock`; no dependency changes are needed. Raw preprocessing is not repeated.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m garmin_comprehensive_regression.run --workers 4
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m garmin_comprehensive_regression.evaluate
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m garmin_comprehensive_regression.verify
PYTHONPATH=src .venv/bin/python -m garmin_comprehensive_regression.report
PYTHONPATH=src .venv/bin/python -m garmin_comprehensive_regression.privacy
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests
```

The current run's identity includes input/configuration/specification, frozen assignment, source and
package hashes. Model jobs are independently checksum-verified and resumable. Changed inputs/code create
a new version. Final public aggregates and presentation are sealed; replay commands verify and reuse them.
Create a new report directory for a later experiment instead of overwriting this checkpoint. To execute
the notebook independently, copy it to a private review path and run `jupyter nbconvert --execute` there;
do not change the sealed delivered notebook's execution metadata.

The privacy scan uses the existing token-boundary rules and schema-validated aggregate exceptions, with
equivalent compiled matching for efficiency; regression tests verify those semantics. The existing
privacy validator is extended narrowly to permit the explicitly requested executed Notebook 14.

This analysis does not make acquisition-time, physiological-validity, clinical-diagnosis or causal claims.
It preserves the final-test seal, raw files, old bundles and folds. No commit, push or provider messages.
