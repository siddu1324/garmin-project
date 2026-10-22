# Notebook 16: binary stress improvement

Read `index.html` for the self-contained presentation and `summary_for_will.md` for the findings.
`specification.md` and `config/binary_improvement.json` were frozen before new performance.
`context_control_amendment.md` discloses 40 context-only fits specified after reviewing the original
600 fits. These controls isolate physiological information from rebalancing effects.
`audit.md` distinguishes confirmed behavior, possible problems and missing measurement evidence.

Inputs are the unchanged development matrix
`private/comprehensive_features/features/eecf1da995c39b27/`, saved Morning retrospective bundle
`private/morning_recall/runs/4c844f737b74ee62/`, and prior binary experiment
`private/binary_6hour/runs/11442e75c7e434aa/`. Previous artifacts are never overwritten.

Current experiment outputs are in `private/binary_improvement/runs/911ce0ce6a7b6943/`.
`accounting.parquet` retains all 3,706 development rows. `all_predictions.parquet` contains
repeated experiment views of real OOF responses, not independent observations. Each `jobs/`
bundle contains the estimator, training/validation predictions, inner memberships/OOF probabilities,
donor lineage, tuning trials, selected features, resampled counts and a checksum manifest.
The controls are in `controls/2533d80203c85392/`; `context_controls.json` records their provenance.
`primary_stage_aggregates/`, `primary_all_predictions.parquet` and `primary_bootstrap_draws.npz`
preserve the original 600-fit analysis before the disclosed amendment.
All individual artifacts are ignored/private. No raw sensor samples are synthesized.

Public aggregate tables and figures are in `reports/binary_improvement/911ce0ce6a7b6943/`.
`metrics.json`, `paired.json`, `fold_metrics.json`, `fold_support.json`, `thresholds.json`,
`per_class.json`, `calibration.json`, `selected.json`, `resampling.json` and
`selection_stability.json` retain the detailed results behind the presentation.
`audit/` stores source checks, classes, quality, missingness and cohort composition.

Reproduce in the existing pinned environment, without adding dependencies:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_binary_improvement.audit
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m garmin_binary_improvement.run --workers 4
# Initial build only, before sealing; completed controls are verified by the verify command.
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m garmin_binary_improvement.controls
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_binary_improvement.evaluate
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_binary_improvement.diagnostics
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_binary_improvement.verify
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/garmin-binary-improvement-mpl .venv/bin/python -m garmin_binary_improvement.report
PYTHONPATH=src .venv/bin/python -m garmin_binary_improvement.privacy
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests
```

Completed job hashes are verified and reused. Changes to fitting code, configuration or inputs
create a different version. The completed aggregate notebook is
`notebooks/16_binary_stress_improvement.ipynb`; it reads only reviewed aggregate results.
For a later execution check, copy it to an ignored directory instead of changing its sealed metadata.
No final-test evaluation, raw modification, commit, push or provider message is authorized here.
