# Binary PSS six-hour experiment

Open `index.html` locally for the standalone presentation. `summary_for_will.md` explains the
actual findings. `specification.md` was fixed before new performance calculations. The executed
aggregate-only notebook is `notebooks/15_binary_pss_6hour_experiments.ipynb`.

## Inputs and exact outputs

- Reused unchanged feature bundle: `private/comprehensive_features/features/eecf1da995c39b27/`.
  `development_features.parquet` retains all 3,706 development responses. `dictionary.json`
  defines 438 supported candidates and 240 provisional PulseOx candidates. PulseOx is excluded here.
- Existing support/provenance inputs: that bundle's `quality.parquet` and `lineage.parquet`.
  No raw sensor extraction is repeated. The `source_partitions.json` and bundle manifest retain
  original source hashes. Existing corrected time boundaries and feature support rules remain fixed.
- Current run: `private/binary_6hour/runs/11442e75c7e434aa/`.
  `ema_accounting.parquet` preserves inclusion/exclusion status, outcomes and original folds privately.
  `all_predictions.parquet` holds repeated experiment views of OOF responses, not independent samples.
  `quality_accounting.parquet` and `new_readings_120m.parquet` / `new_readings_360m.parquet` are private.
  `jobs/` stores fold estimators, training/validation predictions, inner memberships, trials and manifests.
  `bootstrap_draws.npz` contains paired metric draws, conditional on the saved predictions.
- Public aggregates and figures: `reports/binary_6hour/11442e75c7e434aa/`.
  `metrics.json`, `fold_metrics.json`, `paired_differences.json`, `fold_support.json`,
  `settings.json`, `selection_stability.json`, `quality.json`, `missingness.json`,
  `new_readings.json` and `composition.json` distinguish fitting, evaluation and recording support.
- `feature_inventory.json` lists every predictor column in each block. The complete original
  definitions remain in `reports/comprehensive_features/feature_dictionary.md`.

Each single-window full block has 47 physiological candidates. Six earlier-history columns and
adjacent-window contrasts are excluded because their support extends outside the named interval.
The 438-feature multiscale sensitivity includes those supported features and is explicitly separate.
Context uses two cyclic local-time columns and training-encoded survey category. Missingness flags
are separately named recording indicators. The quality-only model is a separate control.

## Reproduce and verify

Use the existing `.venv` and pinned `uv.lock`; no dependencies were added.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m garmin_binary_6hour.run --workers 4
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_binary_6hour.evaluate
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_binary_6hour.verify
PYTHONPATH=src .venv/bin/python -m garmin_binary_6hour.report
PYTHONPATH=src .venv/bin/python -m garmin_binary_6hour.privacy
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests
```

The immutable run identity pins model source, reused preprocessing, configuration/specification,
feature bundle, assignment and package hashes. Completed model jobs are verified and reused. Changed
fitting inputs/code create a new version. Final manifests and the presentation completion seal protect
the checkpoint. The support audit is run once before sealing; it refuses to overwrite sealed outputs.
To independently execute the notebook after sealing, copy it to a private review directory rather
than modifying the delivered notebook's execution metadata. Its cells read only public aggregates.

The checkpoint-local privacy validator authorizes only Notebook 15's reviewed aggregate outputs.
Prior privacy code and Notebook 13/14 seals remain unchanged. Participant keys, actual timestamps,
individual outcomes/predictions, fold memberships and estimators are ignored/private.

No final-test sensor/outcome evaluation, raw modification, regression search, new physiological
extraction, commit, push or provider message occurs in this experiment.
