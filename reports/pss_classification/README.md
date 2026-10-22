# PSS classification checkpoint

This is an exploratory development-only classification experiment, authorized after Notebooks 00–11.
It changes the statistical task, not the verified three-item score or sensor features. The final test
remains sealed. The prespecified search is bounded; no candidate list or threshold changes after results.

- Specification: [specification.md](specification.md).
- Fixed configuration: [../../config/pss_classification.json](../../config/pss_classification.json).
- Presentation: [../../notebooks/12_pss_classification.ipynb](../../notebooks/12_pss_classification.ipynb).
- Findings: [report.md](report.md); research abstract: [abstract.md](abstract.md).
- Executed run: `8970bf5ebcc47bfd`.
- Aggregate results and figures: [8970bf5ebcc47bfd/](8970bf5ebcc47bfd/).
- Private input: `private/pss_classification/runs/8970bf5ebcc47bfd/development.parquet`.
- Private OOF predictions: `private/pss_classification/runs/8970bf5ebcc47bfd/out_of_fold_predictions.parquet`.
- Private job bundles: `private/pss_classification/runs/8970bf5ebcc47bfd/jobs/` (split membership,
  fitted transformations/estimators, candidate scores, probabilities). Never publish these artifacts.

The preliminary `f51cae6fa96c2296` private preparation bundle contains inputs only, no model results;
it was superseded before fitting when the training-source hash list was made explicit. It is preserved.

## Reproduction

Use the pinned project environment and original ignored input bundles. The existing `uv.lock` pins
XGBoost 3.0.5, LightGBM 4.6.0 and scikit-learn 1.9.1; macOS LightGBM requires OpenMP (available locally).
`OMP_NUM_THREADS=1` and one thread per estimator avoid nested oversubscription; four independent jobs
run concurrently. No GPU, random row-level split, outer early stopping or raw preprocessing is used.

```bash
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/garmin-mpl OMP_NUM_THREADS=1 .venv/bin/python -m garmin_pss_classification.run
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/garmin-mpl .venv/bin/python -m garmin_pss_classification.evaluate private/pss_classification/runs/8970bf5ebcc47bfd
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/garmin-mpl .venv/bin/python -m garmin_pss_classification.figures reports/pss_classification/8970bf5ebcc47bfd
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/garmin-mpl .venv/bin/python -m garmin_pss_classification.verify private/pss_classification/runs/8970bf5ebcc47bfd
```

The training entrypoint reuses completed checksum-verified immutable jobs. It creates a different
version when training source/configuration/input identity changes. Aggregation/figure writers refuse
to overwrite existing result directories; use a separate copy of the workspace for an independent
reproduction. Do not delete validated outputs to force a rerun. The notebook renderer reads saved
aggregate results, avoiding an accidental repeat of the full nested experiment.

```bash
MPLCONFIGDIR=/private/tmp/garmin-mpl .venv/bin/pytest -q
.venv/bin/ruff check src tests
MPLCONFIGDIR=/private/tmp/garmin-mpl .venv/bin/jupyter nbconvert --to notebook --execute notebooks/12_pss_classification.ipynb --output 12_pss_classification.ipynb --output-dir private/pss_classification/executed --ExecutePreprocessor.timeout=300
```

After reviewing execution outputs for privacy, the executed Notebook 12 is copied to its requested
source location. This notebook alone has an explicit aggregate-output exception to the historical
output-free notebook policy; all previous notebooks stay unchanged. `verification.json` reports split,
preprocessing, prediction-replay and source-integrity checks. The companion checkpoint verification
records notebook execution, tests, Ruff and privacy scanning.

## Reading the tables

- `metrics.json`: pooled OOF metrics with full-fold status and cohort counts.
- `fold_metrics.json`, `training.json`, `survey_metrics.json`: fold stability, training fit and survey composition.
- `per_class.json`, `confusion_matrices.json`: class-specific errors (observed rows, predicted columns).
- `paired_differences.json`: paired participant-bootstrap macro-F1 differences. Positive favors the named model.
- `cohort_accounting.json`, `class_distribution.json`: all 3,706 development rows accounted for.
- `common_lookback_metrics.json`: common **validation** subset; native training cohorts still differ.
- `settings.json`: every selected configuration and every candidate's inner-fold scores; no participant identifiers.
- `importance.json`: validation permutation decreases, descriptive only; not a feature-selection mechanism.

For the ≥5 binary task, fields named `elevated_*` mean the positive class **PSS ≥5**, which includes
middle and high responses. Only three-class label 2 and the ≥10 binary task isolate the requested high range.
All missing physiological values are feature-level missingness. No target, raw reading or recording interval
is imputed. A probability from balanced training is not assumed calibrated.
