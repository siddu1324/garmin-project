# Central research overview

Open `notebooks/21_central_research_overview.ipynb`: 15 executed code-only sections, aggregate tables and plots. No model fitting or raw-data preprocessing is repeated.

- `complete_feature_inventory.csv`: all 678 candidates, formulas, units, support rules, status and per-feature missing counts/percentages over all 3,706 development EMAs.
- `missingness_modality.csv`, `missingness_lookback.csv`: missing feature-cell counts and percentages. Denominator = responses × feature columns; these are not counts of unique missing responses.
- `sensor_inventory.csv`: development physical/logical record totals and duplicate excess by modality.
- `actual_model_inputs_by_fold.csv`: exact Notebook 13 full-logistic input names by frozen fold. Context and missingness indicators are explicitly distinguished from physiological values.
- `verification.json`: current checks and separately labeled prior replay checks.

The 438 supported columns are observed summaries under provisional measurement assumptions; 240 PulseOx columns remain a sensitivity block. Primary full-logistic fits use 881 encoded inputs in folds 0–3 and 880 in fold 4. The 3,689-response full-feature results and 1,355-response earlier boosting results are presented separately.

Use the existing environment and authorized development artifacts. From the repository root:

```sh
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/garmin-overview-mpl .venv/bin/jupyter nbconvert --to notebook --execute notebooks/21_central_research_overview.ipynb --inplace --ExecutePreprocessor.timeout=300
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/garmin-overview-mpl .venv/bin/python -m garmin_central_overview.verify
```

Rerunning refreshes only this overview's aggregate outputs. Previous notebooks, immutable bundles, frozen assignments and the sealed final test remain unchanged. Full implementation: `notebooks/20_complete_pipeline_for_review.ipynb` and canonical `src/` modules.
