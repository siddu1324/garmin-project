# Garmin measurements preceding EMA stress reports

This checkpoint inventories corrected raw data and proposes the next research pipeline.
It does **not** calculate prediction features, HRV, PSS scores or models. Structural cohort
counts use all participants; exploratory summaries use only the frozen development split.

Use Python 3.11 and **uv** as the single environment system. Dependencies and transitive
versions are recorded in `pyproject.toml` and `uv.lock`.

```sh
uv sync --frozen
uv run pytest
uv run ruff check src tests
```

Raw archives belong in ignored `data/`. Configure other local locations using shell environment
variables (never commit local absolute paths or credentials):

```sh
export GARMIN_REFERENCE_PROJECT='/absolute/path/to/previous/sea-garmin'
# Optional; defaults to this repository's data directory:
export GARMIN_DATA_DIR='/absolute/path/to/local/raw-archives'
PYTHONPATH=src uv run python -m garmin_inventory.audit
```

The audit reads ZIP members without extraction, verifies the frozen 125/32 participant assignments
and existing folds, recalculates aggregate diagnostics, and hashes source/reference files before
and after. Expect several minutes and roughly 1–2 GB of available RAM. It never runs previous
project code or loads previous features/models. If the split is unavailable, exploration is skipped;
structural inventory continues. Unresolved identities stop exploratory inspection rather than
creating a new mapping. Errors omit source records and identifying filenames.

Open the notebook with the same environment:

```sh
uv run jupyter lab notebooks/00_data_inventory_and_understanding.ipynb
```

The notebook displays the aggregate reports by default. Set `GARMIN_RUN_FULL_AUDIT=1` before
launching Jupyter to recompute the complete raw audit. Keep the source notebook's outputs empty;
executed notebooks belong under ignored `outputs/`. To verify the aggregate walkthrough:

```sh
mkdir -p outputs
uv run jupyter nbconvert --to notebook --execute \
  notebooks/00_data_inventory_and_understanding.ipynb \
  --output 00_data_inventory_and_understanding.executed.ipynb --output-dir outputs \
  --ExecutePreprocessor.timeout=1800
```

Start with [the inventory report](reports/inventory_report.md),
[the data dictionary](reports/data_dictionary.md), and
[the proposed pipeline](reports/feature_pipeline_plan.md).
`reports/inventory.json` and `reports/data_dictionary.json` contain aggregate machine-readable
findings. Reusable code lives under `src/garmin_inventory/`; tests use synthetic records only.

Raw files, private manifests, exact split assignments, local configuration, participant-level
outputs, models and executed notebooks are ignored. The audit checks Git's tracked/index state
before writing; a sensitive tracked file requires a private review and is never automatically
untracked. Public artifacts are scanned against known identifier/filename tokens. Do not display
raw frames, participant filenames or free-text responses, including in notebook error traces.

The next step is raw-preserving preprocessing and a development-only IBI eligibility audit,
following the pipeline plan. Garmin stress and quality diagnostics remain excluded from the
initial predictor set; VLF-related features remain deferred.
