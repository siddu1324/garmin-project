# Comprehensive physiological feature checkpoint

Open `index.html` locally: it is standalone and embeds all figures. The executed teaching notebook is
`notebooks/13_comprehensive_physio_features.ipynb`. Read `summary_for_will.md`, `original_feature_inventory.md`,
`feature_dictionary.md`, `quality_and_context_dictionary.md`, `verification.md` and `specification.md` for the methodological contract.

Current artifacts:

- Base extraction (preserved): `private/comprehensive_features/features/1bfbb247db489957/`.
- Current completed features: `private/comprehensive_features/features/eecf1da995c39b27/`.
  `development_features.parquet`: all 3,706 rows; 678 physiological candidate columns plus metadata/context/targets.
  `quality.parquet`: separate support/quality/reasons; `lineage.parquet`: original window/group slices.
- Current nested fits and private OOF predictions: `private/comprehensive_features/models/818ddc22d6f28fdc/`.
- Public aggregate results: `reports/comprehensive_features/818ddc22d6f28fdc/`.
- Separate interpretation diagnostics: `private/comprehensive_features/diagnostics/cb836f0ff4e13782/`.

The original workbook is `/Users/siddharthareddy/Downloads/TULIP_Features_Simple.xlsx`.
The feature-family inventory pins its SHA-256. No participant data are in that workbook.
The source sheet does not specify a single exact total of scalar features; its 54 rows contain compound
families and some unresolved windows/percentile definitions. Do not equate its conditional HRV expansion
with reliable HRV predictors. Existing IBI evidence still blocks that expansion.

## Reproduction

Use the existing `.venv` and pinned `uv.lock`; no new package is required.

```sh
PYTHONPATH=src .venv/bin/python -m garmin_comprehensive.augment private/comprehensive_features/features/1bfbb247db489957
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m garmin_comprehensive.run private/comprehensive_features/features/eecf1da995c39b27 --workers 4
PYTHONPATH=src .venv/bin/python -m garmin_comprehensive.diagnostics
PYTHONPATH=src .venv/bin/python -m garmin_comprehensive.evaluate
PYTHONPATH=src .venv/bin/python -m garmin_comprehensive.verify
PYTHONPATH=src .venv/bin/python -m garmin_comprehensive.report
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/13_comprehensive_physio_features.ipynb --inplace
```

Each feature version is immutable. A changed configuration/source creates a new version; existing bundles
are verified and reused. The base extraction is reproducible with `python -m garmin_comprehensive.extract`
from existing normalized development partitions. It reads HR/counter ledgers and PulseOx partitions only,
not original raw archives or final-test sensor/outcome records. The amendment augments cached summaries
with non-overlapping earlier HR references; it does not recalculate tens of millions of raw records.

Model jobs are independently checksum-verified and resumable. The executed model source is archived inside
the private model run. A scheduler correction removed one duplicate quality specification (15 cached entries
across tasks/folds); 375 unique fitted jobs and their predictions are unchanged. Current source generates
375 unique jobs directly; a fresh reproduction gets a new source-hash version. The original partial run,
started before locating the feature workbook, is preserved and superseded; it is not used as evidence.

All source/target row keys, participant identities, actual clocks, memberships, models and predictions are
private and ignored. Public files contain aggregates only. The explicitly authorized Notebook13 aggregate
output exception is narrowly added to the existing notebook privacy validator. No raw changes, final-test
evaluation, provider messages, commit or push.

The finalized aggregate version is checksum-sealed by its `manifest.json`; `completion.json` also pins the presentation and executed notebook. Rerunning report/evaluation commands verifies and reuses this published version. A future experiment must use a new report directory/version; it must not overwrite this checkpoint. `supersession.json` identifies the preserved partial runs that are not current evidence.
