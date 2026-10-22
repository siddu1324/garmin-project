# Notebook 17: temporal and personalized stress investigation

Start with `index.html` or `summary_for_will.md`. The specification precedes new performance.
`data_quality_audit.md` explains the evidence gate and the precise missing provider fields.
No real EMA-history model is presented as online prediction without verified answer availability.

Current immutable feature bundle: `private/temporal_personalization/features/8dcdcaaace493f04/`.
Current experiment: `private/temporal_personalization/runs/a6d6ebf8263d0367/`.
The 3,706-row `development.parquet` retains the original keys/folds plus new summaries, quality and
explicit missing history fields. Model inputs are strict allowlists, not every column in this table.
`ema_history_lineage.parquet` is intentionally empty: no fabricated answer availability.
Participant metrics, predictions, fitted estimators and memberships are private/ignored.

Public aggregate results are in `reports/temporal_personalization/a6d6ebf8263d0367/` and `audit/`.
The main comparison table contains explicit blocked rows for previous-EMA and personalized-label models.
Earlier wearable references and label-history personalization are distinct and labeled accordingly.

Reproduce in the pinned environment:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_temporal_personalization.audit
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_temporal_personalization.data
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m garmin_temporal_personalization.run --workers 4
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_temporal_personalization.evaluate
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_temporal_personalization.diagnostics
PYTHONPATH=src .venv/bin/python -m garmin_temporal_personalization.sequence_diagnostic
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_temporal_personalization.verify
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/garmin-temporal-mpl .venv/bin/python -m garmin_temporal_personalization.report
PYTHONPATH=src .venv/bin/python -m garmin_temporal_personalization.privacy
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests
```

Final artifacts are sealed and never overwritten; completed jobs are hash-checked. Copy the completed
notebook to an ignored directory for later execution checks. No final-test access, commit or push.
