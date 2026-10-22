# Notebook 18 — reproducible root-cause audit

Start with [the presentation](index.html), [summary for Will](summary_for_will.md),
[frozen specification](specification.md), [issue classification](evidence_classification.md)
and [unsent evidence request](missing_evidence_request.md).

The pinned current aggregate version is `c62e4b329f41c611`. Notebook 18 reads only its
aggregate tables and figures. Individual audit traces, cohort membership, gated boundaries,
model inputs, fitted estimators and predictions remain under ignored `private/root_cause_analysis/`.
`current.json` there records the immutable data/run paths; `completion.json` here pins checksums.

Reuse the existing environment, from the repository root:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_root_cause.run
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_root_cause.evaluate
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_root_cause.describe
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_root_cause.evidence
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m garmin_root_cause.verify
PYTHONPATH=src .venv/bin/python -m garmin_root_cause.report
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/18_root_cause_diagnostics.ipynb --inplace
.venv/bin/pytest -q
.venv/bin/ruff check src tests
PYTHONPATH=src .venv/bin/python -m garmin_root_cause.privacy
```

Set `MPLCONFIGDIR` to a writable temporary directory if needed. Verified data/fold jobs are
reused; no expensive raw normalization or previous model search is repeated. Source/config/spec
hashes determine versioned audit data and fold jobs. Completed public artifacts must verify rather
than be silently replaced; changes require a new version. Execution counts are recorded in completion.

The existing 438-feature bundle is reused only for validated inputs and quality diagnostics.
This experiment uses four 60-minute HR features plus supported step rate, not all 438 predictors.
No new real HRV, stress predictor, final-test analysis, provider message, commit or push.
