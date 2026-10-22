# Notebook 19 — verification and reproducibility

Current diagnostic version: **e5359ea4cc4b4efa**. The [supersession record](supersession.json) identifies
unvalidated Notebook 19 drafts; neither draft is current evidence. No prior checkpoint was replaced.

Run from the repository root using the existing `.venv` and `PYTHONPATH=src`:

```
.venv/bin/python -m garmin_measurement_alignment.run
.venv/bin/python -m garmin_measurement_alignment.verify
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/19_measurement_alignment.ipynb --inplace
.venv/bin/python -m garmin_measurement_alignment.privacy
```

The runner uses cached normalized development data and current corrected feature/ledger bundles.
It checks their manifests and rejects revoked window provenance. It does not repeat normalization,
fit real models, populate provider metadata, or touch final-test sensor/outcome records. A nonempty
future provider inbox stops the missing-evidence path and requires explicit reviewed import.

Independent verification checks all 3,706 actual/scheduled clock differences and precision-specific
cutoffs using Python datetime arithmetic; 5,436 scalar window boundaries including the old overflow
points and 24-hour edge; 29,648 zero-shift HR count/occupancy rows against the current feature quality
bundle; and 240 shifted-support cases using independent boolean masks rather than searchsorted.
The ±60 s scenarios never pull stored post-cutoff readings backwards into prediction time.

All 125 development participant/fold assignments agree with the frozen registry. Every EMA has a
private blocked-reason row. Prior Notebook 18's completed seal is verified. Preservation checks cover
969 prior files, 3,245 prior private manifests, and size/mtime metadata for 7,870 raw/reference files.
Input bundle checksums are checked in full; raw/reference size/mtime verification is not a new complete
rehash of all raw files. New exact-source discovery reads archive structure/headers, not final-test
measurements. No new real predictive estimator or prediction output is produced.

Synthetic tests exercise approved/unapproved hashes; successful import and immutable reuse; timezone
conversion and precision; wrong participant/EMA joins; event retries/conflicts; delayed latest prompt
without fallback; current-prompt rule; independent waking date; incomplete log coverage; DURING/POST
and late-receipt exclusion; counter resets/rejected/missing predecessors; conflicting HR; exact-version
history without future/current labels or later edits; incomplete latest answers without backfilling;
and held-out-participant injected-signal recovery on identical model cohorts. Tests do not establish
physiological validity. Synthetic outputs are not reported as real findings.

The new importer keeps nested original JSON in memory; partition/stream handling should be reviewed for
large future exports. Its precision bounds assume source-defined truncation, not rounding or drift.
Metadata approval requires provider evidence, not just a syntactically valid file. Future history
features are version-aware, but activating real chronological models still requires a reviewed fit-time
schedule. These limitations remain explicit rather than silently supplying missing clocks.

The final `completion.json` records actual test, notebook, privacy and browser results, artifact hashes
and environment. Public outputs contain aggregates and synthetic timelines only. Participant rows,
source filenames and models remain private/ignored. No provider messages, commit or push.
