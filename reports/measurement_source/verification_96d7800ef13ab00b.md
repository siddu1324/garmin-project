# Notebook 11 verification

Current notebook: `notebooks/11_measurement_source_and_timestamp_audit.ipynb`.
Current immutable evidence bundle: `private/measurement_source/runs/96d7800ef13ab00b`.
Current aggregate report: `reports/measurement_source/1c44cb036f4ba779`.
Executed copy (ignored):
`outputs/11_measurement_source_and_timestamp_audit.96d7800ef13ab00b.executed.ipynb`.

- Full test suite: **146 passed**; three focused evidence/duplicate tests also passed after the
  final audit-versioning adjustment. The actual audit rerun returned the same immutable bundle.
- Ruff: **passed for all source, tests and Notebook 11**. A broader scan of all notebooks found
  one pre-existing F401 unused `Markdown` import in Notebook 02; that prior notebook is preserved.
- Notebook 11: 5 code cells executed, zero errors; source notebook has no outputs. Only saved
  aggregate evidence and synthetic rows are displayed. No real sensor/EMA values are loaded.
- Privacy: 500 existing/new public source, notebook, configuration and report files passed the
  known-identifier/name scan. Executed notebook text passed; final verification files were checked
  separately after writing. Private evidence/execution paths are ignored; no identifying manifest
  was copied into the public report. No new binary figures were produced.
- Preservation: 533 prior source/public files unchanged; the original README prefix is byte-identical,
  and prior privacy logic is identical after removing the two new allowlist names. All **19 prior
  private manifests and complete artifact checksum sets** match the pre-work snapshot.
- Raw/reference file hashes, sizes and modification times match the inventory baseline. Frozen
  assignment bytes and five folds are unchanged: 125 development / 32 sealed final-test participants.
- Corrected bundle manifests pass the current-window provenance guard; known-invalid window
  versions remain rejected. Earlier Notebook 02 citations are restricted to unaffected stream and
  immediate-window diagnostics. No eligibility or feature bundle is regenerated.
- Local Markdown report links resolve. Git whitespace check passed; zero staged paths. Earlier
  audit draft builds and execution copies are retained; README and Notebook 11 point to the version
  above. The initial audit-ID sensitivity to the addition of its own notebook was removed; no
  measurement calculations or old checkpoint artifacts were affected.

There are **zero new model fits, real HRV calculations, sensor/EMA-value reads, provider messages,
commits or pushes**. The source scan found no collection application/build candidate; a separate
metadata-only scan also found no app/SDK binary or FIT file candidate in the bounded roots.

Recheck the computational checks with:

```sh
.venv/bin/pytest -q
.venv/bin/ruff check src tests notebooks/11_measurement_source_and_timestamp_audit.ipynb
PYTHONPATH=src .venv/bin/python -m garmin_measurement_source.audit \
  --reference "$GARMIN_REFERENCE_PROJECT" --documentation-root "$GARMIN_DOCUMENTATION_ROOT"
.venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/11_measurement_source_and_timestamp_audit.ipynb \
  --output 11_measurement_source_and_timestamp_audit.recheck.ipynb --output-dir outputs
```

Remaining limitations: no release-specific collection contract or collector/exporter code, no
separate observed acquisition/receipt clocks, no original batch indices/beat-quality annotations,
and no complete prompt-event history. Bounded absence is not proof these never existed upstream.
Raw preservation, synthetic tests and numerical package agreement cannot resolve these semantics.
The provider request is drafted only; no new physiological validity or real-time prediction claim.
