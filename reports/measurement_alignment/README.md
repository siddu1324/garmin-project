# Current Notebook 19 checkpoint

Start with [the self-contained presentation](index.html) or [Will's briefing](summary_for_will.md).
Current aggregate version: **e5359ea4cc4b4efa**. Use only that version for Notebook 19 findings.

- [Frozen specification](specification.md)
- [Exact source contract and implementation limitations](source_contract.md)
- [Provider request and decision table — draft only](provider_data_request.md)
- [Verification](verification.md)
- [Explicit draft supersession record](supersession.json)
- Fully executed source notebook: `notebooks/19_measurement_alignment.ipynb`
- Reusable modules: `src/garmin_measurement_alignment/`
- Ignored row-level diagnostics/provenance: `private/measurement_alignment/runs/e5359ea4cc4b4efa/`

This is Path A: real acquisition, prompt-delivery and exact-answer-version evidence is still absent.
No new real predictions were fit. The conditional aligned-feature importer and fixed logistic experiment
are implemented and synthetic-tested, not evidence of improved real prediction. Existing Notebook 17
chronological-history machinery remains blocked pending actual version availability and a reviewed
chronological fit-time schedule.

The diagnostic runner verifies cached corrected inputs and reuses this immutable version on rerun.
The report generator refuses to overwrite a sealed presentation. To change methods or incorporate
provider evidence, make an explicit new version, retain this checkpoint and record what it supersedes.
The earlier Notebook 19 draft folders are preserved but are not current results.
