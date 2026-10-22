# Notebook 13 verification

Completed 18 October 2026. This is a new development-only checkpoint; earlier experiments remain preserved.

| Check | Result |
| --- | --- |
| Full test suite | 175 passed; 13 dependency deprecation warnings, no failures |
| Ruff | Passed across `src` and `tests` |
| Notebook execution | All 11 code cells executed without errors; only aggregate readers and synthetic examples |
| Nested fitting | 375 unique outer jobs, 1,590 candidate configurations; no nonconverged candidates |
| Saved-estimator replay | All 375 reproduced saved probabilities exactly; maximum absolute error 0 |
| Independent fresh fits | Six deterministic fits reproduced stored probabilities within 1e−10 |
| Grouped inner memberships | 1,125 checks; disjoint participants and no outer-validation rows in inner training/selection |
| Training-only transformations | Saved medians/scaler means independently checked against corresponding training rows |
| Existing-feature agreement | 326,128 legacy HR/counter checks and 11,118 original 60-minute history checks passed |
| Timing and numerical tests | Independent Python datetime boundaries at all eight lookbacks; rejected-pair barriers, counter resets, missingness, constant values, rolling summaries, non-overlapping history and synthetic positive control passed |
| Raw/reference inputs and assignments | Match existing integrity baselines and frozen assignment hashes |
| Existing artifacts | 623 existing files and 138 previous private manifests unchanged; sole intended existing-source edit authorizes aggregate Notebook 13 output in the notebook privacy validator |
| Public privacy | Known participant identifiers/names scanned in public text and notebook outputs; final scan count is recorded in `completion.json` |
| Private output location | Matrices, lineages, memberships, estimators and individual predictions remain ignored under `private/` |
| HTML browser review | Nine sections, eight embedded figures, 15 desktop/mobile screenshots; no page overflow, broken embedded images, external requests or script errors |

The 3,706-row feature matrix has 678 physiological candidates plus 15 metadata/context/target columns. Separate quality and lineage tables have 148,240 and 118,592 rows. Private concatenated OOF predictions have 292,935 rows across tasks, comparator views and cohorts; these are repeated experiment views, **not** 292,935 independent responses. Primary comparisons use the same 3,689 responses from 125 participants; the original complete-case sensitivity uses 1,355 responses from 113 participants. Every fold's training/validation class support is in `818ddc22d6f28fdc/fold_metrics.json`.

A duplicate quality-control scheduler entry initially listed 390 jobs. It reused the same 15 cached jobs; there were always 375 unique fits. The scheduler/completeness count was corrected and regression-tested. Executed `models.py` and `run.py` are archived with their original hashes. No model prediction or fit decision changed. Empty-ledger handling in the separate quality report was also corrected before publication. The report loader now requires a single verified diagnostic bundle matching the current feature manifest.

Aggregate results and presentation files are checksum-sealed. `supersession.json` distinguishes the complete current run from preserved partial runs; `completion.json` pins the final notebook, source/configuration, and presentation. Evaluation/report reruns reuse and verify sealed outputs rather than overwrite them. Re-executing a notebook can change execution metadata, so preserve the delivered notebook and execute a copy for independent review.

These checks establish reproducible computation, not acquisition-time semantics, validated oxygen saturation, normal-beat status, continuous sensor coverage, causal associations or deployable prediction. Current data do not justify expanded real HRV. Calibration diagnostics remain poor. Participant-bootstrap intervals are conditional on fixed development predictions and do not account for the full history of exploratory analyses or multiple comparisons.

All 32 final-test participants remain sealed. No raw files, frozen folds or previous experiments were changed. No commit, push or provider messages occurred.
