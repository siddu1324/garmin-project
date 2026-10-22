# Classification checkpoint verification

- **157 tests passed**, including synthetic class-boundary/missing-target checks, grouped separation,
  training-only imputation/scaling/encoding, all-missing training columns, class weights, metric arithmetic,
  bootstrap pairing, both boosting libraries' binary/multiclass probability ordering and held-out synthetic
  participant signal recovery. Existing preprocessing/window/feature/regression tests also pass.
- **Ruff passed** for `src` and `tests`; Git whitespace checks passed. The only test warnings are
  existing matplotlib/pyparsing deprecations. Sandboxed Arrow CPU-detection warnings did not affect computation.
- **Notebook 12 executed:** 15 code cells, zero execution errors. Outputs are aggregate text/tables,
  eight figures, the research abstract and synthetic examples. No participant-level row or timestamp is displayed.
- **115/115 fold jobs completed**, zero skips. Every selected configuration matches the maximum frozen
  inner-fold mean macro F1; all comparison membership checks passed. No final-test participant was evaluated.
- **2,130 estimator replays agree** with saved OOF probabilities. Fitted outer medians, scaler parameters
  and survey encodings exactly match their training cohorts. No validation data are used to learn them.
- **Input integrity:** unchanged raw/reference snapshot, frozen assignment checksum and corrected feature,
  outcome/history/eligibility bundle hashes. Prior artifacts remain intact: 574 public/source files and
  22 private bundle manifests were checked against the initial snapshot. README, dependency files and
  the narrow Notebook 12 aggregate-output privacy exception are the intentional existing-file edits.
- **Privacy:** known participant identifiers and private source filenames are scanned across public
  source, reports and notebooks, including executed Notebook 12 outputs. Images were generated solely
  from aggregate tables; confusion, lookback and importance plots were visually reviewed. Private inputs,
  membership, fitted models and OOF predictions remain under ignored `private/pss_classification/`.
- **Reproducibility:** training-source/configuration/input hashes were frozen before fitting. Every fold
  is an immutable checksummed bundle. Candidate logs and fitted estimators are retained privately; full
  aggregate metrics, all paired comparisons, selected settings and search configuration are public.

The tests validate implementation and numerical reproducibility, not physiological timestamp semantics,
clinical cutoffs, sensor representativeness or external generalization. Bootstrap intervals condition on
this fixed exploratory workflow and are not adjusted for multiple comparisons. No model is deployed,
no final-test labels are opened, and no commit, push or provider message is made.

Machine-readable split/preprocessing/replay results: [verification.json](verification.json).
The completion record includes the final public privacy-scan count and artifact hashes.
