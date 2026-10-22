# Notebook 08 verification — retrospective Morning experiment

Current [notebook](../../notebooks/08_morning_recall_window_comparison.ipynb),
[report](b1fd2e61d20cdbe4/report.md), [professor summary](b1fd2e61d20cdbe4/professor_summary.md), and
[frozen specification](../morning_recall_specification.md).

## Checks completed

- **125 tests passed**, including seven new synthetic tests for unchanged HR gates, sample SD,
  conflicting/duplicate timestamps, right-edge exclusion, later-minute sensitivity, partial bins,
  absent observations, pairwise/common intersections, explicit predictor projection, training-only
  scaling, validation-label perturbation, participant separation, shared infeasible-fold skipping,
  participant-averaged errors and deterministic cluster bootstrap.
- **Ruff passed** for `src`, `tests`, and Notebook 08; `git diff --check` passed. Existing environment
  and dependency lockfile were preserved. No extra package or model family was installed.
- **Notebook execution passed**, seven cells; source notebook remains output-free. The final ignored
  executed copy is `outputs/08_morning_recall_window_comparison.4c844f737b74ee62.final.executed.ipynb`. Only aggregates and labeled synthetic examples.
- **Feature/clock verification:** all 934 Morning rows retained per scenario, with unchanged targets,
  context, fixed features, identity and folds. Wake boundaries and cutoffs independently reconstructed
  using Python datetime arithmetic. All 4,464 distinct positive feature/quality windows checked against
  ledger Boolean masks and scalar statistics; fixed-window quality repeats exactly across scenarios.
  Undefined wake windows retain missing values and explicit reasons. No repaired dates or extra readings.
- **All 220 models independently replayed** (140 Ridge / 80 dummy fits) using training-only moments
  and Ridge normal equations. Maximum prediction discrepancy 1.21e-13.
  Exact matching training/validation key hashes, fold separation, constant Morning encoder and predictor
  allowlists checked. Wake clock/duration, quality, current answers, IDs and other-window features are excluded.
- **264 metric rows and 88 paired contrasts independently checked.** Bootstrap replay uses 2,000
  participant-cluster draws with fixed seed; all models in a contrast share OOF rows. All 40 cohort-fold
  fits are feasible; no skipped or duplicated OOF response within a comparison. Both wake scenarios have
  identical cohort memberships. Common equals the five-minute pairwise cohort; results are not independent
  replications. Missingness and counts remain explicit in all other accounting tables.
- **Corrected provenance verified:** current Notebook 04 feature/eligibility parents and Notebook 07
  boundaries, explicit supersession record, frozen assignments and all pre-fit hashes match. The guarded
  rerun reuses the same immutable bundle without overwriting it.
- **Raw/reference integrity passed:** hashes, sizes and modification times match the original inventory
  baseline. Fourteen previous standard-format private bundles verified; two legacy manifests without the
  newer complete flag retain exact artifact checksums and are checked as history only.
- **Existing work preserved:** 301 prior source/documentation files are byte-identical. Only README
  checkpoint links and the privacy helper's exact notebook/config allowlist additions change existing
  files. Previous uncommitted dependency/source work remains intact. Git index empty; no commit or push.
- **Privacy:** public known-ID/source-filename and credential/local-path scans, executed-notebook visible
  output scan, ignored-output checks and four-figure visual review are recorded in the accompanying JSON.
  All participant-level tables/audits remain private and ignored. No provider messages sent.

## Exact outputs

Private bundle: `private/morning_recall/runs/4c844f737b74ee62`.

| File | Rows / purpose |
| --- | --- |
| `accounting.parquet` | 1,868; all 934 Morning EMAs × two scenarios, with missing/exclusion reasons |
| `features.parquet` | 1,868; same rows with reused fixed HR, new wake HR, existing outcome/context |
| `quality.parquet` | 7,472; separate timing/support diagnostics, never predictors |
| `membership.parquet` | 7,472; all response memberships across two scenarios × four cohorts |
| `out_of_fold_predictions.parquet` | 1,684 matched comparison views, not independent observations |
| `lineage.parquet` | 1,868 wake source references, including unavailable boundaries |
| `training_audit.json` | 40 matched cohort-fold records containing 220 model audits |

Aggregate tables, four figures and narratives: `reports/morning_recall/b1fd2e61d20cdbe4`.
Reusable modules: `src/garmin_morning_recall/`; frozen configuration: `config/morning_recall.json`.

## Limits of verification

All 32 final-test participants remain sealed. This verifies computational consistency, not correct
acquisition timestamps, physiological validity or real-time deployability. Wake time is retrospective,
minute granularity is not a confidence bound, and same-day/same-offset assumptions remain provisional.
Only 24.3% of Morning responses enter the primary cohort. Freshness/occupied bins do not establish
continuous coverage. Conditional bootstrap intervals omit model-refitting uncertainty and multiplicity
adjustment; no optimal window, equivalence or absence of physiological signal is inferred.

PyArrow's sandbox CPU-cache probing notices did not prevent Parquet operations or any check.
