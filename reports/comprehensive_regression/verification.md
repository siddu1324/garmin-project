# Notebook 14 verification

Completed 2026-10-19. Current version: `4e769e522b239f43`.

The benchmark reuses the unchanged Notebook 13 feature bundle. It accounts for all **3,706**
development responses and models **3,689** valid outcomes from **125** development participants.
The frozen five participant folds are unchanged. No final-test participant sensor records or outcomes
were evaluated. No raw preprocessing, commit, push or provider communication occurred.

## Executed checks

| Check | Result |
|---|---|
| Full test suite | 190 passed; 13 dependency deprecation warnings |
| Ruff | Passed for `src` and `tests` |
| Saved outer-fold estimators | 490 prediction replays; maximum absolute discrepancy 0 |
| Independent deterministic fresh fits | 14 passed |
| Inner participant-group membership checks | 1,470 passed |
| Tuning | 2,930 candidate configurations, evaluated across three inner folds; no failed jobs or nonconverged candidates |
| Earlier fixed-Ridge comparisons | Four matched-cohort legacy replays; maximum absolute discrepancy 2.49e-14 |
| Notebook execution | All nine code cells executed without error; aggregate outputs only |
| Browser review | Nine sections, eight loaded figures; desktop/mobile review, 15 screenshots; no external requests, broken internal links or page overflow |
| Input preservation | Feature-bundle and frozen-assignment checksums unchanged; Notebook 13 completion seal unchanged |
| Prior work | 682 prior files and 597 prior private manifests preserved |
| Raw/reference integrity | 7,870 file size/mtime checks passed; raw contents were not reread |

The synthetic positive-control test checks whether the actual preprocessing/model machinery can
recover a deliberately injected relationship across held-out synthetic participants. Tests also cover
training-only imputation, constant/all-missing columns, selection, model interfaces, paired uncertainty,
strict JSON serialization and privacy-matcher equivalence. These checks address tested computational
failures; they cannot establish sensor validity or rule out every possible implementation error.

The known-token privacy scan covers public source, tests, configuration, documentation and notebook
text using the existing identifier-boundary rules and validated aggregate exceptions. It additionally
checks the new standalone HTML. Private data, fold memberships, estimators and individual predictions
remain ignored. The final scan count and artifact hashes are recorded in `completion.json`.
The sole intentional edit to an existing source file narrowly permits this explicitly requested
executed aggregate-only notebook in the existing privacy validator.

## Versioning and reproducibility

The preserved predecessor `b3eedd7230280409` encountered a strict-JSON metadata error when XGBoost's
default nonfinite parameter was serialized. This was not a model-fit failure. The corrected version
encodes that metadata explicitly without changing the fitted estimator or its inputs. All 430 complete
compatible jobs were copied only after input, configuration, package and fitting-code compatibility
checks; the 60 unpublished XGBoost jobs were rerun. See `supersession.json` and the private
`metadata_compatible_reuse.json`. The predecessor remains intact and is not counted as another experiment.

The current private run contains `ema_accounting.parquet` (**3,706 rows**) and
`all_predictions.parquet` (**352,938 rows**). The latter repeats responses across experiment views;
these are not independent observations. Each fold's estimator, selected settings, inner memberships,
training predictions and held-out predictions has its own immutable manifest. Final run/public
manifests and the presentation completion record protect the delivered checkpoint. Replays verify
existing results; changed experimental code or inputs require a new version.

## Interpretation limits

All primary comparisons use identical responses and folds. The 1,355-response complete-case analysis
uses separately matched training and validation cohorts; metrics from different cohorts are not
presented as direct improvements. Participant bootstrap intervals condition on saved out-of-fold
predictions. They do not incorporate retuning uncertainty or correct the many exploratory comparisons.

The 438 main candidates retain the measurement limitations documented in Notebook 13. The 240
provisional PulseOx candidates are confined to a sensitivity experiment. No unsupported HRV is added.
Training-only imputation fills model inputs, not targets or raw sensor recordings. Sparse/stale
recordings, acquisition-versus-receipt timing, quality annotations and survey recall boundaries remain
uncertain. This experiment cannot identify which of these causes the weak predictive results.
