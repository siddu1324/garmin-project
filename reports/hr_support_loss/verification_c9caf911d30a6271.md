# Notebook 06 verification

Completed the prospectively specified exploratory follow-up informed by Notebook 05. This is not
independent preregistered confirmation or validated real-time prediction. No support/loss combination
search, alpha tuning, new activity feature, expanded HRV, imputation, nonlinear model, feature
selection, final-test evaluation, commit or push.

## Fixed specification and input preservation

- Config/specification and the Notebook 05 parent manifest were hashed in an ignored pre-comparison
  freeze record. A/B/C rules are unchanged: original complete cases; ≥80% accepted-minute occupancy;
  and additionally freshness ≤60 seconds. Inclusive integer-bin comparisons avoid threshold rounding.
- All **3,706 development EMAs**, **125 participants**, original **five participant-grouped folds** and
  existing targets/features are preserved. There are still **3,689 complete outcomes / 17 missing**.
  No score was recoded, imputed or dichotomized. The 32 final-test participants remain sealed.
- Current normalized/feature/Notebook 05 manifests and corrected eligibility provenance match the
  supersession record. Default guards reject obsolete window artifacts. No previous window threshold
  or feature value was altered.
- All **29,648 HR lookbacks** were independently verified from original accepted-reading ledgers:
  bin occupancy, accepted count, cutoff/left boundaries, freshness, timestamp span and maximum
  internal/edge gap. Search-based bin checks use Python integer offsets, not the original floor-based
  occupancy calculation. Bins count accepted timestamp groups, not rejected raw rows.

## Fair comparisons and numerical checks

- Saved-subset filtering and support-cohort refitting are separate stages. Saved native/common
  predictions remain exactly equal to the corresponding Notebook 05 predictions. Their original
  training populations are retained; refits use the new eligible training population.
- Independently checked all **177,888 membership rows** and **112,221 cohort/EMA prediction views**.
  All-EMA inclusion accounting, unchanged outcomes, unique comparison keys and original participant
  folds are verified. Model predictors use only the authorized HR/context fields, never quality.
- Reconstructed **1,200 support-refit models** with independent dummy arithmetic / centered Ridge
  normal equations. Maximum discrepancy is **8.71×10⁻¹⁴** score points. Policy A refits exactly
  reproduce Notebook 05. Every comparison uses identical validation and training rows across models.
- Median regression is restricted to unchanged A native/common cohorts. All **240 QuantileRegressor
  fits converged** without convergence warnings. Independent parameter replay checks prediction
  values, training-only scaler means/scales, training-only categories and penalized objective values.
  Each objective is no worse than the feasible constant-training-median solution.
- **234/240** quantile fits have constant validation predictions equal to the matching median dummy;
  six have varying predictions. No settings were changed to manufacture variability. Regularization
  differs across model families, so the experiment does not isolate the loss function alone.
- All **240 support-refit fold/cohort combinations** and **80 median-loss fold/cohort comparisons**
  were numerically feasible. The common B/C cohort (22 responses / seven participants) and native
  24-hour B/C cohorts (35/nine and 30/eight) remain explicitly small. Original folds were never redrawn.
- Exact full replay reproduces every fitted prediction, training audit, cohort/composition/quality
  table, metric, prediction-variability summary and participant-bootstrap interval. No validated
  bundle was overwritten. A guarded rerun returns the same immutable version.

## Tests, execution and privacy

- **112 tests passed**, including eight new synthetic tests for accepted-bin boundaries, duplicates,
  inclusive support thresholds, common intersections, participant separation, training-only encoding,
  legitimate constant quantile predictions, convergence warnings, empty/small/infeasible cohorts,
  preserved saved predictions when refitting is infeasible, and matched-fold exclusion after failure.
- Ruff passes on `src tests notebooks/06_hr_support_and_loss.ipynb`; `git diff --check` and offline
  lock verification pass. Existing package versions/environment are unchanged. Installed scikit-learn
  is **1.9.1**; no new dependency was installed. The historical Notebook 02 unused import remains
  unchanged; current source/tests/Notebook 06 pass.
- Notebook 06 executed **seven code cells** without errors. Source outputs are empty. Local Jupyter
  loopback ports required authorized execution outside the sandbox; no remote service was used.
- **208 public text artifacts**, including this verification, pass known-identifier/private-filename
  scanning. Executed notebook text also passes. Credential and local-home-path scans pass; four
  aggregate figures were visually reviewed. Private/output ignore rules were checked for all new
  dataset files and the executed notebook. No participant rows or identifiers are displayed.
- Raw/reference hashes, sizes and mtimes match the original integrity baseline before/after the
  pipeline and on rerun. **Ten prior immutable private bundles** pass full checksum verification.
  Historical checksum checks are preservation checks, not permission to use superseded windows.
- **188 pre-existing source/report/notebook/config files** remain byte-identical. Only README and
  the privacy allowlist changed among pre-existing files. Pre-existing pyproject/lock changes and all
  previous working-tree changes remain intact. Nothing is staged and the Git index is empty.

## Outputs

- `private/hr_support_loss/runs/c9caf911d30a6271/membership.parquet` — **177,888 rows**, every original
  EMA across three policies, two scopes and eight lookbacks, with inclusion/reference flags/reasons.
- `private/hr_support_loss/runs/c9caf911d30a6271/predictions.parquet` — **112,221 comparison views**,
  with matched predictions; overlapping cohort/EMA views are not independent observations.
- `private/hr_support_loss/runs/c9caf911d30a6271/verified_hr_support.parquet` — **29,648 HR windows**.
- The same ignored bundle stores fitted transform/coefficient audits and all aggregate tables.
- [Report, four figures and readable metric/composition appendices](838b9efbf4bfd731/report.md).
- [Frozen specification](../hr_support_loss_specification.md) and
  [provider-question draft](../hr_support_loss_provider_questions_draft.md) (not sent).
- `notebooks/06_hr_support_and_loss.ipynb` — source.
- `outputs/06_hr_support_and_loss.c9caf911d30a6271.executed.ipynb` — ignored executed copy.

## Limits and next experiment

Recording-support restrictions do not show a consistent HR gain. C at 12 hours has a −0.012 point
MAE difference versus matching median, with conditional interval [−0.141,+0.117], while excluding
91.9% of its A responses. Common B/C fits are too small for reliable window comparison. Median
regression usually improves over Ridge by reproducing the median dummy, with no added HR signal.
These are many exploratory comparisons; bootstrap intervals concern fixed development predictions,
not model-refitting uncertainty, external validation, causal effects or adjusted hypothesis tests.

Clarify acquisition/receipt timestamps, batching and the intended “since waking” / “since last prompt”
periods. If explicit, verified reference times are available, specify a recall-aligned development HR
experiment with unchanged fixed models and matched cohorts. Do not infer those times from completed
surveys or search thresholds/alphas to improve these results. No final-test opening is warranted.
