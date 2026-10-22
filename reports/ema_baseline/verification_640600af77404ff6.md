# Notebook 05 verification

Completed development-only outcome linkage and the frozen HR/activity baseline. All 32 final-test
participants remain sealed; only assignment identities and byte-level integrity hashing are touched
outside development. No final-test response/sensor transformations, tuning, imputation, binary targets,
expanded HRV, feature selection, commit or push.

## Outcome and provenance

- Verified all five stored response codes (1–5) against all three blank questionnaire PDFs. Display
  labels are 0–4; questionnaire suffixes _1/_4 correspond to unsuffixed raw export fields. Evidence
  records content hashes, pages and exact item names in the aggregate report folder.
- PDFs do not define exporter missing sentinels. All observed missing cells are empty strings; the
  conservative candidate-token exclusions are documented as provisional and do not change observed
  nonblank codes. All 47 missing item cells account for 17 incomplete outcomes; no nonmissing invalid
  item value is scored.
- Independently recomputed all **3,706** scores with scalar arithmetic and checked every original
  feature/metadata value against the corrected source. **3,689 valid outcomes**, 17 missing; all 125
  development participants and original folds remain linked through a one-to-one stable-key join.
- Feature, normalized and corrected eligibility manifests match the supersession record. The private
  pre-fit record binds the model specification, configuration and feature manifest before scoring or
  fitting. Frozen assignments match the original inventory and reference checksums.

## Modeling and independent numerical verification

- **32 matched comparisons**: eight lookbacks × two feature cohorts × native/common membership.
  All **160 fold/cohort combinations**, containing **880 individual model fits**, are feasible.
- Verified disjoint training/validation participants, original folds, complete-case membership and
  identical model rows within every comparison. Common membership is identical across all lookbacks
  for both training and validation. It is not merely a common validation subset.
- Explicit feature lists exclude identifiers, targets/current answers, quality, floors, Garmin stress,
  IBI/HRV, calories and intensity. Pipelines fit numerical scalers and category encoders on training
  rows only. Actual local start defines cyclic context. No imputation or prediction clipping.
- An independent implementation using training population moments, training categories and centered
  ridge normal equations reproduces **all 880 fits**, with maximum prediction discrepancy
  **8.71×10⁻¹⁴** score points. Mean and median dummy values are independently recomputed too.
- An exact repeat of the scikit-learn fits reproduces every saved prediction, training-transform audit,
  coefficient, cohort/fold table, pooled/fold metric and paired bootstrap interval. Bootstrap intervals
  use paired participant clusters and fixed predictions; they do not include refitting uncertainty.
- `coefficients.json` is an aggregate fold/model table. Numerical coefficients are score changes per
  training-fold SD conditional on the other inputs; one-hot coefficients depend on the full encoded
  category/intercept parameterization. Neither coefficient signs nor bootstrap intervals establish
  physiological mechanisms or causal effects. No coefficient was used for feature selection.

## Tests, execution and privacy

- **104 tests passed**, including **15 new synthetic tests** for coding/reversal/missingness, duplicate
  or inconsistent joins, final-test/fold rejection, cyclic midnight/seconds context, explicit predictor
  exclusions, held-out category/extreme-value handling, training-only scaling, outcome independence
  of held-out predictions, cohort intersections and paired bootstrap weighting/determinism.
- Ruff passes on `src tests notebooks/05_ema_outcome_baseline.ipynb`; `git diff --check` and offline
  lock verification pass. The existing environment is reused, with installed scikit-learn **1.9.1**
  added as a pinned direct dependency. Previous dependency changes are preserved.
- Notebook 05 executed **six code cells** successfully. The source is output-free; the executed copy
  is ignored. Jupyter required authorized local loopback ports outside the filesystem sandbox; no
  remote service was used. Displayed data are aggregate or explicitly synthetic.
- **167 public text artifacts**, including this verification record, pass known-identifier and private
  filename scanning. Executed notebook text also passes. Credential and local-home-path checks pass.
  Four figures contain aggregates only and were visually reviewed. Output ignore rules are checked.
- Raw/reference file hashes, sizes and mtimes match the original integrity baseline before and after
  computation and on the guarded rerun. **Nine prior immutable private bundles** pass full checksum
  verification, including retained historical versions (historical checksums do not authorize reuse).
- **158 pre-existing source/document/notebook files** remain byte-identical. Only README, the privacy
  allowlist, pyproject.toml and uv.lock changed among pre-existing files in this checkpoint. Other
  pre-existing working-tree changes are preserved. New work remains local and the Git index is empty.
- The old Notebook 02 unused-import lint issue remains pre-existing and unchanged; the current
  source/test suite and Notebook 05 pass. No old notebook/report or validated bundle was overwritten.

## Exact outputs

- `private/ema_baseline/runs/640600af77404ff6/linked_development.parquet` — **3,706 rows**.
- `private/ema_baseline/runs/640600af77404ff6/out_of_fold_predictions.parquet` — **46,843 rows**,
  each a cohort/EMA view with matched model predictions, not 46,843 independent EMAs.
- `private/ema_baseline/runs/640600af77404ff6/cohort_membership.parquet` — **118,592 inclusion rows**,
  preserving every EMA's membership for all 32 comparisons, including missing-outcome exclusions.
- `training_audit.json` in the same ignored bundle — training key hashes, scaler moments, categories,
  sample counts and dummy constants for every fit. No executable serialized model is created.
- [Aggregate tables, figures and report](d933d3999d949477/report.md).
- [Frozen specification](../ema_baseline_specification.md).
- `notebooks/05_ema_outcome_baseline.ipynb` — output-free source notebook.
- `outputs/05_ema_outcome_baseline.640600af77404ff6.executed.ipynb` — ignored executed notebook.

## Remaining limitations and next experiment

No useful incremental MAE gain appears for these fixed wearable models; the matched median dummy
has lower pooled MAE throughout. This does not establish absence of physiological signal. Complete-case
selection, sparse/unrepresentative recording, provisional timestamp/counter semantics, differing survey
reference periods and conditional development uncertainty limit interpretation. Two HR observations,
or even the slope minimum, do not establish representative window coverage.

Obtain provider clarification of timestamp/batching and counter attribution, then preregister a
measurement-support sensitivity experiment with externally justified density/gap conditions and the
same fixed matched models. Do not choose thresholds from score associations or open the final-test
cohort. The draft provider questions remain unsent.
