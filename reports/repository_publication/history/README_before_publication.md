# Garmin measurements preceding EMA stress reports

**Current checkpoint: [Notebook 08](notebooks/08_morning_recall_window_comparison.ipynb)** —
[Retrospective Morning report](reports/morning_recall/b1fd2e61d20cdbe4/report.md), [professor summary](reports/morning_recall/b1fd2e61d20cdbe4/professor_summary.md),
[frozen specification](reports/morning_recall_specification.md), and
[verification](reports/morning_recall/verification_4c844f737b74ee62.md).
On the primary matched 227-response / 86-participant cohort, wake and 60-minute context+HR models
have OOF MAE 2.2506 and 2.2510: difference −0.0003 (conditional 95% interval −0.0275 to +0.0263).
Neither improves on matching median/context baselines by point estimate. Secondary windows, common
cohort and later-minute sensitivity do not establish a wake advantage; no optimal window is selected.
Wake reports define retrospective intervals only and never enter predictors. All 934 Morning EMAs
remain accounted for; all 32 final-test participants remain sealed. No commit or push.

Ignored immutable bundle: `private/morning_recall/runs/4c844f737b74ee62` (1,868 feature/accounting rows per table,
7,472 quality/membership rows per table, 1,684 OOF comparison views). Reproduce with
`PYTHONPATH=src uv run python -m garmin_morning_recall.pipeline` after setting
`GARMIN_REFERENCE_PROJECT`. Run Notebook 08 for the aggregate/synthetic walkthrough. Prior artifacts,
raw files, assignments and five folds are preserved; provider timing/quality questions remain unresolved.

**Previous checkpoint: [Notebook 07](notebooks/07_recall_alignment_and_model_diagnostics.ipynb)** —
[Recall/model diagnostics](reports/recall_alignment/824551e40b8e301d/report.md), [professor briefing](reports/recall_alignment/824551e40b8e301d/professor_briefing.md),
[exact local questionnaire citations](reports/recall_alignment/824551e40b8e301d/questionnaire_evidence.md), and
[frozen specification](reports/recall_alignment_specification.md).
[Verification](reports/recall_alignment/verification_0b217d08cb1c3dce.md) records the completed checks.
Morning refers to waking; Midday/Endday refer to the last prompt. Only retrospective Morning
boundaries are supported: 836 nominal windows, with accepted HR in 253 responses / 91 participants.
Prompt logs are unavailable. In 45 training-only fits, HR-only median regression stays constant
even at alpha=0. Tiny context+HR training improvements do not establish generalization.
No new validation predictions, final-test access, changed features, commit or push.

Current ignored bundle: `private/recall_diagnostics/runs/0b217d08cb1c3dce` (7,412 boundary rows, 1,662 candidate views,
50,220 training-fit prediction rows). All 3,706 development EMAs and original folds are preserved.
Reproduce with `PYTHONPATH=src uv run python -m garmin_recall_diagnostics.pipeline` after setting
`GARMIN_REFERENCE_PROJECT`; run Notebook 07 for the aggregate walkthrough. Previous artifacts remain
immutable. Provider questions are an unsent draft; real-time timing validity remains unresolved.

**Previous checkpoint: [Notebook 06](notebooks/06_hr_support_and_loss.ipynb)** —
[HR support/loss report](reports/hr_support_loss/838b9efbf4bfd731/report.md) and
[frozen exploratory specification](reports/hr_support_loss_specification.md).
[Verification](reports/hr_support_loss/verification_c9caf911d30a6271.md) records the completed checks.
This follow-up was specified after inspecting Notebook 05. Fixed 80% occupancy / 60-second
freshness policies do not reveal a consistent HR gain, and median regression mostly reproduces
the training-median dummy. B/C common cohorts contain only 22 responses from seven participants.
Timing/recall semantics remain unresolved; no optimal policy/window or real-time validity is claimed.
All 32 final-test participants remain sealed. No imputation, expanded HRV, new activity features,
tuning, feature selection, commit or push. Provider questions remain a local unsent draft.

Reproduce with `PYTHONPATH=src uv run python -m garmin_hr_support_loss.pipeline` after setting
`GARMIN_REFERENCE_PROJECT`. Immutable bundles are verified/reused; participant-level outputs are ignored.

**Previous checkpoint: [Notebook 05](notebooks/05_ema_outcome_baseline.ipynb)** and the
[development outcome/baseline report](reports/ema_baseline/d933d3999d949477/report.md).
[Verification](reports/ema_baseline/verification_640600af77404ff6.md) records the completed checks.
All 3,706 development EMAs and original folds remain linked; 3,689 have complete three-item
perceived-stress composites and 17 remain missing. Fixed participant-fold dummy/Ridge comparisons
show no useful incremental MAE gain from these HR/activity summaries. This is exploratory prediction
from recorded pre-EMA data, not validated real-time prediction. All 32 final-test participants remain
sealed. No tuning, imputation, binary targets, feature selection or expanded HRV was performed.

The [baseline specification](reports/ema_baseline_specification.md) was frozen before fitting.
Notebook 04's [feature dictionary](reports/hr_activity_feature_dictionary.md) and
[feature report](reports/hr_activity/7fdba29ad8d6f0fe/report.md) remain the unchanged measurement inputs.
**Old 12-/24-hour IBI results in Notebooks 01/02 are obsolete.** Current loaders reject those bundles.
See the [supersession record](reports/window_supersession.json),
[corrected eligibility report](reports/ibi_eligibility/974e98729579967b_8ac2c64795c6/report.md), and
[corrected dependent diagnostics](reports/hrv_input_validation/d0e494f0d2cbeb42/report.md).
Notebook 03's limited pilot remains unchanged. All participant-level outputs remain ignored.

Reproduce the authorized fixed baseline with `PYTHONPATH=src uv run python -m garmin_ema_baseline.pipeline`
after setting `GARMIN_REFERENCE_PROJECT` below. Existing immutable bundles are verified and reused.
The source notebook displays aggregates and synthetic examples; its executed copy belongs in `outputs/`.

Use Python 3.11 and **uv** as the single environment system. Dependencies and transitive
versions are recorded in `pyproject.toml` and `uv.lock`.

```sh
uv sync --frozen
uv run pytest
uv run ruff check src tests
```

Raw archives belong in ignored `data/`. Configure other local locations using shell environment
variables (never commit local absolute paths or credentials):

```sh
export GARMIN_REFERENCE_PROJECT='/absolute/path/to/previous/sea-garmin'
# Optional; defaults to this repository's data directory:
export GARMIN_DATA_DIR='/absolute/path/to/local/raw-archives'
PYTHONPATH=src uv run python -m garmin_inventory.audit
```

The audit reads ZIP members without extraction, verifies the frozen 125/32 participant assignments
and existing folds, recalculates aggregate diagnostics, and hashes source/reference files before
and after. Expect several minutes and roughly 1–2 GB of available RAM. It never runs previous
project code or loads previous features/models. If the split is unavailable, exploration is skipped;
structural inventory continues. Unresolved identities stop exploratory inspection rather than
creating a new mapping. Errors omit source records and identifying filenames.

Open the notebook with the same environment:

```sh
uv run jupyter lab notebooks/00_data_inventory_and_understanding.ipynb
```

The notebook displays the aggregate reports by default. Set `GARMIN_RUN_FULL_AUDIT=1` before
launching Jupyter to recompute the complete raw audit. Keep the source notebook's outputs empty;
executed notebooks belong under ignored `outputs/`. To verify the aggregate walkthrough:

```sh
mkdir -p outputs
uv run jupyter nbconvert --to notebook --execute \
  notebooks/00_data_inventory_and_understanding.ipynb \
  --output 00_data_inventory_and_understanding.executed.ipynb --output-dir outputs \
  --ExecutePreprocessor.timeout=1800
```

Start with [the inventory report](reports/inventory_report.md),
[the data dictionary](reports/data_dictionary.md), and
[the proposed pipeline](reports/feature_pipeline_plan.md).
`reports/inventory.json` and `reports/data_dictionary.json` contain aggregate machine-readable
findings. Reusable code lives under `src/garmin_inventory/`; tests use synthetic records only.

Raw files, private manifests, exact split assignments, local configuration, participant-level
outputs, models and executed notebooks are ignored. The audit checks Git's tracked/index state
before writing; a sensitive tracked file requires a private review and is never automatically
untracked. Public artifacts are scanned against known identifier/filename tokens. Do not display
raw frames, participant filenames or free-text responses, including in notebook error traces.

## Historical checkpoints (preserved for provenance)

The historical [development preprocessing and eligibility report](reports/ibi_eligibility/6747223eb50269ba_90911a7ae670/report.md)
describes the raw-preserving datasets, timing evidence, coverage and attrition. The
[verification report](reports/ibi_eligibility/verification_0c0d9eb75871.md) records the completed
tests, notebook execution, deterministic replay, input-integrity and privacy checks. The
[policy specification](reports/ibi_eligibility_policies.md) and `config/ibi_eligibility.json`
fix three provisional alternatives before retention comparison, each evaluated under interval-start
and interval-end timestamp hypotheses. None certifies normal beats or acquisition semantics.

```sh
PYTHONPATH=src uv run python -m garmin_preprocessing.audit
PYTHONPATH=src uv run python -m garmin_preprocessing.verify
uv run jupyter lab notebooks/01_preprocessing_and_ibi_eligibility.ipynb
```

Both preprocessing commands require `GARMIN_REFERENCE_PROJECT`; pass `--data "$GARMIN_DATA_DIR"`
if using a nondefault raw directory. Normalization reads sensor CSVs in 100,000-row chunks and
processes bounded participant/sensor partitions. Existing complete, content-addressed datasets
are hash-verified and reused; changed code/configuration creates a new version. It does not
overwrite prior validated artifacts. The current bundles occupy approximately 4.5 GB.
Final-test sensor sources are identity-gated before CSV parsing; final-test outcomes are never
transformed or analyzed. All 3,706 development EMAs remain accounted for across every policy.

Ignored datasets are under `private/preprocessing/normalized/31e51c934bcdeffa/` and
`private/preprocessing/eligibility/6747223eb50269ba/`. Source hashes, row numbers, participant
mapping provenance and original folds remain private. Physical sensor rows preserve originals;
the logical view collapses exact duplicates only, retaining references to physical rows.
Window diagnostics carry every eligibility status and rejection reason, with normal-beat status
explicitly unverified. Reports contain aggregates only; a schema-validated numerical window
count is distinguished from identifier text during privacy scanning.

Notebook 01 reads the pinned aggregate report by default. Set `GARMIN_RUN_PREPROCESSING=1`
to run the reusable pipeline first. Execute into a new ignored filename:

```sh
uv run jupyter nbconvert --to notebook --execute \
  notebooks/01_preprocessing_and_ibi_eligibility.ipynb \
  --output 01_preprocessing_and_ibi_eligibility.6747223eb50269ba.executed.ipynb \
  --output-dir outputs --ExecutePreprocessor.timeout=1800
```

The completed [HRV input specification](reports/hrv_input_specification.md) and
[rejection/continuity report](reports/hrv_input_validation/30a0bbe188d56b40/report.md) explain the
severe eligibility losses and validate package arithmetic using synthetic intervals only.
[Notebook 02](notebooks/02_hrv_input_validation.ipynb) displays aggregates and reruns the synthetic
validation. NeuroKit2 0.2.13 and HeartPy 1.2.7 are pinned in the same uv environment. Explicit
SDNN variance and pNN50 denominator conversions reconcile their different conventions.
The [verification record](reports/hrv_input_validation/verification_22dceec9fe90.md) documents
57 passing tests, notebook execution, adjacency/accounting checks, privacy scans and reproducible reruns.

```sh
PYTHONPATH=src uv run python -m garmin_hrv_input.diagnostics
PYTHONPATH=src uv run python -m garmin_hrv_input.report \
  --diagnostics private/hrv_input_validation/diagnostics/bc13763b35972be4
PYTHONPATH=src uv run python -m garmin_hrv_input.verify \
  --diagnostics private/hrv_input_validation/diagnostics/bc13763b35972be4
uv run jupyter lab notebooks/02_hrv_input_validation.ipynb
```

Diagnostics and verification require the configured reference location. They reuse the approved
normalized/eligibility bundles directly. Do not rerun normalization merely because adding package
dependencies changed the environment lock hash. `config/hrv_input_diagnostics.json` contains
separate labeled sensitivities; the frozen eligibility policies are unchanged.

The new ignored bundle has an index for every approved window and materialized immediate-window
records/run boundaries. Other windows hydrate on demand from immutable logical records. Original
timestamps, interval values, quality flags and rejected rows remain available; no new adjacency
is created by deletion. An observation window, supported duration and uninterrupted run are
different quantities. A two-minute Screen run is not five-minute HRV.

```sh
uv run jupyter nbconvert --to notebook --execute \
  notebooks/02_hrv_input_validation.ipynb \
  --output 02_hrv_input_validation.30a0bbe188d56b40.executed.ipynb \
  --output-dir outputs --ExecutePreprocessor.timeout=1800
```

The next proposed step is a small development-only pilot of mean/median IBI, sample SDNN, RMSSD
and pair-denominator pNN50 on unique Screen-eligible runs with at least 240 seconds of both span
and supported duration, after documenting source interpretation. No real pilot is calculated here.
Keep actual duration and unverified normal-beat status explicit. Spectral and additional nonlinear
features remain deferred; Garmin stress derivatives and quality diagnostics remain outside the
initial predictors. No PSS targets, imputation, final feature linkage or model is produced.

## Notebook 03: bounded real-data IBI variability pilot

[Notebook 03](notebooks/03_ibi_variability_pilot.ipynb), the [prespecified protocol](reports/ibi_pilot_protocol.md),
and the [aggregate pilot report](reports/ibi_pilot/78af22e7f0ead75b/report.md) describe the completed
20-run calculation. Both timestamp span and supported duration must be ≥240 seconds within a
Screen-eligible observation window. Mean/median IBI, sample SDNN, RMSSD and valid-pair pNN50 are
labeled **IBI variability—normal-beat status unverified**, not complete five-minute HRV.

**Long-lookback correction:** Notebook 03 caught int16 overflow in the older window generator
from backward step 547. Old 12/24-hour counts in Notebooks 01/02 are historical and must not be
used as valid pre-EMA evidence. The new immutable pilot index corrects window locations and
re-evaluates the unchanged Screen policy. Immediate through six-hour results are unaffected.
The generator now uses int64; no EMA cutoff or sensor timestamp was shifted. Prior bundles
remain intact. Notebook 04 now completes the full three-policy reissue; use the current links at the top of this README.

The selection phase is outcome-free and publishes its manifest before the separate guarded
calculation phase. Existing versions are verified and reused; participant-level outputs stay ignored.

```sh
PYTHONPATH=src uv run python -m garmin_ibi_pilot.pilot select
PYTHONPATH=src uv run python -m garmin_ibi_pilot.pilot calculate \
  --selection "$PWD/private/ibi_pilot/selection/52eb55d251e1d5e1" \
  --execute-authorized-pilot
uv run jupyter nbconvert --to notebook --execute notebooks/03_ibi_variability_pilot.ipynb \
  --output 03_ibi_variability_pilot.executed.ipynb --output-dir outputs
uv run ruff check src tests notebooks/03_ibi_variability_pilot.ipynb
```

The original Notebook 02 synthetic-only adapters remain unchanged. No further real-data
calculation beyond this fixed small pilot, no final-test exploration, and no model is authorized
by these commands. Provider clarification is recommended before expanding physiological HRV use.

[Notebook 03 verification](reports/ibi_pilot/verification_434dc451641c.md) records the checks and remaining limitations.

## Notebook 04: current HR/activity checkpoint

The [prespecified rules](reports/hr_activity_feature_specification.md) and
[feature dictionary](reports/hr_activity/7fdba29ad8d6f0fe/feature_dictionary.json) define all 104
measurement columns, the 88-column provisional predictor list, units and feature-specific support.
Calorie/intensity fields are missing and excluded; metadata, gaps, freshness and quality are not
predictors. The [provider-question list](reports/provider_questions_draft.md) is a local draft only.

Current ignored datasets:

- `private/hr_activity/features/76f5885b3955d5b0/features.parquet`: 3,706 rows.
- `private/hr_activity/features/76f5885b3955d5b0/quality.parquet`: 148,240 EMA/sensor/lookback rows.
- `private/hr_activity/features/76f5885b3955d5b0/window_lineage.parquet`: 148,240 source/window references.
- The same bundle contains every HR/counter reading disposition and original counter-pair boundaries.
- Corrected IBI windows: `private/preprocessing/eligibility/974e98729579967b/`.
- Corrected dependent index: `private/hrv_input_validation/diagnostics/55eb9ba094e893b7/`.

```sh
PYTHONPATH=src uv run python -m garmin_hr_activity.reissue
PYTHONPATH=src uv run python -m garmin_hrv_input.diagnostics
PYTHONPATH=src uv run python -m garmin_hr_activity.pipeline
PYTHONPATH=src uv run python -m garmin_hr_activity.verify \
  --reference "$GARMIN_REFERENCE_PROJECT" \
  --bundle "$PWD/private/hr_activity/features/76f5885b3955d5b0"
uv run jupyter nbconvert --to notebook --execute notebooks/04_hr_activity_features.ipynb \
  --output 04_hr_activity_features.executed.ipynb --output-dir outputs
uv run ruff check src tests notebooks/04_hr_activity_features.ipynb
```

Reissue/feature commands reuse immutable normalized data; they do not rerun raw preprocessing.
`verify_bundle` and window loaders reject known-invalid manifests/versions even after renaming.
`verify_archive_bundle` is checksum-only historical inspection, used solely to compare/preserve
old versions. The new `garmin_preprocessing.access.load_eligibility_windows` accepts corrected
window provenance; the old Notebook 02 index is never silently redirected or repaired in place.
The explicit supersession record includes coverage, attrition and continuity changes. Full
shorter-lookback results, immediate diagnostics and synthetic validation remain unchanged.

The earlier executable commands above are archival walkthroughs, not the current pipeline.
In particular, historical pilot commands that reference revoked dependencies now fail closed.
Use Notebook 04/current loaders for future work. No new real-data HRV is authorized here.

[Notebook 04 verification](reports/hr_activity/verification_25de581b4c9d.md) records checks, exact paths and limitations.


## Notebook 09 — current window-information audit and bounded nested benchmark

[Teaching notebook](notebooks/09_window_information_and_nonlinear_benchmark.ipynb),
[frozen experiment specification](reports/window_benchmark_specification.md),
[report](reports/window_benchmark/e3e34a2ea014f38f/report.md),
[professor summary](reports/window_benchmark/e3e34a2ea014f38f/professor_summary.md).
Wake and 60-minute windows use identical source readings in 116/227 matched Morning responses;
latest readings match in all 227. Independent tracing and a synthetic positive control pass.
On the fixed 60-minute development cohort (1,634 responses / 118 participants), nested HGB
context+HR MAE is 2.081 versus matching median 2.071; no clear wearable advantage is established.
Supported steps do not establish additional value on their separately matched cohort. This is
exploratory, not validated real-time prediction or an optimal-window selection.

The immutable private bundle is `private/window_benchmark/runs/22397f30f1439325`;
current aggregate report is `reports/window_benchmark/e3e34a2ea014f38f`.
Reproduce without overwriting existing bundles:

```sh
PYTHONPATH=src .venv/bin/python -m garmin_window_benchmark.pipeline --reference "$GARMIN_REFERENCE_PROJECT"
```

The pipeline verifies the frozen specification, assignments, corrected inputs and raw integrity.
Participant-level membership, source-overlap diagnostics and predictions remain ignored. Earlier
bundles and all 32 sealed final-test participants are preserved. No commit or push was performed.

[Notebook 09 verification](reports/window_benchmark/verification_22397f30f1439325.md) records tests, notebook execution, source tracing and privacy/integrity checks.


## Notebook 10 — within-person diagnostics and earlier-recording HR reference

[Teaching notebook](notebooks/10_within_person_signal_and_past_hr_reference.ipynb),
[frozen specification](reports/past_hr_reference_specification.md),
[report](reports/past_hr_reference/d021427887616868/report.md),
[professor summary](reports/past_hr_reference/d021427887616868/professor_summary.md),
[feature status](reports/past_hr_reference/d021427887616868/feature_status.md),
[provider request draft — not sent](reports/past_hr_reference/d021427887616868/provider_request_draft.md).

About 53.5% of development composite variation is within participants. Full-record means are
retrospective diagnostics only; predictive inputs use no personal outcome history. The fixed
reference [cutoff−25h,cutoff−1h) averages observed UTC-minute HR means and requires 30 occupied
minutes across three UTC-hour bins. It retains 1,575/1,634 existing 60-minute HR responses / 116
people. Reference/deviation models do not improve on matched raw HR or reference-only models.
HGB E MAE is 2.0949 versus raw-HR C 2.0646 and matching median 2.0673. No automatic expansion.

Ignored immutable bundle: `private/past_hr_reference/runs/510f1ad96a9e46f3` (all 3,706 responses
accounted for, 1,575 OOF predictions). Aggregate report: `reports/past_hr_reference/d021427887616868`.

```sh
PYTHONPATH=src .venv/bin/python -m garmin_past_hr_reference.pipeline --reference "$GARMIN_REFERENCE_PROJECT"
```

Timing, coverage and physiological interpretation remain provisional; this is prediction after an
unlabeled wearable observation period, not resting HR or a complete daily baseline. Earlier bundles,
raw files and participant folds remain unchanged. All 32 final-test participants stay sealed.
No imputation, expanded HRV, Garmin stress, binary outcomes, commit, push or provider messages.

[Notebook 10 verification](reports/past_hr_reference/verification_510f1ad96a9e46f3.md) records tests, deterministic rerun, notebook execution and privacy/integrity checks.


## Notebook 11 — measurement-source and timestamp evidence

[Notebook](notebooks/11_measurement_source_and_timestamp_audit.ipynb),
[scope specification](reports/measurement_source_specification.md),
[report](reports/measurement_source/1c44cb036f4ba779/report.md),
[professor briefing](reports/measurement_source/1c44cb036f4ba779/professor_provider_briefing.md),
[answer/decision table](reports/measurement_source/1c44cb036f4ba779/decision_table.md),
[provider request draft — not sent](reports/measurement_source/1c44cb036f4ba779/provider_request_draft.md).

The collection API/SDK and version remain unidentified. The study document's proposed Vivoactive
5/SDK integration does not establish provenance of these exports. Our parser is traceable from CSV
onward; acquisition versus receipt, upstream batching/order and beat-quality semantics need source
code or a release-specific contract. Both interval-start/end hypotheses remain provisional and may
both fail under receipt clocks. Physical duplicates survive; identical scalar rows alone cannot
identify duplicate delivery versus distinct intervals flattened from a batch.

This checkpoint reuses saved diagnostics and matched-cohort results. It does not read new sensor/EMA
values, run models or calculate HRV. It preserves frozen rules and corrected window provenance. The
next action is a minimal provider-assisted payload-to-CSV trace, not a larger model search. The 53.5%
within-person score variation does not establish wearable predictability.

Ignored immutable evidence bundle: `private/measurement_source/runs/96d7800ef13ab00b`.
Public report: `reports/measurement_source/1c44cb036f4ba779`.
Set the two existing local project paths before running the bounded audit:

```sh
PYTHONPATH=src .venv/bin/python -m garmin_measurement_source.audit \
  --reference "$GARMIN_REFERENCE_PROJECT" \
  --documentation-root "$GARMIN_DOCUMENTATION_ROOT"
```

No collection code or metadata outside those roots is claimed to have been searched. Private paths
and study-document extracts remain ignored. All 32 final-test participants remain sealed. No commit,
push, provider messages, threshold relaxation or new real-data calculations.

[Notebook 11 verification](reports/measurement_source/verification_96d7800ef13ab00b.md).


## Measurement-evidence follow-up — source location pending

[Review status](reports/measurement_evidence_review/8dcdae2b91ab162e/review.md),
[five-measure dictionary](reports/measurement_evidence_review/8dcdae2b91ab162e/feature_dictionary.md),
[existing aggregate coverage](reports/measurement_evidence_review/8dcdae2b91ab162e/coverage_comparison.md),
[conditional matched-cohort proposal](reports/measurement_evidence_review/8dcdae2b91ab162e/proposed_matched_cohort_experiment.md).

The referenced new measurement evidence has not been located; no uncertainty is marked resolved.
These versioned documents reuse existing corrected coverage only. No revised pipeline, eligibility
rebuild, expanded IBI calculation or model has been run. The exact source path/link is needed before
those dependent steps. Prior artifacts and all 32 sealed final-test participants remain preserved.

## Notebook 12 — exploratory PSS classification

[Notebook 12](notebooks/12_pss_classification.ipynb) compares majority/logistic baselines with nested,
participant-grouped XGBoost and LightGBM for exploratory three-item composite classes 3–4, 5–9 and
10–15. Binary ≥5/≥10 targets are secondary 60-minute checks. This new authorization permits class
thresholds, boosting, class weights and feature-level imputation; prior checkpoints remain unchanged.

The [frozen specification](reports/pss_classification/specification.md),
[report](reports/pss_classification/report.md), [abstract](reports/pss_classification/abstract.md) and
[reproduction guide](reports/pss_classification/README.md) distinguish matched physiological comparisons
from changes in coverage or recording missingness. Results use development participants only; the final
test stays sealed. Individual predictions, fold membership and fitted models remain ignored under
`private/pss_classification/`. Notebook 12 alone retains reviewed aggregate execution outputs, as requested.
