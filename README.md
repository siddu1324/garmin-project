# Garmin + EMA postpartum stress research

**Start with [Notebook 21 — central research overview](notebooks/21_central_research_overview.ipynb).**
It contains 15 executed, code-only sections covering data, missingness, all candidate features,
training inputs and matched-cohort results. For the full implementation, see
[Notebook 20](notebooks/20_complete_pipeline_for_review.ipynb) and the canonical [`src/`](src/) modules.

This project asks whether recorded Garmin measurements preceding ecological momentary assessments
(EMAs) predict the study's **three-item perceived-stress composite**, scored 3–15. It is exploratory
postpartum research, not a clinically validated diagnostic or a validated real-time prediction system.

## Data and preprocessing

The original structural inventory contains 4,680 EMA responses from 157 participants. Frozen
assignments reserve **125 development participants and 32 sealed final-test participants**.
Development processing preserves **all 3,706 EMA rows**; 3,689 have a valid composite and context
for the comprehensive modeling cohort. Garmin recordings are available for 123 development participants.
The held-out final test has not been evaluated.

Original EMA tables differ by survey category. Garmin exports contain HR, IBI, steps, floors,
PulseOx, calories, intensity minutes and device stress. Development normalization preserves
57,552,038 physical sensor records; the logical view contains 57,551,189 after collapsing 849 exact
duplicate excess records. Raw files, original strings/clocks, source hashes, row lineage and the
verified participant crosswalk remain private.

1. Match identities using the verified crosswalk and source provenance; preserve assignments/folds.
2. Parse EMA UTC as actual local start minus that row's local-minus-UTC offset. Parse Garmin `ts`
   as Unix seconds without another timezone shift. Flag discrepancies; do not shift clocks to improve alignment.
3. Use actual-start cutoffs: subtract 60 seconds for minute-resolution starts, no buffer for
   second-resolution starts. Feature intervals are left-inclusive and right-exclusive.
4. Keep physical records; distinguish exact duplicates, repeated values and timestamp conflicts.
   Counter increments require defensible original adjacent pairs fully inside the interval; resets
   and missing observations never become negative or zero activity. No sensor interpolation.
5. Generate observed-reading summaries and separate quality diagnostics; link by the stable EMA key.
   Score `nervous + (6 - cope) + (6 - control)` only when all three items are valid integers 1–5.
   Preserve missing outcomes; never impute them.

See the [inventory](reports/inventory_report.md), [data dictionary](reports/data_dictionary.md),
[feature specification](reports/hr_activity_feature_specification.md) and
[current measurement contract](reports/measurement_alignment/source_contract.md).
**Old 12-/24-hour IBI window results are superseded**: use the
[window supersession record](reports/window_supersession.json) and corrected loaders. No old bundle is overwritten.

## Features: 678 candidates, not 678 validated model inputs

[Complete feature inventory CSV](reports/central_research_overview/complete_feature_inventory.csv)
includes every name, formula, unit, support rule, status, lookback and missing count/percentage.

| Feature group | Candidate columns | Status |
|---|---:|---|
| HR observed-reading summaries | 240 | Supported observed summaries |
| Supported step increments | 56 | Supported observed summaries |
| Supported floor increments | 56 | Supported observed summaries |
| HR/activity interactions | 24 | Supported observed summaries |
| Adjacent-window HR differences | 14 | Supported observed summaries |
| Earlier HR references and deviations | 48 | Supported observed summaries |
| **Supported subtotal** | **438** | Main physiological candidate block |
| PulseOx summaries | 240 | Provisional, separate sensitivity only |
| **Total** | **678** | Generated candidates |

Lookbacks are **5, 15, 30, 60, 120, 360, 720 and 1,440 minutes**. Summaries include mean, median,
sample SD, extrema, quantiles, trends, observed-minute/rolling summaries and original-adjacent changes.
Earlier HR references are observed-history summaries, not resting HR or complete daily baselines.
“Supported” describes implemented measurement/support rules; physiological and timestamp validity remain provisional.
Garmin stress and unvalidated HRV are excluded from these predictors. The small IBI variability pilot
is a separate measurement study; a ≥240-second run is not complete five-minute HRV.

The saved feature frame is **3,706 × 693** including metadata/context/targets; its candidate block
is **3,706 × 678**. Primary full-logistic models use 438 supported candidates plus time/survey context,
training-only preprocessing and missingness indicators: **881 encoded inputs in folds 0–3, 880 in fold 4**.
Inspect [actual model inputs by fold](reports/central_research_overview/actual_model_inputs_by_fold.csv),
[missingness by modality](reports/central_research_overview/missingness_modality.csv) and
[missingness by lookback](reports/central_research_overview/missingness_lookback.csv).
Quality-only analyses are labeled diagnostics; missingness indicators are not physiological measurements.

## Models, evaluation and findings

All validation uses the five frozen participant-grouped development folds. Nested experiments use
participant-grouped inner folds; imputation, scaling, encoding, filtering and feature selection are fit
inside training splits. Compare models on identical eligible training/validation cohorts. Bootstrap
intervals resample participants and describe uncertainty in fixed exploratory OOF predictions.
Repeated development exploration remains exploratory even with nested tuning.

Regression includes mean/median dummies, Ridge, Lasso, Elastic Net, Huber, histogram gradient boosting,
XGBoost and LightGBM. Classification includes majority/logistic baselines, XGBoost and LightGBM,
class-weighting and feature-level missingness sensitivities. Three classes are 3–4, 5–9 and 10–15;
binary cutoffs ≥5 and ≥10 are exploratory research definitions, not clinical diagnoses.

| Matched full-feature cohort: 3,689 responses / 125 participants | Context logistic | Context + 438 candidates |
|---|---:|---:|
| Three-class macro AUROC (one-vs-rest) | 0.472 | 0.499 |
| Three-class macro F1 | 0.245 | 0.335 |
| Three-class balanced accuracy | 0.333 | 0.338 |
| High-class recall | 0.000 | 0.109 |

Notebook 13's F1 improvement does not establish useful ranking: the paired AUROC difference
interval includes zero. In Notebook 14's same-sized matched regression cohort, the median dummy
has MAE 2.209; full-feature models range approximately 2.245–2.314 and do not improve that baseline.
The earlier Notebook 12 boosting comparison uses a **different, 1,355-response / 113-participant cohort**:
XGBoost/LightGBM macro F1 is 0.306/0.311, with high-class recall only 1.8%/1.2%. Do not rank these
against the full-feature cohort's metrics. Latest full-block binary follow-ups also show weak discrimination.

Recording gaps and missingness are substantial. At 60 minutes, 1,725/3,706 responses have any accepted
HR; any reading is not representative coverage. Acquisition versus receipt/batching semantics,
beat quality and delivered-prompt logs remain unresolved. Morning items refer to waking;
Midday/Endday refer to the last prompt. Within-person score variation alone does not imply wearable
predictability. Measurement timing/quality evidence remains the next priority, not automatic search expansion.

## Navigation

- [Research index: Notebooks 00–21 and current reports](reports/README.md)
- [Central overview and verification](reports/central_research_overview/README.md)
- [Full code review](reports/complete_pipeline_review/README.md)
- [Feature engineering presentation](reports/comprehensive_features/index.html) and [dictionary](reports/comprehensive_features/feature_dictionary.md)
- [Regression presentation](reports/comprehensive_regression/index.html) and [summary](reports/comprehensive_regression/summary.md)
- [Classification report](reports/pss_classification/report.md) and [abstract](reports/pss_classification/abstract.md)
- [Binary follow-up](reports/binary_improvement/index.html), [temporal experiment](reports/temporal_personalization/index.html),
  [root-cause audit](reports/root_cause_analysis/index.html), [measurement audit](reports/measurement_alignment/index.html)
- [Provider request draft — not sent](reports/measurement_alignment/provider_data_request.md)

Download HTML presentations and open locally; GitHub displays their source. Versioned reports,
supersession records and prior results are preserved. The [previous README](reports/repository_publication/history/README_before_publication.md)
is retained as historical documentation; its relative links are interpreted from the repository root.

## Setup and reproducibility

Python **3.11** and **uv** are used throughout; `pyproject.toml` and `uv.lock` pin the environment.

```sh
uv sync --frozen
uv run pytest -q
uv run ruff check src tests
uv run jupyter lab notebooks/21_central_research_overview.ipynb
```

The repository includes code, source notebooks, reviewed aggregate execution outputs and reports.
Viewing saved results and running the synthetic tests does **not** require participant data.
Re-execution of data-dependent notebooks requires authorized local private artifacts; they are deliberately
not downloadable from GitHub. This checkout is not a self-contained public participant dataset.

For an authorized research checkout, configure paths in the shell, never a tracked local config:

```sh
export GARMIN_REFERENCE_PROJECT='/absolute/path/to/authorized/reference/project'
export GARMIN_DATA_DIR='/absolute/path/to/authorized/raw/archives'
export MPLCONFIGDIR="${TMPDIR:-/tmp}/garmin-mpl"
PYTHONPATH=src uv run jupyter nbconvert --to notebook --execute \
  notebooks/21_central_research_overview.ipynb --inplace --ExecutePreprocessor.timeout=300
PYTHONPATH=src uv run python -m garmin_central_overview.verify
```

Notebook 21 reuses corrected immutable bundles, independently checks feature missingness and saved
OOF metrics, and performs no new fits. Exact required artifact paths are listed in
[its guide](reports/central_research_overview/README.md) and `config/complete_pipeline_review.json`.
Keep private snapshots/manifests and privacy-token files alongside the authorized bundles. Historical
checkpoint code/manifests may retain original workstation paths; consult each reproduction guide and
supply the corresponding authorized references before regenerating that checkpoint. Historical seals
refer to their original source versions. The publication maintenance record explicitly accounts for
README/Git-guard changes; it does not replace any earlier metric, fold or feature artifact.

For expensive feature/model regeneration, use each checkpoint's specification and reproduction guide
in the research index. Changed code/configuration must produce a new version. Do not rerun raw
preprocessing merely to view a report, and do not overwrite a sealed result.

## Privacy and repository policy

Raw data, participant crosswalks/assignments, private manifests, participant-level features, predictions,
fitted models, archives, credentials, `.env` files, local configuration and caches stay ignored.
Only individually reviewed aggregate CSVs are exempted from the default CSV exclusion. The generated
source ZIP/unpacked source copy is omitted because canonical code is already present; it is preserved
locally. Historical duplicate research reports remain as provenance, not deleted as clutter.

Publication checks validate the exact staged Git blobs, known private identifiers, notebook output
schemas, report table schemas, credential patterns and outgoing history before a normal push.
The [publication record](reports/repository_publication/README.md) documents verification and exclusions.
