# Methodology audit: Garmin + EMA feature-engineering review

Review date: 2026-10-16. Scope: a read-only explanation for Will of existing development preprocessing,
feature engineering, outcome construction and logistic classification. No real-data model fitting,
feature regeneration, threshold change, source-code edit, provider message or final-test evaluation.
The report describes the postpartum project framing without independently re-adjudicating the cohort's
clinical/peripartum eligibility.

Deliverable: `index.html` is self-contained: inline CSS/JavaScript, inline SVG diagrams and an embedded
copy of the existing aggregate survey-availability PNG. Its internal anchors work without companion
files. Source filenames are evidence references, not rendering dependencies. It is readable without
JavaScript; JavaScript only manages navigation highlighting, expansion and printing.

## 1. Exact implementation trace

All links below are relative to this audit. Names refer to the current code actually inspected.

| Pipeline step | Implementation | Verified behavior |
|---|---|---|
| Raw archive discovery | [sources.py](../../src/garmin_inventory/sources.py), `discover` and source readers | Corrected ZIP/CSV inputs; no collection-app/exporter source is inferred from these analysis readers. |
| Participant resolution | [audit.py](../../src/garmin_inventory/audit.py#L85), `resolve_pair` | Exact participant key; unambiguous respondent crosswalk; conflicting usable keys rejected. |
| Source identity fallback | [normalize.py](../../src/garmin_preprocessing/normalize.py#L70), `reference_identity_registry`, `normalize` | Content-hash registry from verified reference metadata; requires one participant per source hash. This corroborates prior identity provenance, not independent evidence within a keyless CSV. |
| Development gate and stable EMA key | [normalize.py](../../src/garmin_preprocessing/normalize.py#L92), `normalize_ema` | Keep development identity rows; key combines source index and original source row; retain original fold, schema, strings and source hashes. |
| Clock conversion | [normalize.py](../../src/garmin_preprocessing/normalize.py#L25), `parse_ema_clock`, and `normalize_chunk` | EMA actual local minus row offset; Garmin Unix seconds without additional shift; preserve scheduled clocks and discrepancy flags separately. |
| Raw-preserving normalization | [normalize.py](../../src/garmin_preprocessing/normalize.py#L132), `normalize_chunk`, `normalize` | Numeric copies and quality flags alongside originals; chunked processing, participant/sensor partitions, physical and logical views, Parquet round-trip checks. |
| Exact duplicate annotation | [normalize.py](../../src/garmin_preprocessing/normalize.py#L51), `annotate_duplicates` | Compare six raw fields plus original source schema, retain all physical rows, mark excess logical copies. Preserve repeated values at different times. |
| Reading acceptance and counter adjacency | [features.py](../../src/garmin_hr_activity/features.py#L30), `prepare` | HR/clock gates and whole-timestamp-group tie/conflict handling. Counter pairs preserve rejected groups as barriers. |
| Direct EMA lookbacks | [pipeline.py](../../src/garmin_hr_activity/pipeline.py#L187), `run` | For every EMA and L, `end=cutoff`, `start=end−60×int(L)`; no int16 offset multiplication. |
| HR/counter summaries | [features.py](../../src/garmin_hr_activity/features.py#L117), `window` | Left-inclusive/right-exclusive search; feature-specific support rules; independent quality and lineage outputs. |
| Earlier reference | [reference.py](../../src/garmin_past_hr_reference/reference.py#L14), `earlier_reference`, `build_reference` | Accepted observations in the past-only 24-hour reference interval; equally weighted occupied UTC-minute means, with minimum support. |
| Outcome coding and context | [linkage.py](../../src/garmin_ema_baseline/linkage.py#L48), `score_items`, `local_context` | Valid integer 1–5 items only; reverse coping/control; local time-of-day sine/cosine. |
| One-to-one outcome linkage | [linkage.py](../../src/garmin_ema_baseline/linkage.py#L74), `check_development`, `link_outcomes` | Unique key and matching key sets; participant, partition, fold, category, cutoff, precision and invalid-clock metadata agree. |
| Class mapping and predictor whitelist | [data.py](../../src/garmin_pss_classification/data.py), `targets`, `feature_set`, `cohort_mask` | Fixed target categories and permitted feature names. Matched union-of-features cohorts; no outcome imputation. |
| Model transformation | [models.py](../../src/garmin_pss_classification/models.py), `Transform.fit`, `Transform.transform` | Training-only medians, StandardScaler and OneHotEncoder; wholly missing training numeric columns dropped; authorized wearable missingness flags only in imputed branch. |
| Logistic fitting and tuning | [models.py](../../src/garmin_pss_classification/models.py#L63), `classifier`, `fit_model`, `inner_splits`, `tune`; [run.py](../../src/garmin_pss_classification/run.py), `fold_job` | Frozen five outer participant folds, three grouped inner folds; training-only processing; inner macro-F1 selection; fixed decision rule. |
| Metrics and support | [metrics.py](../../src/garmin_pss_classification/metrics.py), `confusion`, `score`; [evaluate.py](../../src/garmin_pss_classification/evaluate.py), `cohort_accounting` | Pooled OOF metrics, fixed task labels, class support and original fold accounting. |
| Invalid artifact protection | [provenance.py](../../src/garmin_preprocessing/provenance.py), `assert_current_windows`; [common.py](../../src/garmin_preprocessing/common.py), `verify_bundle` | Reject known-obsolete int16-window versions/manifests; verify current artifact checksums. |

Notebook headings and relevant report sections from 00–12 were inspected to distinguish historical
plans from current behavior. In particular, the older feature plan's broad summary list does not
replace the current classification whitelist. Notebook 04 computes additional HR summaries and floors;
Notebook 12's primary logistic model uses only the four HR features, supported steps and authorized context.

## 2. Verified parameters and exact formulas

### Clocks, row preservation and identity

- Offset convention: minutes **local minus UTC**. EMA UTC = actual local start − row offset.
- Source precision is detected from the actual-start string. Minute-resolution start: subtract 60 s;
  second-resolution start: subtract 0 s. Invalid actual clock/offset/precision yields invalid cutoff.
- Scheduled time and `instance_id` discrepancies are retained; they do not replace actual start or
  automatically shift it. No current/past survey is silently substituted for a missing prompt event.
- Garmin `ts` → numeric `n_epoch_s` → UTC directly. Epoch guard: 946684800 through 2082758400 inclusive.
- Text clock `dte_tme` is parsed separately. Inferred offset = round((text-clock seconds − epoch)/60).
  Residual magnitude **>1.1 seconds** flags disagreement. Allowed offsets: **−480, −420, −360, −300 minutes**.
  These are existing dataset-specific checks, not a general timezone or device standard.
- Normalization chunks: **100,000 rows**; per-participant/sensor memory ceiling: **3,000,000 rows**.
- Physical rows retain original strings, original schema, source/archive hashes, source row number,
  participant mapping method, partition and fold. Logical rows also link back to physical rows.
- Exact duplicate keys are `ts`, `dte_tme`, `rsp_id`, `mbl_cod`, `data_type`, `val`, plus source-column schema.
  This is not a claim to compare arbitrary future additional columns. In the inspected export union,
  these are the six raw sensor fields. Same-value HR at distinct timestamps is retained.
- Same-time/value ties contribute one analysis reading only if all members pass. Different parsed
  values at a timestamp, including mixed missing/nonmissing values, reject the whole group.

### HR features: each L ∈ {5,15,30,60,120,360,720,1440} minutes

Let `t_i = n_epoch_s`, `v_i = n_value` for accepted `data_type=hr` groups within `[cutoff−60L, cutoff)`.
Acceptance requires a finite value in **[30,240] bpm**, no clock flag listed above, and no timestamp conflict.
These provisional gates do not verify source signal quality, normal beats, rhythm, contact or motion state.

| Feature and exact column | Formula / units | Minimum support | Missingness and interpretation |
|---|---|---|---|
| `hr_mean_bpm__{L}m` | μ = Σv/n; bpm | ≥2 distinct accepted timestamps, span ≥1 s | Missing otherwise. Reading-weighted, not time-weighted. |
| `hr_sd_bpm__{L}m` | sqrt(Σ(v−μ)²/(n−1)); bpm | Same as mean | Sample SD of observed HR; not IBI SDNN or another HRV feature. |
| `hr_latest_bpm__{L}m` | Value at maximum accepted timestamp below cutoff; bpm | ≥1 accepted reading | Missing otherwise. No extra freshness gate or carry-forward from outside the interval. |
| `hr_slope_bpm_per_minute__{L}m` | Σ((x−x̄)(v−μ))/Σ((x−x̄)²), x=(t−t_first)/60; bpm/min | ≥3 distinct accepted timestamps, span ≥60 s | Missing otherwise. OLS with intercept across observed points, potentially across large gaps. |

No primary HR minimum occupancy, maximum gap or maximum age is imposed beyond these feature gates.
Gaps are not interpolated. The inventory's 2-second diagnostic is not an HR acceptance rule.
Recorded rows, accepted readings, occupied bins, timestamp span, edge/internal gaps and latest age
are separate quality measures, excluded from the primary predictor vector.

Additional implemented summaries not used by this logistic model: HR median, min/max and Q25/Q75,
all with the mean's ≥2/≥1 s support gate. Median/min/max are the corresponding accepted-value order
statistics; quartiles use NumPy's linear quantile estimator. That estimator does not create missing
sensor readings. Floors are also implemented under counter rules but excluded here. Calories and
intensity remain unsupported placeholders. Garmin stress and IBI/HRV remain excluded.

### Supported step additions

`steps_added_supported__{L}m` sums nonnegative differences in `val → n_value` between **original
consecutive timestamp groups**, not consecutive rows after deleting bad readings. Both groups must:

1. Be accepted under the same clock/tie rules; counters must be finite nonnegative integers.
2. Have strictly increasing time with **0 < Δt ≤300 s**.
3. Share exported local calendar date, inferred offset and source content hash.
4. Have nondecreasing counter value.
5. Lie fully inside `[cutoff−60L, cutoff)`; the earlier endpoint must be at or after its left edge.

A negative reset pair is excluded; its valid lower endpoint can serve as a new baseline for the next
consecutive valid reading. Rejected groups sever incoming/outgoing pairs, preventing artificial adjacency.
Right-crossing pairs are not inspected using a future endpoint. No proration or whole-increment attribution
across a window boundary occurs. At least one supported pair is needed; otherwise the feature is missing.
Zero is emitted only when supported observed increments sum to zero. This is partial recorded activity,
not a complete daily/window total. Counter/reset semantics and the 300-second attribution cap are provisional.

### Context and earlier-recording reference

| Feature | Source and calculation | Support / missingness | Limitation |
|---|---|---|---|
| `local_time_sin` | `n_actual_local`; sin(2πu/86400), u = local seconds since midnight including fractional seconds | Parseable actual local time; context validity also requires permitted category and valid cutoff | Not time since waking or last prompt; same value across all lookbacks. |
| `local_time_cos` | Same source; cos(2πu/86400) | Same | Combined sine/cosine expresses cyclic time without a midnight discontinuity. |
| `n_survey_category` | EMA schema: `SLEEP_SLEEPTIME` column → Morning; else `ENDDAY_TEXT` → Endday; else Midday | Known permitted category; training-fitted one-hot encoding | No response item value used. Survey label alone does not establish the recall interval's start. |
| `reference_mean_bpm` | Accepted HR ledger `n_epoch_s`, `n_value`, `a_analysis_keep`; restrict to `[cutoff−25h, cutoff−1h)`, group by floor(epoch/60), average within each occupied UTC minute, then equally average minute means | ≥30 occupied minute bins across ≥3 UTC-hour bins; missing otherwise | Separate 60-minute experiment. Empty minutes stay missing; partial boundary minutes use only in-interval observations. Not resting HR or a complete day. |
| `hr_mean_deviation_bpm` | `hr_mean_bpm__60m − reference_mean_bpm` | Both operands available | Current mean is reading-weighted, reference is minute-weighted. Sampling changes can contribute to the difference. |
| `hr_latest_deviation_bpm` | `hr_latest_bpm__60m − reference_mean_bpm` | Both operands available | The latest value may itself be old or activity-related. |

The reference builder accepts explicitly unlabeled inputs and no past EMA outcomes, whole-person outcome
means, future wearable data or participant IDs as predictors. Unseen development participants may use their
own preceding unlabeled wearable history, subject to unresolved acquisition/availability semantics.

### Outcome and logistic transformation

- Raw fields: `PSS_NERVOUS`, `PSS_COPE`, `PSS_CONTROL`. Export codes are 1–5; displayed anchors are 0–4.
  Trimming/parsing preserves original strings and per-item missing/invalid/valid status.
- PSS = nervous + (6−cope) + (6−control), requiring three valid integer items; otherwise missing.
- Three classes: 0 for 3–4; 1 for 5–9; 2 for 10–15. Secondary binaries: score ≥5 and score ≥10.
- Primary complete-case cohort: valid target and context plus union of four HR summaries and supported steps.
  Context-only comparators use the **same** training and validation members. The separate HR-native sensitivity
  does not require steps; it is not directly compared as though it were the same population.
- Primary inputs: two local-time features + HR mean/SD/latest/slope + supported steps + one-hot survey category.
  With all three categories present: **10 transformed columns**, plus the fitted intercept.
- Numeric StandardScaler uses training mean and population SD (`ddof=0`); zero-variance scale is 1. The
  physiological HR SD feature itself uses `ddof=1`. The one-hot encoder keeps all training categories,
  does not standardize indicator columns, and ignores unseen categories by giving all zeros.
- Imputed sensitivity: training feature medians before scaling; wholly missing training numeric columns
  are dropped, with per-permitted-wearable missingness flags retained. Flags are not standardized.
  No target/raw reading is imputed. A separate context+missingness-only control excludes wearable values.
- Logistic: L2, `lbfgs`, `max_iter=2000`, fixed seed 20261015; C candidates **0.01, 0.1, 1, 10**.
  Three classes use multinomial probabilities; the two binary tasks use binary probabilities.
- Frozen five outer participant folds; original assignment contains 25 development participants per fold.
  Actual model-cohort participant counts are smaller. Three StratifiedGroupKFold inner folds, fixed seed,
  select C by mean macro F1. Each inner/outer transform is fitted on its own training rows.
- Primary weights are 1. The separate balanced sensitivity uses n/(K×n_class) from the corresponding
  training subset. Argmax predictions are used; no post-hoc cutoff optimization or final-test tuning.
- Selected primary wearable-logistic C by original fold 0–4: **10, 1, 1, 1, 1**.

## 3. Numerical and artifact provenance

| Evidence | Current artifact / version | Statements supported |
|---|---|---|
| Structural inventory | `reports/inventory_report.md`, `reports/inventory.json` | 4,680 EMA responses / 157 participants structurally; 125 development / 32 final-test assignments. No new sealed-data value inspection. |
| Raw-preserving development normalization | `private/preprocessing/normalized/31e51c934bcdeffa/manifest.json` | 3,706 EMA rows, 125 development participants, 123 with Garmin; 57,552,038 physical and 57,551,189 logical sensor rows; 849 duplicate excess rows. |
| Current corrected IBI eligibility | `private/preprocessing/eligibility/974e98729579967b` and `reports/window_supersession.json` | Old long-window int16 results are superseded; HR direct lookbacks are distinct from sliding IBI windows. |
| HR/activity feature table | `private/hr_activity/features/76f5885b3955d5b0` | Existing direct-lookback feature/quality/lineage tables, not regenerated for this review. |
| Published HR aggregates | `reports/hr_activity/7fdba29ad8d6f0fe/{availability,gaps,summary}.json` | Feature availability and reasons on all 3,706 EMAs. The embedded `survey_availability.png` is copied without alteration. |
| Linked outcome bundle | `private/ema_baseline/runs/640600af77404ff6` | Existing one-to-one scoring/linkage; no target recoding. |
| Earlier-recording reference | `private/past_hr_reference/runs/510f1ad96a9e46f3` | Separate history features. |
| Classification | `reports/pss_classification/8970bf5ebcc47bfd/{metrics,per_class,cohort_accounting,settings,verification}.json` | Pooled logistic/dummy metrics, all eight cohort counts, per-fold support, chosen C and prior verification. |
| Recall wording | `reports/recall_alignment/824551e40b8e301d/questionnaire_evidence.md` | Morning since waking; Midday/Endday since last prompt; source PDF page anchors. |
| Measurement uncertainty | `reports/measurement_source/1c44cb036f4ba779/evidence_ledger.md`; `reports/measurement_evidence_review/8dcdae2b91ab162e/review.md` | Exporter/SDK/version, acquisition/receipt, batching and quality semantics remain unverified. |

Full checksums were verified for the normalized, feature, linked outcome, earlier-reference, published
HR aggregate and classification aggregate bundles. **Current `normalize.py` matches its saved source hash;
current HR `features.py`, `pipeline.py` and HR configuration match their saved feature hashes.**
The shared `garmin_preprocessing/common.py` does **not** match the original normalization checkpoint's
helper hash. The current helper includes later supersession/version checks. This review does not claim
byte-identical historical helper replay or assume that the complete difference is limited to those checks;
it verifies preserved artifact checksums and the matching core transformation implementations instead.
No normalization rerun was attempted. This provenance distinction belongs in any exact reproduction.

### Availability and primary accounting

| L, minutes | HR mean | Latest | HR slope | Steps | Valid-score joint model cohort | Model participants |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1,486 | 1,498 | 1,403 | 666 | 625 | 104 |
| 15 | 1,542 | 1,554 | 1,458 | 965 | 919 | 111 |
| 30 | 1,598 | 1,610 | 1,517 | 1,197 | 1,146 | 113 |
| 60 | 1,714 | 1,725 | 1,643 | 1,410 | 1,355 | 113 |
| 120 | 1,884 | 1,897 | 1,817 | 1,647 | 1,585 | 118 |
| 360 | 2,690 | 2,693 | 2,619 | 2,341 | 2,272 | 123 |
| 720 | 3,095 | 3,098 | 3,065 | 2,762 | 2,693 | 123 |
| 1440 | 3,459 | 3,464 | 3,439 | 3,295 | 3,207 | 123 |

First four columns use all 3,706 EMA rows, regardless of score availability. The model column additionally
requires valid outcome/context and the complete feature union. No selection of a winning window is implied.
At 60 minutes, HR slope reasons partition all rows: no recording 1,909; all rejected 72; insufficient
accepted observations 16; insufficient span 66; available 1,643. Steps: no recording 2,056; no supported
pair 240; available 1,410. These are feature-level exclusive statuses, not additive raw rejection flags.

Primary modeling funnel: **3,706 −17 missing scores −2,055 missing complete HR −279 unsupported steps =1,355**.
The 17 score exclusions happen first; subsequent losses are conditional, not independent exclusion counts.
Included survey counts: Morning 177/934, Midday 805/1,838, Endday 373/934. Included mean PSS 6.27 versus
6.49 among excluded valid scores. Different representation motivates a selection-bias question; it does
not prove a missingness mechanism or causal bias. Imputed cohorts retain all 3,689 valid-score/context rows.

### Verified primary logistic results and class support

| Model, matched complete 60-minute cohort | Macro F1 | Macro OVR AUROC | Balanced accuracy |
|---|---:|---:|---:|
| Majority | 0.2490 | 0.5000 | 0.3333 |
| Logistic: context | 0.2490 | 0.4530 | 0.3333 |
| Logistic: context + HR | 0.2492 | 0.4615 | 0.3333 |
| Logistic: context + HR + steps | 0.2508 | 0.4644 | 0.3338 |

These are pooled OOF metrics, not averages of fold AUROCs. For the primary wearable-logistic model,
class-2 recall/precision are zero. Macro F1 has no universal chance level; the weak majority reference
should not be mistaken for evidence of useful physiological discrimination. Final-test performance is unknown.

| Frozen fold | Train low | Train middle | Train high | Held-out low | Held-out middle | Held-out high |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 310 | 604 | 134 | 74 | 204 | 29 |
| 1 | 299 | 655 | 121 | 85 | 153 | 42 |
| 2 | 309 | 621 | 136 | 75 | 187 | 27 |
| 3 | 299 | 671 | 120 | 85 | 137 | 43 |
| 4 | 319 | 681 | 141 | 65 | 127 | 22 |

Held-out support totals 384 low / 808 middle / 163 high =1,355 unique responses. Training sets overlap
across folds. Counts were checked against the saved outer membership in the preceding read-only metric
check and against the current aggregate cohort table here; no identifying membership is published.

## 4. Synthetic end-to-end trace

The private review script creates only invented records and uses existing production functions unchanged:
`normalize_chunk → annotate_duplicates → prepare → window → link_outcomes → targets → Transform`.
No actual participant row is copied, and no real-data estimator is trained or modified.

- Synthetic start: 2001-01-01 09:22 local, offset −300 minutes →14:22 UTC →cutoff14:21 UTC.
  The shown 60-minute local interval is [08:21,09:21). Only times of day appear in the HTML.
- Exact duplicate 60 at08:21 collapses in the logical view. Two differently formatted text-clock rows
  with timestamp08:22/value90 form a same-value tie and contribute once. Values80 and100 at08:22:30
  conflict and are both rejected. HR0 and300 are range-rejected; a 30-second clock disagreement is rejected.
  Pre-window and cutoff-edge readings are excluded.
- Accepted values [60,90,120] at elapsed minutes [0,1,2] yield mean90, sample SD30, latest120 and slope30.
  Actual latest age is3,480seconds (58minutes), despite all feature gates passing. Only3of60 relative
  minute bins are occupied. This demonstrates the support limitation without asserting real participant behavior.
- Step counters [90,100,105,119,3,8,8,20] at [08:20,08:21,08:22,08:23,08:24,08:25,08:26,09:21]
  produce24supported steps over240seconds. The left-crossing pair, reset pair and right endpoint are excluded;
  the unchanged8→8pair is an observed zero.
- Synthetic one-to-one linkage asserts identity/fold consistency. Items4,2,3 yield score11/class2.
- The synthetic training-only transform produces, rounded:
  `[0.900, −0.445, 0.000, 1.225, 0.816, 1.225, 1.225, 0, 0, 1]`.
  Order: local sine/cosine, mean, SD, latest, slope, steps, then Endday/Midday/Morning indicators.
  Independent population-SD standardization agrees with the actual transformer.

The first toy harness omitted duplicate annotation on its step stream; adding that required existing
stage fixed the harness. This was not a discovered defect in the production pipeline. A complete
successful trace is retained privately at `private/feature_engineering_review/synthetic_trace.json`.

## 5. Critical issues and bounded next experiments

**Provider evidence needed:** the actual collection SDK/API/version, callback-to-export code, timestamp
assignment and rounding, acquisition/receipt/upload clocks, original ordered payloads, retry/batch identity,
source quality annotations and activity-counter units/reset behavior. Full scheduled and delivered prompt
logs are needed for “since last prompt.” Never infer prompt time from the prior completed response or wake
time from the first Garmin reading. Current-response wake time remains retrospective-only.

**Testable with logistic regression after specifying the comparison:**

1. Compare reading-weighted means against equally weighted means of observed minutes, leaving empty minutes
   missing. Hold cohort and folds fixed; distinguish changed sampling weights from continuous time weighting.
2. Review the existing Notebook06 support sensitivities before proposing a new small support set. Separate
   restricting saved predictions from refitting. Preserve counts/freshness/gaps as diagnostics unless their
   use as predictors is explicitly justified in a separate experiment.
3. Compare wearable values with context-plus-missingness controls at the same lookback and on matched
   evaluation responses. Avoid interpreting broader imputed coverage as improved prediction or physiology.
4. Test documented recall boundaries only; keep retrospective wake comparisons separate from prediction-time
   features. Do not equate Endday with an all-day outcome or assume constant prompt spacing.
5. Prespecify logistic feature-block ablations (context, HR, supported activity, separate history); if any
   selection is proposed, perform it only inside training folds. Do not select a feature/window because
   this report suggests a favorable outcome. Review reference/current weighting differences explicitly.
6. Keep class thresholds fixed during these comparisons. Report class-specific precision/recall, probability
   ranking, participant-level uncertainty and subgroup support. Do not claim validated high-stress detection.

No new demonstrated production bug requiring correction was found. Important unresolved issues are
measurement/recall semantics, sparse descriptive support, activity confounding, class imbalance and selection.
Passing synthetic and leakage checks does not resolve them. No experiment above was launched for this report.

## 6. Validation record

- **72 relevant existing tests passed** across preprocessing, HR/activity, outcome linkage, earlier reference
  and classification. These include boundary inclusion/exclusion, resets, missingness, broken adjacency,
  participant separation and train-only transformations. Warnings were existing matplotlib/pyparsing deprecations.
- **Ruff passed** on existing `src` and `tests`. No production source/test/config changes were needed.
- Six existing input/public bundles verified by their recorded checksums. Current feature core and config
  hashes match their saved inputs; the older shared-helper hash caveat is documented above.
- Before/after hashes protect all existing source, configurations, reports, notebooks and private manifests.
  Only the two requested public deliverables and ignored review/validation artifacts are added.
- Browser rendering, responsive layout, print output, privacy checks and final preservation results are
  recorded at completion below. Participant-level data are never embedded in either deliverable.

### Completed presentation and preservation checks

- Rendered the finished standalone HTML in local headless Chrome at **1440 px desktop** and **390 px
  mobile** widths. Visually reviewed all ten sections and expanded feature/results tables. All 11 tables
  have contained scrolling when necessary; diagrams retain readable dimensions on narrow screens.
- Browser checks: ten sections, valid navigation anchors, embedded image loaded, no page-width overflow,
  no JavaScript exceptions, no external assets and no external requests. The expand/collapse control
  works. Sixteen screenshots and the browser audit are kept in the ignored review directory.
- Print rendering succeeded; its private 24-page verification PDF contains text on every page. The PDF
  includes expanded technical details; the main-screen narrative is the intended 10–15-minute reading path.
- Explicitly scanned **both new deliverables**, including HTML, against the existing private identifier/name
  registry: passed. This targeted HTML check is necessary because the historical public scanner does not
  include the `.html` extension. Existing privacy code was preserved. The embedded aggregate PNG was
  visually reviewed; all displayed row-level examples are synthetic.
- Existing Git privacy/ignore guard passed. Raw/reference file hashes and metadata match the verified
  inventory integrity baseline. Hashing checks integrity without interpreting sealed participant contents.
- Before/after SHA-256 comparisons passed for **625 existing source/configuration/report/notebook files**
  and **138 pre-existing private manifests**. Existing changes were preserved; no existing artifact was
  overwritten. New work is confined to these two public deliverables and ignored review artifacts under
  `private/feature_engineering_review/`.
- No real-data model was fitted or feature bundle regenerated. Frozen assignments and final-test sealing
  remain unchanged. No commit, push or provider message was made.
