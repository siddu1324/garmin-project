# Pre-fit measurement and implementation audit

The audit reuses development-only immutable inputs. No raw sensor processing, final-test exploration,
clock shift, new HRV, or scientific threshold repair was performed. No genuine implementation bug
was identified in the audited scoring, corrected boundaries, or representative saved-feature traces.
`audit/checks.json` records the executed counts; the full test suite covers additional synthetic cases.

| Area | Confirmed implementation | Potential problem or unresolved assumption |
| --- | --- | --- |
| Outcome | All 3,706 scores independently replayed from original normalized item strings; three complete 1–5 integers required, cope/control reversed; ≥5 and ≥10 unchanged | Three-item study composite, not a diagnostic scale; incomplete outcomes excluded without imputation |
| Identity and folds | Stable EMA-key one-to-one matching, participant/fold equality, development-only loader and frozen assignments | Original crosswalk evidence cannot independently establish identity in keyless source CSVs |
| Clock | Independent Python datetime replay of actual local start minus row offset; minute precision subtracts60s, seconds do not | Actual-start application event and acquisition versus receipt/batching remain undocumented |
| Boundaries | Independent datetime checks of all five requested saved lookbacks; left inclusive/right exclusive; deterministic nine-EMA traces into accepted HR source rows | Export time preceding cutoff does not prove a reading was physically acquired or available before the EMA |
| HR | Existing provisional30–240bpm acceptance; rejected clocks/conflicts excluded; identical timestamp/value groups count once; values at different timestamps preserved | Sensor quality and timing accuracy remain unverified; acceptance is not validated physiology |
| Activity | Only supported nonnegative counter pairs wholly contained in windows; resets, source/day/offset changes and long gaps break support; no missing→zero | Counter semantics and sampling remain provisional; no causal activity adjustment |
| Statistics | Minimum2readings/span1s for most moments; slope≥3/span60s; observed UTC-minute means leave empty bins missing; original pair barriers preserved | Sparse bursts can dominate reading-weighted summaries; two points are not representative coverage |
| Model preprocessing | Imputation/scaling/encoding/selection inside each training split; no outcomes, IDs or post-cutoff clocks in predictors | Median filling and missingness flags may encode recording behavior rather than physiology |
| Repeated responses | Original participant outer folds and grouped inner folds; no cross-fold participant donors | Overlapping windows and repeated responses are not independent observations; cluster intervals remain conditional |
| Morning recall | Since waking; supplied current-response wake answer supports a retrospective candidate only | Same-day and unchanged-offset interpretation remain assumptions; do not infer wake from Garmin |
| Midday/Endday recall | Both say since the last prompt, not since the last completed EMA or the whole day | Complete delivered/expired/unanswered prompt history absent; aligned prospective windows cannot be constructed |
| SMOTE | Numeric-only low-dimensional standard interpolation; separate constrained within-person/context variant; every donor remains inside training | Cross-person interpolation and imputed coordinates may be implausible; convex bounds do not establish physiological realism |

Evidence: `src/garmin_ema_baseline/linkage.py`, `src/garmin_preprocessing/normalize.py`,
`src/garmin_hr_activity/features.py`, `src/garmin_comprehensive/features.py`,
`src/garmin_comprehensive/augment.py`, and the exact formulas in
`reports/comprehensive_features/feature_dictionary.md`. Questionnaire code/label evidence is
`reports/ema_baseline/d933d3999d949477/questionnaire_evidence.json`; exact recall wording with local
PDF page citations is `reports/recall_alignment/84f2ff949f1ca478/questionnaire_evidence.md`.
`reports/window_supersession.json` and current-bundle guards reject the earlier int16-overflow versions.
`reports/measurement_evidence_review/8dcdae2b91ab162e/review.md` explains the still-missing exporter evidence.

Synthetic trace: items nervous4, cope2, control3 give11 (positive for both tasks). Local09:22 at
offset−240min gives UTC13:22; minute precision gives cutoff13:21. Six-hour support is
[07:21,13:21). A reading at13:21 is excluded. HR60,60,60 during one observed minute and120 in another
gives reading mean75 but observed-minute mean90. Neither fills unobserved minutes. Counter10→2 is
a reset, never−8steps. These cases are independently covered by the reused/new tests.
