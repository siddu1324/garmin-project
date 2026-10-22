# Notebook 17 — what the data can establish

The numerical pipeline reproduces scores, participant linkage and corrected cutoffs. No new defect in
previous preprocessing was established. This does not establish acquisition time, beat quality, label
reliability or whether a historical answer was available for a prospective prediction.

| Area | Verified local evidence | Remaining limit / consequence |
|---|---|---|
| Outcome | `PSS_NERVOUS + (6−PSS_COPE) + (6−PSS_CONTROL)`; all three stored integer codes 1–5 required; display anchors 0–4 | Study three-item composite, not PSS-10 or a clinical diagnostic threshold |
| Recall | Morning: “Since waking up”; Midday and Endday: “Since the last prompt”; same stem for all three items | Different recall periods; no full prior delivered-prompt logs; Endday is not automatically all day |
| Start/cutoff | Local actual start minus row offset **in minutes**; minute-resolution starts buffered 60s; second starts unbuffered | Start is not submission; scheduled instance is not actual delivery |
| Prior answers | Schema contains actual/scheduled starts, items and an instruction/completion flag | No completion/submission/receipt timestamp or answer-version log. **Zero responses have verified online EMA history** |
| Garmin time | Unix seconds preserved, no extra timezone shift | Acquisition versus callback/receipt/batching unresolved; all sensor timing claims provisional |
| HR | Existing 30–240 bpm range, clock/conflict gates, one logical timestamp/value copy; distinct timestamps preserved | No provider wear/quality metadata; observed values do not represent empty periods |
| Activity | Existing accepted nonnegative consecutive counter differences; reset/rejection breaks; wholly contained intervals | Counter semantics provisional; no partial-boundary assignment, no missing-to-zero substitution |
| Device/API | Study protocol mentions a device and proposed integration | Export does not identify collection SDK/API/version/firmware; generic API documentation cannot establish it |
| Sleep/IBI/PulseOx | Current Morning sleep answers; isolated limited IBI pilot; provisional PulseOx | None establishes a prediction-time validated sleep/HRV/PulseOx feature |
| Splits/transforms | Frozen five outer participant folds; grouped inner folds; train-only medians/scaling/encoding | Repeated development exploration; bootstrap conditions on fixed predictions, not new-study performance |

Exact local evidence:

- [Questionnaire wording, PDF hashes and pages](../recall_alignment/f518537c0a3a6f14/questionnaire_evidence.md).
- [Clock and missing-log audit](../recall_alignment/f518537c0a3a6f14/report.md).
- [Measurement-source evidence ledger](../measurement_source/f647865329b31203/evidence_ledger.md): study `R61-R33_ResearchStrategy.pdf`, pp.5 and7, SHA-256 `cad3443ed2dc340830b20563d36deeb685ec126a4249881e5b34ebbbf66462ff`; this describes a study/proposed integration, not an export contract.
- [Measurement evidence review](../measurement_source/f647865329b31203/professor_provider_briefing.md).
- [Accepted-feature contract](../comprehensive_features/specification.md), [corrected-window supersession](../window_supersession.json).
- Code: `src/garmin_preprocessing/normalize.py` (`parse_ema_clock`),
  `src/garmin_hr_activity/features.py`, `src/garmin_comprehensive/features.py`,
  `src/garmin_temporal_personalization/history.py` and `audit.py`.
- Actual development schema inspected at `private/preprocessing/normalized/31e51c934bcdeffa/ema/development_ema.parquet`;
  no participant rows were displayed. `audit/longitudinal.json` records field-search result and checks.

## Synthetic before/after examples

Stored items nervous=4, cope=2, control=3 → oriented values [4,4,3] → composite11 → both binary targets positive.
A missing item gives a missing composite, never an imputed label. Minute start09:22 → cutoff09:21 →
current HR window[08:21,09:21). Second start09:22:35 → cutoff09:22:35.

A fictional prior survey starts07:00 and has a verified receipt07:05: its answer can enter a09:21 prediction.
Another starts09:10 but arrives09:25: exclude it. If its receipt time is unknown, also exclude it.
The real export supplies starts but not those receipt clocks. A long elapsed time cannot repair this evidence gap.

Four HR readings60,60,60,120, with the first three in one minute and the last in another, have
reading mean75 but observed-minute mean90. Empty minutes do not enter either denominator as zeros.
A counter pair08:20→08:25 crossing a08:22 left boundary is excluded, not wholly assigned to the window.

## Label diagnostics are not a reliability certificate

Item distributions, floor effects and within-person sums-of-squares are reported separately.
Cronbach alpha is a descriptive covariance statistic; repeated participants and changing recall periods
limit interpretation. Neither pooled nor person-centered alpha proves stable measurement or a clinically
meaningful two-point change. Full-record centering is confined to retrospective diagnostics, never predictors.
Consecutive **valid** start-ordered score correlations can reflect stable between-person differences and
serial dependence. They do not prove an earlier response was available before the next cutoff.

## Focused provider request — draft only

1. For each stable EMA instance and each exported answer version: UTC start/resume, final item-save,
   submission, server receipt and subsequent edit times, precision/timezone definitions, and whether
   exported values can be revised after initial submission. We need the time these exact three values
   became available, not merely a completion flag.
2. Complete scheduled/delivered/dismissed/expired/suppressed prompt logs, including unanswered prompts,
   plus instructions defining “last prompt” and waking/overnight handling.
3. Collector app/build, exact Garmin API/SDK/version and device/firmware; timestamp assignment and batch
   flattening code; a redacted payload-to-export trace preserving event/receipt clocks and ordering.
4. Wear/contact/measurement quality flags, rejected/missed-beat encoding, IBI anchoring and activity-counter
   reset/increment semantics. Provide validated sleep output provenance if that modality is intended.

Nothing has been sent. Once verified availability evidence is supplied, version the source contract and
execute the already specified online-history/chronological comparisons; do not substitute guessed delays.
