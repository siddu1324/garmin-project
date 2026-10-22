# What the independent audit does and does not establish

| Suspected issue | Classification | Local evidence / action |
|---|---|---|
| Wrong score direction or coding | Verified correct behavior | Original development answers and all 3,706 normalized scores agree with nervous+(6−cope)+(6−control), stored valid integers 1–5. `audit.py::run`; exact blank forms and codebooks in Notebook 07 |
| Offset sign or unit | Verified correct behavior | Independent Python datetime arithmetic checks all 3,706 starts: local minus local−UTC offset in minutes. `normalize.py::parse_ema_clock`, `root_cause/audit.py` |
| Minute buffer or endpoint leakage | Verified correct behavior | Minute start minus60s, second start unchanged; [start,cutoff), scalar Python/datetime boundary checks including old overflow indices and 24h |
| Historical 12/24h int16 overflow | Confirmed historical implementation defect, already corrected | `reports/window_supersession.json`; current loaders reject obsolete versions; no old long-window counts reused |
| Participant mismatch | Verified correct behavior in audited lineage | Stable EMA key, source-content hash/row, frozen assignment and verified crosswalk lineage agree in sampled original traces; all development joins/folds checked. `garmin_window_benchmark/trace.py` |
| Conflicting readings averaged together | Verified correct behavior | Whole timestamp conflicts rejected; identical timestamp/value excess removed from logical view only; physical records preserved. Original-source trace and existing duplicate tests |
| Constant/stale HR | Unresolved data-source assumption | Equal values at different timestamps remain. They can reflect integer resolution, true stability or stale values; export lacks freshness/quality contract |
| Counter resets / crossing boundaries | Verified correct behavior under provisional counter interpretation | Independent masks reproduce 83,927 original adjacent pairs; 80 eight-lookback step rates agree. Rejection/reset breaks adjacency; no whole increment assigned across a boundary |
| Steps imply independently validated activity | Unresolved data-source assumption | A provisional cumulative-counter interpretation supports recorded differences, not ground-truth activity; only five well-supported zero-change bins remain |
| Garmin Unix seconds get an extra timezone shift | Verified correct behavior | Original `ts` parsed unchanged; text clock discrepancy flagged, never corrected for alignment. `normalize.py::normalize_chunk` |
| Garmin `ts` is acquisition time | Unresolved data-source assumption | API/SDK/version and collector timestamp/batch code absent; generic Garmin docs cannot identify this export |
| Many HR observations imply representative coverage | Scientifically plausible source of overinterpretation | Reading-weighted summaries describe observed readings; quality tables show gaps and missing windows. Two readings do not establish window coverage |
| Full 438-feature set fixes measurement mismatch | Scientifically plausible explanation, not established root cause | Previous matched results remain weak; added summaries cannot create missing events or resolve a recall boundary |
| Current wake answer available at cutoff | Verified unavailable as a prospective input | Current response contains the answer; candidate Morning windows remain retrospective. No new current-answer predictor |
| Previous completed EMA equals last prompt | Verified unsupported substitution | Exact wording says last prompt; complete delivered/unanswered prompt logs absent |
| Earlier survey start establishes answer availability | Verified unsupported substitution | No submission/receipt/version clock. Real history remains blocked, not assigned zero accuracy |
| Train/test leakage through scaling or IDs | Verified correct behavior in audited fits | Saved training scalers/encoders, grouped inner/outer membership and 185 estimator replays; only explicit feature allowlist |
| Model code cannot recover signal | Verified correct behavior on synthetic data | Actual grouped linkage/feature/model positive control recovers injected held-out relationship; this does not validate real measurements |
| Composite masks different item associations | Scientifically plausible, not confirmed prediction gain | Prespecified item-level ordinal and binary models; nervousness has a small descriptive adjusted within-person association, no established held-out HR increment |
| 53.5% within-person variation is wearable-predictable | Scientifically unsupported inference | Sums-of-squares partition does not identify its source: recall variation, unmeasured events, measurement error and physiology all remain possible |
| Pooling OOF scores is equivalent to within-fold ranking | Verified incorrect equivalence | Different fits have different prevalence/calibration; report pooled and within-fold pair-weighted AUROC separately; do not normalize with validation outcomes |

No new confirmed implementation defect was found in the current audited pipeline. Consequently no
prior scientific bundle was corrected or silently replaced. This is a bounded audit, not proof that
every source record or every possible implementation path is error-free. Original HR-source traces,
all-row score/clock checks, eight-window HR/step arithmetic, existing comprehensive-feature tests and
new likelihood/gating tests provide complementary coverage. In particular, code-level acceptance
correctness is distinct from a validated physiological acceptance rule.

Feature rules inspected: `garmin_hr_activity/features.py` (acceptance/counter pairs),
`garmin_comprehensive/features.py` (moments, shape gates, original-adjacent changes, observed-minute
summaries, five-minute rolling summaries, counter rates, coobserved HR/steps),
`garmin_comprehensive/extract.py` (multiscale/history and feature assembly),
`garmin_temporal_personalization/data.py` (past-only references), and their existing tests.
No new expanded physiological features enter Notebook 18: four established 60-minute HR summaries
and supported step rate are sufficient for its controlled target diagnostic.
