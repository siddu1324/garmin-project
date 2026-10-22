# Professor briefing — Notebook 07

**Two limitations checked; neither supports declaring a new winning model.**

- Questionnaire evidence establishes Morning = since final waking; Midday and Endday = since
  last prompt. Endday is not automatically an all-day outcome. Fixed lookbacks are imperfect proxies.
- Morning has 931 reported wake clocks, but only 836 positive buffered windows (826 under the
  later-minute sensitivity). All are retrospective: the wake answer comes from the current survey.
  Only 253 nominal windows / 91 participants contain any accepted HR; this is not continuous coverage
  or necessarily sufficient support for all features. Midday/Endday need prompt logs not supplied here.
- All 15 saved five-minute median models predicted 6. In 45 training-only fits at alpha 0/.001/.01,
  HR-only and context-only remain constant even without the penalty. Context+HR has tiny unpenalized
  in-sample improvements in three folds (about .0008–.0044 PSS-composite points). No validation outcome
  was used in this diagnostic; these are not generalization improvements.

**Provider knowledge required:** complete prompt/delivery/expiry logs and intended “last prompt”
interpretation; start/resume/submission event definitions; wake date/offset behavior; Garmin acquisition
versus receipt/batching. Questions are drafted in the report and have not been sent.

**Our next testable choice:** prospectively specify a retrospective Morning-only matched-response
comparison of wake-to-cutoff versus fixed HR windows, subject to four-feature/support feasibility.
Keep it separate from a deployable prediction claim. Do not tune penalties from these training gains.
Provider logs are needed before any defensible Midday/Endday recall-aligned experiment.

The source notebook contains only aggregate displays and synthetic examples. Final-test data remain
sealed, original folds and corrected provenance are preserved, and nothing was committed or pushed.
See [report](report.md), [exact local questionnaire citations](questionnaire_evidence.md), and the
checkpoint verification record for details.
