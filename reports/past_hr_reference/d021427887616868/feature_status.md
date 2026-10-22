| feature | status | meaning_or_limit |
| --- | --- | --- |
| Continuous composite | Implemented, unchanged | Three valid stored 1–5 codes required; no clinical/binary cutoff |
| Context + raw mean/SD/latest/slope | Implemented and evaluated | Current 60-minute observed readings; provisional timing |
| Earlier-recording reference + deviations | Implemented and evaluated here | 24-hour history excluding current hour; ≥30 minutes and ≥3 UTC hours |
| Supported steps; other existing HR summaries | Implemented earlier; outside this experiment | No additional activity or HR feature expansion here |
| Whole-record person means/centering | Retrospective diagnostics only | Never predictive features or saved person-level means |
| Other history lengths/support thresholds | Untested candidates | Would require a separate specification; no automatic search |
| Equal-minute current HR summaries | Untested predictive candidate | Notebook 09 descriptive weighting audit only |
| Resting HR / complete daily physiological baseline | Blocked by measurement evidence | Requires validated rest/activity/wear and temporal support, not this reference |
| Expanded HRV / normal-beat measures | Blocked by measurement evidence | IBI anchoring, missing beats and beat-quality status unresolved |
| Recall-aligned prompt intervals | Blocked where logs unavailable | Previous completed EMA is not previous prompt |
| Garmin stress, imputation, binary outcomes | Excluded by design | Not authorized predictors/transformations here |
