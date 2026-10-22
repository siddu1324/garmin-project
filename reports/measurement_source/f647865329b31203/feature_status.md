# Feature status

| feature | status | interpretation_limit | required_evidence |
| --- | --- | --- | --- |
| Observed HR mean/median/SD/extrema/quantiles/latest/slope | Implemented; computationally supported, timing provisional | Accepted exported observations only, not continuous or necessarily live physiology | Confirm clocks, units and caching/smoothing |
| Supported step additions; sparse floors | Implemented provisionally; floors not in current models | Only accepted original endpoint pairs; not complete-window activity | Confirm cumulative counter, resets and event interval |
| Earlier-recording HR mean and deviations | Implemented exploratory; no established added predictive value | Unlabeled prior observed UTC-minute means; not resting HR/daily baseline | Confirm event times and observation availability |
| Mean/median IBI, sample SDNN, original-pair RMSSD/pNN50 | Limited 20-row pilot only; arithmetic checked | IBI variability—normal-beat status unverified; ≥240-second runs, not full five-minute HRV | Resolve units, interval order/anchor, missing beats and source quality |
| Expanded physiological HRV, spectral/nonlinear features | Blocked/deferred, not computed here | No proven uninterrupted NN input or appropriate timing support | Original timestamped payload and validated quality/input contract |
| Recall-aligned Morning windows | Implemented retrospective experiment only | Current-survey wake response cannot establish a pre-response boundary | Clarify wake/start clock precision; prospectively available boundary if real-time aim |
| Midday/Endday recall-aligned windows | Blocked by missing prompt-event history | Since last prompt does not mean since last completed survey or whole day | Full prompt delivery/suppression logs and intended recall semantics |
| Calorie/intensity increments and additional sensors | Untested/blocked candidates; no new predictors | Counter units/reset/update semantics or independent scientific specification absent | Confirm exporter definitions before separately specified work |
| Garmin stress and quality columns | Excluded from predictors | Quality remains diagnostic; no automatic expansion | No change authorized by this evidence review |
