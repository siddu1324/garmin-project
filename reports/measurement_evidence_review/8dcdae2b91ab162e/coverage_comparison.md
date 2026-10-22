# Existing coverage comparison — no after-evidence rebuild

These are previously published, corrected counts, **not a new eligibility run**. All denominators
are 3,706 development responses and 125 participants. The 48 rows below cover all three frozen
policies, both interval hypotheses and eight lookbacks. Survey-stratified source aggregates are
preserved in `existing_coverage_by_survey.json`. Unique supported minutes were already calculated
using within-participant unions; overlapping windows/EMA views are not counted as independent time.

No newly reviewed measurement source is available. After-evidence counts and changes are **not
computed**, rather than zero. No policy is designated best because it retains more observations.
Screen can pass with a 120-second run; these eligibility counts are not full five-minute HRV counts.

| Policy | Hypothesis | Lookback min | Eligible windows | EMAs | People | Unique supported min |
| --- | --- | --- | --- | --- | --- | --- |
| conservative | interval_end | 5 | 0 | 0 | 0 | 0.000 |
| conservative | interval_end | 15 | 0 | 0 | 0 | 0.000 |
| conservative | interval_end | 30 | 0 | 0 | 0 | 0.000 |
| conservative | interval_end | 60 | 0 | 0 | 0 | 0.000 |
| conservative | interval_end | 120 | 0 | 0 | 0 | 0.000 |
| conservative | interval_end | 360 | 2 | 1 | 1 | 5.855 |
| conservative | interval_end | 720 | 2 | 1 | 1 | 5.855 |
| conservative | interval_end | 1440 | 6 | 3 | 1 | 6.011 |
| conservative | interval_start | 5 | 0 | 0 | 0 | 0.000 |
| conservative | interval_start | 15 | 0 | 0 | 0 | 0.000 |
| conservative | interval_start | 30 | 0 | 0 | 0 | 0.000 |
| conservative | interval_start | 60 | 0 | 0 | 0 | 0.000 |
| conservative | interval_start | 120 | 0 | 0 | 0 | 0.000 |
| conservative | interval_start | 360 | 0 | 0 | 0 | 0.000 |
| conservative | interval_start | 720 | 0 | 0 | 0 | 0.000 |
| conservative | interval_start | 1440 | 0 | 0 | 0 | 0.000 |
| screen | interval_end | 5 | 4 | 4 | 4 | 18.980 |
| screen | interval_end | 15 | 102 | 39 | 32 | 249.404 |
| screen | interval_end | 30 | 225 | 63 | 45 | 503.175 |
| screen | interval_end | 60 | 536 | 122 | 59 | 1,139.973 |
| screen | interval_end | 120 | 1,319 | 240 | 74 | 2,746.100 |
| screen | interval_end | 360 | 4,413 | 559 | 98 | 7,310.267 |
| screen | interval_end | 720 | 9,096 | 813 | 103 | 9,840.962 |
| screen | interval_end | 1440 | 18,907 | 1,350 | 104 | 10,597.928 |
| screen | interval_start | 5 | 5 | 5 | 5 | 23.690 |
| screen | interval_start | 15 | 85 | 32 | 27 | 213.565 |
| screen | interval_start | 30 | 168 | 53 | 36 | 403.906 |
| screen | interval_start | 60 | 388 | 99 | 51 | 892.242 |
| screen | interval_start | 120 | 1,015 | 193 | 69 | 2,158.051 |
| screen | interval_start | 360 | 3,398 | 465 | 94 | 5,755.291 |
| screen | interval_start | 720 | 7,016 | 693 | 97 | 7,717.801 |
| screen | interval_start | 1440 | 14,647 | 1,192 | 98 | 8,381.359 |
| timing_sensitivity | interval_end | 5 | 20 | 20 | 16 | 93.340 |
| timing_sensitivity | interval_end | 15 | 349 | 116 | 58 | 792.483 |
| timing_sensitivity | interval_end | 30 | 830 | 187 | 70 | 1,681.200 |
| timing_sensitivity | interval_end | 60 | 1,958 | 317 | 88 | 3,722.307 |
| timing_sensitivity | interval_end | 120 | 4,319 | 527 | 98 | 8,165.622 |
| timing_sensitivity | interval_end | 360 | 14,297 | 1,042 | 114 | 20,960.856 |
| timing_sensitivity | interval_end | 720 | 28,126 | 1,375 | 117 | 27,399.556 |
| timing_sensitivity | interval_end | 1440 | 57,478 | 2,104 | 117 | 29,484.528 |
| timing_sensitivity | interval_start | 5 | 22 | 22 | 17 | 102.586 |
| timing_sensitivity | interval_start | 15 | 351 | 121 | 60 | 808.648 |
| timing_sensitivity | interval_start | 30 | 830 | 194 | 72 | 1,709.162 |
| timing_sensitivity | interval_start | 60 | 1,971 | 328 | 83 | 3,793.804 |
| timing_sensitivity | interval_start | 120 | 4,288 | 526 | 98 | 8,177.243 |
| timing_sensitivity | interval_start | 360 | 14,172 | 1,053 | 112 | 20,954.617 |
| timing_sensitivity | interval_start | 720 | 27,979 | 1,378 | 116 | 27,409.339 |
| timing_sensitivity | interval_start | 1440 | 57,224 | 2,122 | 116 | 29,565.020 |

## Existing additional restriction: one run with both span and support ≥240 seconds

The following census is separate from the old calculation sample of ten runs per hypothesis.
It still describes provisional support, not normal beats or a complete five-minute recording.
`existing_pilot_feasibility.json` preserves original window-view/overlap/age aggregates. These values
precede HR/outcome intersection and do not establish matched-cohort feasibility.

| Hypothesis | Lookback min | Qualifying EMAs | People | Window views | Unique supported min |
| --- | --- | --- | --- | --- | --- |
| interval_end | 5 | 1 | 1 | 1 | 4.849 |
| interval_end | 15 | 6 | 6 | 11 | 34.123 |
| interval_end | 30 | 10 | 10 | 19 | 56.132 |
| interval_end | 60 | 20 | 19 | 41 | 112.426 |
| interval_end | 120 | 40 | 29 | 125 | 315.396 |
| interval_end | 360 | 110 | 43 | 494 | 969.866 |
| interval_end | 720 | 184 | 48 | 1,120 | 1,436.881 |
| interval_end | 1440 | 334 | 51 | 2,447 | 1,528.677 |
| interval_start | 5 | 1 | 1 | 1 | 4.848 |
| interval_start | 15 | 3 | 3 | 8 | 20.410 |
| interval_start | 30 | 8 | 8 | 16 | 45.449 |
| interval_start | 60 | 12 | 12 | 24 | 68.952 |
| interval_start | 120 | 22 | 17 | 71 | 191.102 |
| interval_start | 360 | 73 | 32 | 276 | 592.035 |
| interval_start | 720 | 145 | 38 | 634 | 861.798 |
| interval_start | 1440 | 262 | 40 | 1,325 | 940.849 |

Immediate eligibility is insufficient for a five-fold predictive comparison: only one response/person
per hypothesis meets this additional restriction. Earlier runs may permit a feasibility study, but
counts after matching HR/outcomes and original folds are unknown here. One acceptable segment in
24 hours is not complete 24-hour coverage. Hypothesis samples overlap and their counts cannot be added.

Sources: [current corrected coverage](../../ibi_eligibility/974e98729579967b_8ac2c64795c6/coverage_table.md)
and [corrected pilot census](../../ibi_pilot/78af22e7f0ead75b/report.md).
The old pilot execution configuration references superseded dependencies; citing its independently
corrected census does not authorize rerunning that configuration. Use current provenance-checked
loaders for any future rebuild.
