# Notebook 19 — briefing for Will

**We have precise exported start clocks, but not verified sensor acquisition or delivered-prompt clocks.**
All available Path A diagnostics were executed; no new real predictive models were fit.

- All 3,706 development responses have actual/scheduled starts: 1,868 minute-resolution Morning/Endday and 1,838 second-resolution Midday.
- Median actual−scheduled delay: Morning 10.00 min; Midday 6.39 min; Endday 5.00 min. Starts >60 min after schedule: 140, 209 and 127 respectively. No missing starts among these exported responses; unanswered prompts are not observable from this denominator.
- At 60 min, 1,725 responses have any accepted HR; at 24 h, 3,464. Median 24-h occupied-minute fraction is 22.9%. Longer availability is not a verified recall interval or continuous coverage.
- HR and steps show roughly 1.05-second median participant cadence. Step pairs are overwhelmingly increasing; gaps, repeated values and fractional clocks cannot identify callback/batch/acquisition semantics. 884 negative counter pairs remain breaks, not negative activity.
- Availability/label differences are descriptive and affected by survey/time composition and sparse strata. Participant-bootstrap standardized intervals use only 672/2,000 valid draws at ≥5 and 1,067/2,000 at ≥10; interpret cautiously. Tables retain crude and standardized results rather than claiming independence.
- The fixed ±60 s support sensitivity preserves stored clocks and excludes original post-cutoff readings in every scenario. No model performance selects a shift.

Morning recall is since waking; Midday/Endday is since the last prompt. Current-response waking is retrospective.
A previous completed response is not a delivery log; Endday does not mean all day. See the exact questionnaire
citations in source_contract.md. The minute precision buffer remains 60 s; second-resolution starts are unbuffered.

**All 3,706 prospective recall and history gates remain BLOCKED.** The reviewed-hash importer, conservative
boundary/feature extractor, original-adjacency counter handling, exact-version history bridge and fixed matched
logistic runner are implemented and synthetic-tested. No real metadata have been invented. The history-model
schedule remains to be reviewed if independent version availability becomes available.

Notebook 18, on the same 1,355-response matched cohort, found context versus context+HR nervousness AUROC
0.4875 versus 0.5107; paired difference +0.0233 (95% interval −0.0210 to +0.0663). No established HR increment.
Do not compare scores across different cohorts. The 53.5% within-person score variance does not establish
wearable predictability. The limited IBI variability pilot was separate from HR/activity prediction and did not
establish generally usable normal-beat HRV. The activity probe had only five inactive versus 591 active bins.

**Recommendation:** obtain one development-only original payload/log slice plus collector/exporter code and
complete prompt/answer-version semantics. If the gates open, execute the frozen A–E matched logistic comparison;
do not expand the model search. The provider request is a draft only.

Outputs: `reports/measurement_alignment/e5359ea4cc4b4efa/` (aggregate tables/figures), `private/measurement_alignment/runs/e5359ea4cc4b4efa/` (ignored diagnostic rows and exact source catalog),
`notebooks/19_measurement_alignment.ipynb`, this directory's self-contained index.html, source_contract.md and
provider_data_request.md. Previous artifacts/folds/raw files are preserved; final test remains sealed.
