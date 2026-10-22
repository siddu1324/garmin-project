# Notebook 18 verification

| Check | Result |
|---|---|
| Development accounting | All 3,706 rows, original 125 participants and five folds |
| Matched stress cohort | 1,355 complete observed responses / 113 participants |
| Original source trace | 18,153 HR readings from 10 development sources; 320 feature values; no mismatch |
| Original EMA answer checks | Six deterministic development source rows, plus one complete raw-to-held-out-prediction trace |
| Clock / score arithmetic | All 3,706 development responses independently checked |
| Corrected window boundaries | 5,436 scalar datetime comparisons across eight lookbacks, including overflow edges and 24 hours |
| Additional independent features | 320 eight-window HR values and 80 step rates |
| Counter handling | 83,927 pair rejection masks independently reconstructed; resets/rejected endpoints/boundaries preserved |
| Model execution | 185 completed outer fits; no failed jobs; binary three-group inner validation only |
| Prediction replays | 185 training + 185 held-out replays; tolerance 1e−12 |
| Fresh deterministic fits | Three, covering binary, ordinal and continuous models |
| Participant separation | 315 inner membership checks; unchanged outer folds; explicit predictor allowlist |
| Full tests | 231 passed, 13 dependency warnings |
| Ruff | Passed across src and tests |
| Notebook | Eight aggregate-only code cells executed; no errors |
| Presentation | Nine sections, seven embedded figures; desktop/mobile checks; 15 private screenshots; no external asset requests |
| Preservation | 789 prior public/source hashes and 2,873 private manifest hashes unchanged |
| Raw/reference integrity | 7,870 size/mtime checks; selected original source contents also hashed |
| New real recall-aligned / EMA-history predictions | Zero: necessary evidence is absent |
| Final-test evaluation | None |

New tests cover independent HR arithmetic, interval unions/crossings, missing counter observations
versus true zero change, rejected endpoints, strict recall availability, approved evidence hashes,
participant/partition identity, frozen cutoffs, ordinal gradients/likelihood/recovery, dense-versus-profile
random-intercept arithmetic, grouped synthetic predictive recovery, training-only transforms and
sealed-source mutation rejection. Existing source/feature/window/privacy tests remain unchanged.

The original-answer audit streams source CSV containers to locate selected development rows.
Only selected development answers are compared or used in diagnostics. The internal legacy-style
`final_test_values_parsed: 0` audit field counts selected diagnostic/test rows, not low-level parsing
of CSV containers. No final-test outcomes are summarized, evaluated, selected for fitting or exported.

Privacy scans check known identifiers and source names in public text and notebook outputs;
browser review inspects aggregate-only figures. These checks do not prove anonymity of arbitrary
future artifacts. Individual predictions, clocks, source rows, cohort memberships and fitted models
are ignored/private. Final exact checksums, privacy file count and row counts are in completion.json.
No raw modifications, provider message, commit or push.
