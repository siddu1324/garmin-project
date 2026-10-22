# Notebook 15 verification

Run: `11442e75c7e434aa`. All 3,706 development EMAs remain accounted for; 3,689 with valid outcomes/context enter the main matched comparisons (125 participants). The common complete-HR sensitivity contains 1,634 responses from 118 participants. Final-test participants were not evaluated.

| Check | Result |
| --- | --- |
| Full test suite | 198 passed; 13 dependency warnings |
| Ruff | Passed |
| Completed outer model fits | 500; no failed selected refits |
| Saved validation prediction replays | 500, maximum absolute difference 0 |
| Saved training prediction replays | 500, maximum absolute difference 0 |
| Independent deterministic fresh fits | 12 passed |
| Inner participant separation checks | 1,500 passed |
| Candidate configurations assessed | 5,760, each across three inner folds |
| Nonconverged candidate configurations | 11 excluded from selection |
| Prior files and private manifests | 730 files and 1,518 manifests unchanged |
| Notebook 13/14 seals, feature bundle and frozen assignments | Unchanged |
| Raw/reference metadata integrity | 7,870 size/mtime checks passed; raw contents not reread |

The 11 excluded configurations all involved weighted Binary B fits: four original-HR 60-minute, four original-HR 120-minute, and one each activity/full/sparse 120-minute. Every selected outer refit converged. No tuning grid, tolerance, cohort or threshold was changed to rescue a candidate. Eight new tests cover target coding, single-window feature exclusions, fold membership, training-only transformations, filters and metric/bootstrap arithmetic; prior tests remain intact.

Single-window full blocks contain 47 physiological candidates, excluding earlier-history and cross-window features. The 438-feature multiscale sensitivity is separate. Missingness flags are explicit recording-process indicators; the primary full-versus-context contrast therefore does not isolate physiological values alone. The no-flags and missingness-only controls address this limitation.

Confidence intervals use 2,000 paired participant-bootstrap draws conditional on saved OOF predictions. They do not include tuning uncertainty or multiplicity correction. Pooled AUROC compares probabilities across fold-specific models; mean fold AUROC and fold ranges are provided to expose calibration-related ordering differences. Neither changes the frozen primary endpoint.

Notebook 15 reads reviewed aggregate files only. The checkpoint-local privacy validator leaves prior privacy source and seals unchanged. Individual outcomes, participant keys, real timestamps, memberships, predictions and estimators stay in ignored private bundles. Notebook execution, local browser checks and the final privacy scan are recorded in `completion.json` after they finish; this document does not substitute for that seal.

No raw modification, new sensor extraction, final-test evaluation, commit, push or provider message was performed.
