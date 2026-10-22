# Notebook 16 verification

Version `911ce0ce6a7b6943` uses the unchanged supported-feature bundle and frozen development folds.
All 3,706 responses remain accounted for; 3,689 valid-score/context responses form the main cohort.
The final-test participants were not evaluated. No raw sensor extraction or scientific threshold
correction was needed; original artifacts remain unchanged.

| Check | Executed result |
| --- | --- |
| Full test suite | 213 passed; 13 dependency warnings |
| Ruff | Passed |
| Completed outer fits | 640 (600 primary + 40 added context controls), no failed/nonconverged selected fits |
| Candidate evaluations in this checkpoint | 5,290 × three inner folds (15,870 fits); includes 180 reused decisions and 30 fixed negative controls |
| Nonconverged candidates | 0 |
| Exact Notebook 15 prediction reproductions | 180 training/validation model pairs |
| Saved validation/training replays | 640 each, tolerance 1e−12 |
| Independent fresh selected-configuration refits | 8 |
| Inner participant separation checks | 1,920 |
| Resampling lineage records checked | 419,760, all donors inside the corresponding training split |
| Original analysis preserved | All 401,654 original prediction views and every original metric/paired contrast unchanged |
| Matched context-control checks | 40; identical outer/inner memberships, class counts and base-donor sequences |
| Target and independent clock checks | 3,706 each |
| Independent saved window-boundary checks | 74,120 |
| Initial source reading traces | 45 windows from nine EMA keys, one participant source |
| Broader source verification | 36 windows from nine distinct development participants; reading and observed-minute means independently replayed |
| Prior files/private manifests preserved | 785 / 2,019 |
| Raw/reference metadata integrity | 7,870 size/mtime checks; no raw contents reread |

New tests also verify that the four matched context resampling procedures preserve labels and donor-context coordinates exactly.

New tests validate numeric convex interpolation, donor provenance, balanced counts, same-person/
same-category conditional sampling, preservation of cyclic context, training-only compact selection,
threshold choice, analytic prior correction, independent threshold-metric arithmetic, and a held-out synthetic participant positive control.
Existing tests cover offset conversion, precision buffers, long-window overflow, counter resets,
left-inclusive/right-exclusive boundaries, missingness, conflicts, rejected adjacency and feature exclusions.

The source audit uses independently parsed Python datetimes, not the vectorized window generator.
The broader trace uses boolean timestamp masks rather than trusting stored slice indices. No real
participant rows or survey clocks are included in the public notebook/report. Conditional SMOTE donor checks confirm participant
and category equality; numeric-only SMOTE explicitly records cross-person/category mixing and
imputed-donor use. Bounds checks establish arithmetic consistency, not physiological validity.

Thresholds use selected-configuration inner OOF predictions only. Reusing inner predictions for
model and threshold selection can overfit training decisions; outer outcomes never choose either.
Analytic prior correction uses training prevalences only. It is not fitted calibration and can alter
pooled ranking through fold-specific transformations despite preserving within-fold ranks.

All matched differences require identical EMA keys and labels. The 2,000-draw participant bootstrap
conditions on saved OOF predictions; it excludes retuning uncertainty and multiplicity correction.
Three training-label permutation runs are computational controls, not repeated-measures permutation
significance tests. The largest observed model AUROC is descriptive after repeated exploration.

Notebook execution, browser inspection and the final privacy scan are recorded in `completion.json`
when complete. The notebook reads aggregate results only. Prior validators and seals are unchanged;
the checkpoint-local validator permits only the reviewed new aggregate notebook outputs.

Individual accounting, memberships, augmentation donors, estimators and predictions are ignored/private.
No final-test evaluation, commit, push or provider message was performed.
