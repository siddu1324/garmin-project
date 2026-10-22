# Notebook 17 verification

| Check | Result |
|---|---|
| Development accounting | 3,706 rows; 3,689 valid-score/context model responses; original 125 development participants |
| Outcome/cutoff replay | All 3,706; raw offset verified in minutes, precision buffer unchanged |
| Completed outer fits | 210; no failed jobs |
| Nested candidates | 620 evaluations × three participant-grouped inner folds |
| Nonconverged candidates | 0 |
| Saved estimator replays | 210 validation and 210 training predictions; tolerance1e−12 |
| Fresh independent refits | Six, including both targets and logistic/boosting/physiology-only examples |
| Inner separation checks | 630; memberships identical across matched feature/model comparisons |
| Cached input hashes | 250 HR/counter partition files |
| Independent ledger calculations | Eight participant source traces, Python loops independent of feature functions |
| Preserved previous artifacts | 848 prior public/source files; 2,661 prior private manifests |
| Raw/reference integrity | 7,870 size/mtime checks; no raw archive modification or re-extraction |
| Full test suite | 221 passed; 13 dependency warnings |
| Ruff | Passed |
| Online EMA-history predictions | Zero: unavailable clocks rejected, not fabricated |
| Final test | Not evaluated |

New synthetic tests check strict receipt-before-cutoff boundaries, delayed/missing answer clocks,
start ties, wrong-person exclusion, current/future-label perturbations, chronological population fitting,
five-answer calibration, history arithmetic, UTC-clock matching across midnight, partial counter
boundaries, rejected/reset pairs, support thresholds and an injected-signal positive control through
actual participant-held-out model code. Existing clock, window-overflow, accepted-reading, score,
assignment and missingness tests remain intact.

Independent feature checks use actual cached accepted readings and counter pairs; they verify arithmetic
and recorded-clock containment, not upstream acquisition/quality semantics. Feature imputation is trained
inside splits and never imputes outcomes or sensor records. Missingness-only/quality models are separate.

The public report contains only aggregate metrics and synthetic examples. Participant metrics, sequence
keys, exact event times, models and predictions are ignored/private. The notebook execution, browser
review, final privacy scan and checksum seals are recorded in completion.json. No commit or push.
