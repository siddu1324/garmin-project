# Complete pipeline review

- Executed notebook: `notebooks/20_complete_pipeline_for_review.ipynb`
- Source archive: `garmin_complete_pipeline_source.zip`
- Canonical code: 40 modules, 7,496 lines embedded verbatim.
- Feature matrix: 3,706 × 678 candidates; 438 supported observed summaries and 240 provisional PulseOx columns.
- Modeling cohort: 3,689 valid-outcome/context responses; 125 development participants; frozen five folds.
- Full logistic inputs: 881 in folds 0–3; 880 in fold 4. These include missingness indicators and context.
- Validation: 6,780 feature values recomputed; 50 classifier and 70 regressor predictions replayed; no new searches.

`feature_dictionary.csv` contains all formulas and support rules. `mapping.csv` and `encoded.csv` trace
all 438 supported columns to each full-logistic fold's actual inputs. Metrics retain checkpoint, target,
feature block and cohort labels. Notebook 19's prospective alignment/history gates remain blocked.

Run the notebook in the authorized project checkout. Raw/private data, assignments, mappings, models,
credentials and local configuration are excluded from the ZIP. `completion.json` records final checksums
and verification; prior notebooks and artifacts are unchanged. No commit or push.
