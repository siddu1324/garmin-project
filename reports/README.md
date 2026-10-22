# Research reports and notebooks

Start with [Notebook 21](../notebooks/21_central_research_overview.ipynb). [Notebook 20](../notebooks/20_complete_pipeline_for_review.ipynb) contains the detailed canonical-code review.

| Notebook | Focus | Report / specification |
|---|---|---|
| [00](../notebooks/00_data_inventory_and_understanding.ipynb) | data inventory and understanding | [Read](inventory_report.md) |
| [01](../notebooks/01_preprocessing_and_ibi_eligibility.ipynb) | preprocessing and ibi eligibility | [Read](ibi_eligibility_policies.md) |
| [02](../notebooks/02_hrv_input_validation.ipynb) | hrv input validation | [Read](hrv_input_specification.md) |
| [03](../notebooks/03_ibi_variability_pilot.ipynb) | ibi variability pilot | [Read](ibi_pilot/78af22e7f0ead75b/report.md) |
| [04](../notebooks/04_hr_activity_features.ipynb) | hr activity features | [Read](hr_activity/7fdba29ad8d6f0fe/report.md) |
| [05](../notebooks/05_ema_outcome_baseline.ipynb) | ema outcome baseline | [Read](ema_baseline/d933d3999d949477/report.md) |
| [06](../notebooks/06_hr_support_and_loss.ipynb) | hr support and loss | [Read](hr_support_loss/838b9efbf4bfd731/report.md) |
| [07](../notebooks/07_recall_alignment_and_model_diagnostics.ipynb) | recall alignment and model diagnostics | [Read](recall_alignment/824551e40b8e301d/report.md) |
| [08](../notebooks/08_morning_recall_window_comparison.ipynb) | morning recall window comparison | [Read](morning_recall/b1fd2e61d20cdbe4/report.md) |
| [09](../notebooks/09_window_information_and_nonlinear_benchmark.ipynb) | window information and nonlinear benchmark | [Read](window_benchmark/e3e34a2ea014f38f/report.md) |
| [10](../notebooks/10_within_person_signal_and_past_hr_reference.ipynb) | within person signal and past hr reference | [Read](past_hr_reference/d021427887616868/report.md) |
| [11](../notebooks/11_measurement_source_and_timestamp_audit.ipynb) | measurement source and timestamp audit | [Read](measurement_source/1c44cb036f4ba779/report.md) |
| [12](../notebooks/12_pss_classification.ipynb) | pss classification | [Read](pss_classification/README.md) |
| [13](../notebooks/13_comprehensive_physio_features.ipynb) | comprehensive physio features | [Read](comprehensive_features/README.md) |
| [14](../notebooks/14_comprehensive_pss_regression.ipynb) | comprehensive pss regression | [Read](comprehensive_regression/README.md) |
| [15](../notebooks/15_binary_pss_6hour_experiments.ipynb) | binary pss 6hour experiments | [Read](binary_6hour/README.md) |
| [16](../notebooks/16_binary_stress_improvement.ipynb) | binary stress improvement | [Read](binary_improvement/README.md) |
| [17](../notebooks/17_temporal_personalized_stress.ipynb) | temporal personalized stress | [Read](temporal_personalization/README.md) |
| [18](../notebooks/18_root_cause_diagnostics.ipynb) | root cause diagnostics | [Read](root_cause_analysis/README.md) |
| [19](../notebooks/19_measurement_alignment.ipynb) | measurement alignment | [Read](measurement_alignment/README.md) |
| [20](../notebooks/20_complete_pipeline_for_review.ipynb) | complete pipeline for review | [Read](complete_pipeline_review/README.md) |
| [21](../notebooks/21_central_research_overview.ipynb) | central research overview | [Read](central_research_overview/README.md) |

## Current result pointers

- [All 678 candidate features and missingness](central_research_overview/complete_feature_inventory.csv).
- [Actual encoded full-logistic inputs by fold](central_research_overview/actual_model_inputs_by_fold.csv).
- [Matched classification metrics](complete_pipeline_review/classification_metrics.csv), [class support](complete_pipeline_review/support.csv), [regression metrics](complete_pipeline_review/regression_metrics.csv).
- [Full feature engineering presentation](comprehensive_features/index.html), [regression presentation](comprehensive_regression/index.html), [measurement audit](measurement_alignment/index.html).

## Reading historical results

Reports are preserved in their original version directories. Use each family's README and supersession record for its current version; a folder's existence does not establish validation.

- [IBI window correction](window_supersession.json): original 12-/24-hour counts are obsolete; corrected immediate-through-six-hour counts are unchanged.
- [Feature checkpoint supersession](comprehensive_features/supersession.json).
- [Regression supersession](comprehensive_regression/supersession.json).
- [Measurement-audit draft supersession](measurement_alignment/supersession.json): use `e5359ea4cc4b4efa`; the other two versions are preserved unvalidated drafts.

Earlier notebooks retain their originally requested output policy (00–11 source-only; 12–21 reviewed aggregate outputs). Download HTML presentations and open locally. Individual predictions, memberships, raw inputs and fitted estimators are private and excluded. The source ZIP/unpacked copy referenced by Notebook 20 is locally generated, excluded as redundant, and reproducible from canonical code.

The [publication record](repository_publication/README.md) lists repository verification and exclusions. No report result was recomputed by a model search for publication.
