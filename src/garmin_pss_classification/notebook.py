"""Create the aggregate-only presentation notebook; execution is a separate step."""
import argparse
from pathlib import Path

import nbformat as nbf


def create(root, run_id):
    n = nbf.v4.new_notebook()
    n.metadata = dict(kernelspec=dict(display_name='Python 3 (ipykernel)', language='python', name='python3'),
                      language_info=dict(name='python', version='3.11'),
                      garmin_public_aggregate_notebook=True)
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text))
    md('''# Can recorded HR and activity distinguish perceived-stress categories?
## Notebook 12 · exploratory development classification

**Primary question:** does adding recorded physiology improve participant-held-out classification beyond survey category and local time?

We use the study's **three-item perceived-stress composite (3–15)**, not PSS-10 or PHQ-9.
The thresholds are exploratory, not clinical diagnoses. All 32 final-test participants remain sealed.

This presentation reads only privacy-reviewed aggregate results. Model fitting was executed separately
through the reproducible entrypoint below; rerunning these cells does not refit or touch sensor records.
No participant identifiers, actual timestamps, complete records or individual predictions are displayed.''')
    code(f'''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = Path.cwd() if (Path.cwd() / 'reports').exists() else Path.cwd().parent
RESULTS = ROOT / 'reports/pss_classification/{run_id}'
pd.options.display.float_format = '{{:.3f}}'.format
pd.options.display.max_columns = 15

def table(name):
    return pd.DataFrame(json.loads((RESULTS / f'{{name}}.json').read_text()))

def figure(name):
    display(Image(filename=str(RESULTS / 'figures' / f'{{name}}.png')))

metrics = table('metrics')
differences = table('paired_differences')
print('Aggregate source:', RESULTS.relative_to(ROOT))
display(pd.DataFrame([json.loads((RESULTS / 'packages.json').read_text())]))''')
    md('''### The target and a synthetic example
The verified stored items each use 1–5. Compute `nervous + (6 − cope) + (6 − control)` only when all three are valid.

A **synthetic** response with nervous=4, cope=2, control=3 has score 11 → high class.
A score of 4 maps to low; 5 and 9 to middle; 10 to high. No outcome is imputed.

| Task | Definition | Role |
|---|---|---|
| Three classes | low 3–4 / middle 5–9 / high 10–15 | Primary, eight lookbacks |
| Binary ≥5 | 3–4 versus 5–15 | Secondary, 60 minutes |
| Binary ≥10 | 3–9 versus 10–15 | Secondary, 60 minutes |

Macro F1 gives each class equal weight, but discretization discards differences within a class.
Classification performance cannot be compared numerically with regression MAE.''')
    code("figure('01_class_distribution')\ndisplay(table('class_distribution').query(\"task == 'three_class' and survey_category != 'All'\"))")
    md('''### What was frozen before fitting?
The primary benchmark is **60-minute joint complete cases, unweighted**. The specification was written
before classification performance. It follows earlier regression exploration and is not independent confirmation.

- Existing frozen five outer participant folds; three grouped inner folds within each outer training set.
- Majority dummy; logistic regression (four C values); XGBoost and LightGBM (12 fixed candidates each).
- Select settings by inner macro F1. No outer early stopping, threshold search or calibration.
- Context, +HR, and +HR+steps share the **same training and validation responses** in each cohort.
- All eight lookbacks, balanced training weights and feature-level imputation are specified sensitivities.
- Reference features are an additional 60-minute comparison, using only earlier unlabeled wearable data.

Class weights are recomputed from each training split. Imputation medians, scaling and category encoding
are learned inside each training split. Validation outcomes never choose settings.
See `reports/pss_classification/specification.md` and `config/pss_classification.json` for the complete search.''')
    code("display(table('cohort_accounting').query(\"task == 'three_class' and cohort == 'joint' and included == True and fold == 'All' and survey_category == 'All'\")[[ 'minutes','handling','responses','participants','class_0','class_1','class_2']])")
    md('''### Features and observation limits
HR inputs are existing mean, sample SD, latest value and slope. Steps are supported counter additions,
not assumed activity during unrecorded time. No new physiological features are calculated.

For a **synthetic** 09:22 minute-resolution start, the cutoff is 09:21. A 60-minute window is
[08:21, 09:21). A second-resolution 09:22:35 start uses [08:22:35, 09:22:35).
Missing HR summaries remain missing in complete-case analysis. In the imputed branch, only the model's
feature vector is filled using training medians, with missingness flags; no raw readings are invented.

Two readings do not establish representative coverage. Acquisition versus receipt timing, batching,
activity-counter semantics and mismatch with survey recall periods remain unresolved.''')
    md('''### Primary result: compare models and physiological increments separately
Every bar uses the same 1,355 responses from 113 participants. Higher macro F1 than the dummy does
not by itself show that HR helps: compare each wearable model with its matching context model.''')
    code("figure('02_primary_models')\nprimary = metrics.query(\"experiment == 'three_class__60m__joint__complete' and weighting == 'none'\")\ndisplay(primary[['view','model','macro_f1','balanced_accuracy','accuracy','weighted_f1','roc_auc','equal_participant_macro_f1']])")
    code("display(differences.query(\"experiment == 'three_class__60m__joint__complete' and weighting == 'none' and view == 'context_hr_steps' and comparison in ['dummy','context','logistic','incremental_steps']\")[[ 'model','comparison','macro_f1_difference','difference_low','difference_high','responses','participants']])")
    md('''The intervals resample participants, keeping their responses together. They describe uncertainty
conditional on the saved predictions and this exploratory workflow. They do not repeat tuning or correct
for the many comparisons. A point improvement or an isolated interval is not independent validation.''')
    md('''### Can the high class be detected?
Confusion-matrix rows are observed classes; columns are predictions. A model can improve macro F1
mainly by separating low from middle while still missing nearly every high response.
Balanced fitting changes errors and probabilities; it does not create new evidence or guarantee calibration.''')
    code("figure('03_confusion_matrices')\nfigure('04_per_class')")
    code("display(metrics.query(\"experiment == 'three_class__60m__joint__complete' and view == 'context_hr_steps'\")[[ 'model','weighting','elevated_precision','elevated_recall','elevated_f1','elevated_pr_auc','predicted_elevated_fraction']])")
    md('''### Lookbacks, weighting and missing-data handling
Complete-case lookbacks have different native cohorts. Their scores mix window information with selection.
The imputed branch uses the same valid-score responses across all eight lookbacks. A separate common
complete-case table restricts validation responses only; its training populations still differ.
No lookback or threshold is declared optimal.''')
    code("figure('05_lookbacks')\nfigure('08_weighting_and_handling')")
    code("common = table('common_lookback_metrics')\ndisplay(common.query(\"view == 'context_hr_steps' and weighting == 'none'\")[[ 'minutes','model','responses','participants','macro_f1','elevated_recall']])")
    md('''### Is imputation helping through physiology or availability?
Left: imputed versus complete models scored on identical complete validation responses; training cohorts differ.
Right: physiological values plus missingness versus context plus missingness alone, on identical all-valid cohorts.
A missingness flag represents the recording process, not physiological stress.''')
    code("figure('06_missingness_matched')")
    md('''### Secondary binary and earlier-reference checks
The ≥5 task includes middle and high responses as positive; its positive-class metrics are not high-only detection.
The ≥10 task specifically tests the high threshold. Reference history excludes the current hour and requires
30 occupied minutes over three UTC-hour bins before being available. It is not resting HR.''')
    code("display(metrics.query(\"task != 'three_class' and view == 'context_hr_steps'\")[[ 'task','handling','model','weighting','responses','macro_f1','elevated_precision','elevated_recall','roc_auc']])\ndisplay(differences.query(\"comparison == 'incremental_history'\")[[ 'experiment','model','weighting','responses','macro_f1_difference','difference_low','difference_high']])")
    md('''### Interpretation checks: importance and fold stability
Permutation importance measures the drop in held-out macro F1 when an input is shuffled. Small or negative
values indicate little stable contribution under this perturbation, not proof of biological irrelevance.
Correlated summaries can substitute for each other. No importance result is used to select features.''')
    code("figure('07_permutation_importance')\ntraining = table('training')\ndisplay(training.query(\"experiment == 'three_class__60m__joint__complete' and view == 'context_hr_steps'\")[[ 'fold','model','weighting','training_macro_f1','validation_macro_f1']])")
    code("settings = table('settings')\ndisplay(settings.query(\"experiment == 'three_class__60m__joint__complete' and view == 'context_hr_steps' and weighting == 'none'\")[[ 'fold','model','selected_index','selected_params','converged']])")
    md('''### Findings, limitations and the next justified experiment
The following abstract is generated from the actual saved results. The full report includes matched
bootstrap comparisons, coverage, secondary checks and unresolved measurement questions.''')
    code("display(Markdown((ROOT / 'reports/pss_classification/abstract.md').read_text()))")
    md('''### Verification and reproducibility
The fitted estimators, fold memberships, trial scores and individual predictions stay in ignored private
storage. Public results are aggregate only. No prior notebook, raw file, frozen assignment or final-test
outcome is changed. The notebook intentionally retains only these aggregate execution outputs.

Reproduce fitting from the repository environment:
```bash
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/garmin-mpl OMP_NUM_THREADS=1 .venv/bin/python -m garmin_pss_classification.run
```
Completed immutable fold jobs are verified and reused. Aggregation and figure commands are documented
in `reports/pss_classification/README.md`; public output writers refuse to replace an existing result directory.''')
    code("display(pd.DataFrame([json.loads((RESULTS / 'verification.json').read_text())]).T.rename(columns={0:'result'}))")
    n.cells = cells
    path = root / 'notebooks/12_pss_classification.ipynb'
    if path.exists():
        raise FileExistsError('Notebook exists; preserve prior version.')
    nbf.write(n, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    args = parser.parse_args()
    create(Path.cwd(), args.run_id)
