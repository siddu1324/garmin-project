"""Aggregate-only report and figures; never render participant rows."""
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle

PUBLIC_TABLES = ['outcomes', 'questionnaire_evidence', 'cohorts', 'fold_counts', 'metrics',
                 'paired_intervals', 'availability', 'quality_diagnostics', 'recording_status', 'coefficients']
MODEL_ORDER = ['mean', 'median', 'context', 'hr', 'context_hr', 'context_hr_steps']


def table(frame):
    def cell(x):
        if pd.isna(x):
            return '—'
        if isinstance(x, float):
            return f'{x:.3f}'
        if isinstance(x, int):
            return f'{x:,}'
        return str(x)
    return ('| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n'
            + '\n'.join('| ' + ' | '.join(cell(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)))


def figures(build, tables, outcomes):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    cohorts = tables['cohorts']
    metrics = tables['metrics']
    intervals = tables['paired_intervals']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    native = cohorts.loc[cohorts.scope.eq('native')]
    for block, label in [('hr', 'Four HR summaries'), ('hr_steps', 'HR + supported steps')]:
        f = native.loc[native.block.eq(block)]
        axes[0].plot(f.lookback_minutes.astype(str), f.responses, 'o-', label=label)
    axes[0].axhline(3706, color='gray', ls=':', label='All development EMAs')
    axes[0].set(xlabel='Lookback (minutes; categorical spacing)', ylabel='Responses with complete model inputs', title='Availability before performance')
    axes[0].legend(fontsize=8)
    d = pd.DataFrame(outcomes['distribution']).groupby('pss_combo').responses.sum()
    axes[1].bar(d.index, d.values, color='#387d91')
    axes[1].set(xlabel='Three-item composite (3–15)', ylabel='Development responses', title='Observed scores; 17 outcomes missing', xticks=list(range(3, 16)))
    fig.savefig(build / 'coverage_and_scores.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for r, scope in enumerate(['native', 'common']):
        for c, block in enumerate(['hr', 'hr_steps']):
            f = metrics.loc[metrics.scope.eq(scope) & metrics.block.eq(block) & metrics.fold.eq('pooled')]
            for name in MODEL_ORDER:
                m = f.loc[f.model.eq(name)].sort_values('lookback_minutes')
                if len(m):
                    axes[r, c].plot(m.lookback_minutes.astype(str), m.mae, 'o-', ms=3, label=name)
            axes[r, c].set(title=f'{scope.title()} cohort: {block}', ylabel='Pooled OOF MAE (score points)', xlabel='Lookback (minutes)')
            axes[r, c].legend(fontsize=7, ncol=2)
    fig.suptitle('Compare models within panels; native lookbacks contain different responses')
    fig.savefig(build / 'matched_performance.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, block, model, baseline in [(axes[0], 'hr', 'context_hr', 'context'),
                                      (axes[1], 'hr_steps', 'context_hr_steps', 'context_hr')]:
        f = intervals.loc[intervals.scope.eq('native') & intervals.block.eq(block)
                          & intervals.model.eq(model) & intervals.baseline.eq(baseline)].sort_values('lookback_minutes')
        ax.errorbar(f.lookback_minutes.astype(str), f.mae_difference,
                    yerr=[f.mae_difference - f.mae_difference_low, f.mae_difference_high - f.mae_difference], fmt='o', capsize=3)
        ax.axhline(0, color='gray', ls=':')
        ax.set(title=f'{model} minus {baseline}', xlabel='Lookback (minutes)', ylabel='Paired MAE difference; negative favors added inputs')
    fig.suptitle('Native matched cohorts; 95% participant bootstrap intervals of fixed predictions')
    fig.savefig(build / 'paired_differences.png', dpi=150)
    plt.close(fig)
    q = tables['quality_diagnostics']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, measures, title in [
        (axes[0], ['occupied_minute_fraction', 'supported_pair_fraction'], 'Density / supported step fraction'),
        (axes[1], ['latest_age_seconds', 'maximum_gap_including_edges_seconds'], 'HR freshness and gaps (seconds)'),
        (axes[2], ['observed_span_seconds', 'supported_pair_seconds'], 'HR span / supported steps (seconds)')]:
        for measure in measures:
            f = q.loc[q.block.eq('hr_steps') & q.cohort.eq('matched_model_cohort') & q.measure.eq(measure)].sort_values('lookback_minutes')
            ax.plot(f.lookback_minutes.astype(str), f['median'], 'o-', label=measure.replace('_seconds', '').replace('_', ' '))
            ax.fill_between(range(len(f)), f.q25, f.q75, alpha=.13)
        ax.set(title=title, xlabel='Lookback (minutes)')
        if ax != axes[0]:
            ax.set_yscale('log')
        ax.legend(fontsize=7)
    fig.suptitle('Separate diagnostics in the secondary native cohort: median and interquartile range')
    fig.savefig(build / 'recording_support.png', dpi=150)
    plt.close(fig)


def build_report(root, bundle):
    root, bundle = Path(root), Path(bundle)
    verify_bundle(bundle)
    inputs = {'run_manifest_sha256': sha256(bundle / 'manifest.json'), 'report_code_sha256': sha256(Path(__file__))}
    identifier = digest_object(inputs)[:16]
    parent = root / 'reports/ema_baseline'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / identifier
    if destination.exists():
        verify_bundle(destination)
        return destination
    build = Path(tempfile.mkdtemp(prefix='.building_', dir=parent))
    for name in PUBLIC_TABLES:
        shutil.copyfile(bundle / (name + '.json'), build / (name + '.json'))
    outcomes = json.loads((build / 'outcomes.json').read_text())
    tables = {name: pd.DataFrame(json.loads((build / (name + '.json')).read_text()))
              for name in PUBLIC_TABLES if name not in ['outcomes', 'questionnaire_evidence']}
    figures(build, tables, outcomes)
    summaries = []
    for scope in ['native', 'common']:
        for block in ['hr', 'hr_steps']:
            m = tables['metrics']
            m = m.loc[m.scope.eq(scope) & m.block.eq(block) & m.fold.eq('pooled')]
            pivot = m.pivot(index='lookback_minutes', columns='model', values='mae')
            pivot = pivot[[c for c in MODEL_ORDER if c in pivot]]
            c = tables['cohorts']
            c = c.loc[c.scope.eq(scope) & c.block.eq(block)].set_index('lookback_minutes')
            joined = c[['responses', 'participants']].join(pivot).reset_index()
            summaries.append(f'### {scope.title()} {block} cohort: pooled MAE\n\n' + table(joined))
    i = tables['paired_intervals']
    selected = i.loc[i.scope.eq('native') & ((i.block.eq('hr') & i.model.eq('context_hr') & i.baseline.eq('context'))
                                           | (i.model.eq('context_hr_steps') & i.baseline.eq('context_hr')))]
    delta_table = table(selected[['block', 'lookback_minutes', 'mae_difference', 'mae_difference_low', 'mae_difference_high']])
    a = tables['availability']
    a = a.loc[a.survey_category.eq('all') & a.lookback_minutes.isin([5, 1440]),
              ['block', 'lookback_minutes', 'availability', 'responses', 'valid_outcomes', 'pss_mean']]
    report = f'''# Notebook 05 — exploratory EMA outcome linkage and HR/activity baseline

**These fixed, simple models show no useful incremental predictive gain from the tested wearable
summaries.** The training-median dummy has lower pooled MAE than every Ridge model in every native
and common comparison. Adding HR to context gives tiny, inconsistent changes; all eight primary
native intervals include zero. Adding supported steps increases native pooled MAE at every lookback.
This describes the specified models and available recordings, not absence of any physiological
relationship. No final lookback is selected.

All **3,706 development EMAs / 125 participants / five original folds** remain linked. All 32 final-test
participants remain sealed. Results are **exploratory prediction from recorded pre-EMA data, not
validated real-time prediction**: acquisition versus receipt/batching and counter semantics remain
provisional. No feature support threshold changed after viewing outcomes.

## What was scored and linked

The three raw export fields are `PSS_NERVOUS`, `PSS_COPE`, `PSS_CONTROL`. All three blank questionnaires
confirm stored codes 1–5 paired with display labels 0–4. Morning questionnaire names end in `_1`,
Endday in `_4`, and Midday has no suffix; the corrected raw exports use the unsuffixed fields in all
categories. `questionnaire_evidence.json` records item names, pages, option codes and document hashes.

For complete, valid integer-coded items only, score = nervous + (6−cope) + (6−control).
This is **the study's three-item perceived-stress composite**, range 3–15, not a full standardized PSS,
a diagnosis or a clinically validated cutoff. No binary targets were made. Morning asks since waking;
the other questionnaires refer to since the last prompt. Different reference periods limit a single
instantaneous-stress interpretation.

**3,689 scores are available; 17 responses remain missing.** The observed mean is 6.407, sample SD
2.696, median 6, range 3–15. All observed missing item cells are empty strings (47 cells across 17
responses); there are no nonmissing invalid codes. The PDFs do not define exporter missing sentinels.
The conservative token exclusions in the frozen specification are therefore assumptions, not claimed
survey documentation; they do not affect any observed nonblank value in this dataset.

{table(pd.DataFrame(outcomes['summary']))}

Original item strings, parsed values, per-item statuses, outcome reason, source hash/row, actual local
start, stable EMA key and original folds remain in the ignored linked table. A one-to-one join checks
both key sets and participant/category/clock/fold agreement. Incomplete outcomes are retained but not
fit. The existing feature and separate quality tables are preserved byte-for-byte.

## The comparison fixed before fitting

[The frozen specification](../../ema_baseline_specification.md) fixes all models, cohorts, metrics
and bootstrap settings. Primary inputs are HR mean, sample SD, latest and slope. The secondary
block adds supported step additions. Context is survey category plus sine/cosine of actual local
start time, separate from wearable inputs. Floors, all other sensors, quality columns, identifiers
and current survey answers never enter the design matrix.

Within each block/lookback, mean and median dummies and all Ridge models train and validate on the
**same complete responses**. The secondary cohort refits the HR/context/dummy models on its smaller
matched cohort. Ridge alpha=1, with a training-only numerical scaler and category encoder, no tuning,
imputation, missingness predictors or prediction clipping. Participant folds are frozen; no person's
responses enter both sides of a fit. StandardScaler uses population variance; source HR SD uses N−1.

Native HR cohorts range from 1,395 responses / 117 participants at five minutes to 3,426 / 123 at
24 hours. Secondary cohorts range from 625 / 104 to 3,207 / 123. Common intersections across every
lookback retain **1,395 / 117 for HR** and **625 / 104 for HR+steps**. These exceed the prespecified
minimum descriptive-size flags, with 22–25 and 20–21 validation participants per fold respectively.
They are selected recording-available subsets, not representative samples. Common-cohort training
AND validation membership is held constant across lookbacks. All 160 fold/cohort fits were feasible.

![Availability and score distribution](coverage_and_scores.png)

## Matched performance

MAE is in composite-score points; lower is better. Each row is a matched comparison. Native rows at
different lookbacks contain different responses and must not be ranked as if they share a test set.
`mean`/`median` denote matching training-outcome dummies. The unchanged common-cohort dummy/context
values across lookbacks are an additional matching check.

{chr(10).join(summaries)}

`metrics.json` includes all pooled and fold-level **MAE, RMSE and equally weighted participant MAE**;
`fold_counts.json` supplies every training/validation response and participant count. At five minutes
on the primary cohort, context+HR MAE/RMSE/participant-MAE are **2.092 / 2.548 / 2.220**, versus the
matching median dummy **2.052 / 2.563 / 2.181**. This illustrates that optimizing mean-squared loss
can modestly improve RMSE while losing on the prespecified primary MAE. At 24 hours those primary
context+HR metrics are **2.227 / 2.695 / 2.236**, versus **2.203 / 2.721 / 2.208** for median.

![Matched performance](matched_performance.png)

## Paired uncertainty and wearable increment

Below are native pooled MAE differences: context+HR minus context in the HR block; context+HR+steps
minus context+HR in the secondary block. Negative favors the added inputs. The full
`paired_intervals.json` also compares every Ridge model to matching mean/median/context baselines,
and reports participant-averaged differences and intervals for both native and common cohorts.

{delta_table}

Each 95% percentile interval uses 2,000 paired participant-cluster resamples with a fixed seed. All
responses of a sampled participant move together. Predictions are fixed; models are not refit during
bootstrapping. These are conditional development-prediction intervals, not independent-response
uncertainty, external validity, causal effects, or multiplicity-adjusted significance. Some small step
loss intervals exclude zero, but the many correlated comparisons do not justify a selected winner.

![Paired differences](paired_differences.png)

## Availability, missingness and recording support

The available/unavailable comparison below is descriptive; it did not select rules. Counts before
outcome filtering differ from the model cohort counts. The complete table includes all eight
lookbacks and all three survey categories, valid/missing outcomes and score means/medians/SDs.
Participants can appear in both groups at different EMA times.

{table(a)}

`recording_status.json` separates absent recordings from entirely rejected readings.
`quality_diagnostics.json` shows counts, density, freshness, span, internal/edge gaps and step support
for all EMAs and matching model cohorts. These columns are diagnostic only. The Notebook 04
feature-specific exclusion reasons remain available in its separate quality dataset.

In the five-minute secondary cohort, median HR count is 192, span 298 seconds, freshness 0.80 seconds,
and maximum gap including edges 8.16 seconds; median supported step duration is 107 seconds. At
24 hours median HR span is about 20.4 hours, yet only 27.5% of minute bins contain readings, median
freshness is 38.4 minutes and maximum gap including edges is 11.8 hours. Median supported step time
is just 93.2 minutes (6.47% of 24 hours). Large span therefore does not establish continuous coverage.

**Two HR observations do not establish representative window coverage.** The four-feature block
also requires slope's three readings and 60-second span; even that is only a numerical support rule.
A supported zero step increment is observed zero, whereas no supported increment remains missing.
Step sums are supported additions, not whole-window activity. No interpolation, gap filling or
quality-derived predictor was introduced. Overlapping EMA lookbacks are not independent recordings;
no window count is multiplied by its length to claim coverage.

![Separate recording diagnostics](recording_support.png)

## Clearly synthetic teaching example

Imagine a second-resolution actual EMA start at 12:00, with a five-minute interval [11:55,12:00).
Synthetic HR values 60,90,120 at 11:55,11:56,11:57 give mean 90, sample SD 30, latest 120 and slope
30 bpm/min. A synthetic supported step pair 100→110 wholly inside that window contributes 10.
Synthetic answers nervous=4, cope=4, control=2 yield 4+2+4=10. The stable EMA key links that outcome
and its prior measurements. If the person's frozen fold is held out, all their records are excluded
from scaler/encoder/model fitting. A **hypothetical** held-out prediction of 7.5 has absolute error
2.5; it is not a real result or a calculated fit to this teaching row. A minute-resolution 12:00
start instead has cutoff 11:59. Neither changing the cutoff nor using the current answers as inputs
is allowed. Missing cope would leave the outcome missing and this response out of fitting.

## Interpretation and next experiment

The numerical pipeline is verified, but physiological/timing validity remains unresolved. More
available long-window data does not establish better measurement or a better prediction horizon.
Complete-case selection, repeated observations, changed native cohorts, unequal survey reference
periods and possible informative missingness limit generalization. Frozen grouped folds estimate
transfer to held-out development participants, not forecasting a participant's future responses.
Ridge is deliberately simple and these features may miss nonlinear or within-person relationships;
this checkpoint provides no authorization or evidence to pursue a more complex model as a remedy.

**Recommended next experiment:** after provider clarification of HR timestamp/batching and step
counter attribution, preregister a measurement-support sensitivity analysis on development data:
compare the same fixed models and matched cohorts under externally justified HR density/gap support
conditions. Select those conditions from device semantics before viewing score associations; retain
the current benchmark as the reference. The current tiny, inconsistent increments do not justify
expanded HRV, a chosen lookback, or final-test evaluation. The provider-question draft remains unsent.

## Outputs and verification

- Linked rows: `{bundle.relative_to(root)}/linked_development.parquet` (3,706).
- Fixed OOF predictions: `{bundle.relative_to(root)}/out_of_fold_predictions.parquet` (46,843 cohort/EMA rows,
  each carrying matched model predictions; repeated comparison views are not distinct EMAs).
- Complete inclusion flags: `{bundle.relative_to(root)}/cohort_membership.parquet` (118,592 rows).
- Private fitted transform audit: `training_audit.json` in that bundle; no serialized executable models.
- Original separate quality: `private/hr_activity/features/76f5885b3955d5b0/quality.parquet`.
- Notebook: `notebooks/05_ema_outcome_baseline.ipynb`. Source output cells stay empty.
- Aggregate JSON tables, four figures and this report: `{destination.relative_to(root)}`.

All prior bundles, assignments and corrected supersession provenance are preserved. Independent
scalar arithmetic verifies every score; independent centered normal equations reproduce all 880
model fits within 9×10⁻¹⁴ score points. Tests, execution, privacy and input-integrity checks are
recorded in the separate checkpoint verification report. No final-test outcomes/sensors, expanded
HRV, binary targets, tuning, imputation, feature selection, commit or push.
'''
    (build / 'report.md').write_text(report)
    publish_bundle(build, destination, {'version': identifier, 'inputs': inputs, 'aggregate_only': True})
    return destination
