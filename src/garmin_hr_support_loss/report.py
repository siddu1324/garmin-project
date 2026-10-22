"""Aggregate-only Notebook 06 figures and an explicitly exploratory interpretation."""
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from garmin_ema_baseline.report import table
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle, write_json

TABLES = ['cohorts', 'fold_counts', 'composition', 'quality_diagnostics', 'metrics',
          'paired_intervals', 'retraining_intervals', 'prediction_variability', 'occupancy_verification']


def figures(build, tables):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    c = tables['cohorts']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for scope, ax in zip(['native', 'common'], axes):
        for policy in ['A', 'B', 'C']:
            f = c.loc[c.scope.eq(scope) & c.policy.eq(policy)].sort_values('lookback_minutes')
            ax.plot(f.lookback_minutes.astype(str), f.responses, 'o-', label=policy)
        ax.set(title=scope.title() + ' cohort', xlabel='Lookback (minutes; categorical spacing)', ylabel='Retained responses', yscale='log')
        ax.legend(title='Support policy')
    fig.suptitle('B/C common cohort: only 22 responses from 7 participants — descriptive only')
    fig.savefig(build / 'retention.png', dpi=150)
    plt.close(fig)
    i = tables['paired_intervals']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for r, stage in enumerate(['saved_subset', 'support_refit']):
        for col, baseline in enumerate(['median', 'context']):
            ax = axes[r, col]
            for j, policy in enumerate(['A', 'B', 'C']):
                f = i.loc[i.stage.eq(stage) & i.scope.eq('native') & i.policy.eq(policy)
                          & i.model.eq('context_hr') & i.baseline.eq(baseline)].sort_values('lookback_minutes')
                x = list(range(len(f)))
                shifted = [v + (j - 1) * .18 for v in x]
                ax.errorbar(shifted, f.mae_difference,
                            yerr=[f.mae_difference - f.mae_difference_low, f.mae_difference_high - f.mae_difference],
                            fmt='o', ms=3, capsize=2, label=policy)
                ax.set_xticks(x, f.lookback_minutes.astype(str))
            ax.axhline(0, color='gray', ls=':')
            ax.set(title=stage + ': context+HR minus ' + baseline, xlabel='Lookback (minutes)', ylabel='MAE difference (negative favors HR model)')
            ax.legend(title='Policy', fontsize=8)
    fig.suptitle('Native paired differences; 24-hour B/C cohorts are small; 95% conditional intervals')
    fig.savefig(build / 'support_comparisons.png', dpi=150)
    plt.close(fig)
    m = tables['metrics']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for scope, ax in zip(['native', 'common'], axes):
        for name in ['median', 'context_hr', 'q_context_hr']:
            f = m.loc[m.stage.eq('loss_only') & m.scope.eq(scope) & m.survey_category.eq('all')
                      & m.fold.eq('pooled') & m.model.eq(name)].sort_values('lookback_minutes')
            ax.plot(f.lookback_minutes.astype(str), f.mae, 'o-', label=name)
        ax.set(title=scope.title() + ' unchanged A cohort', xlabel='Lookback (minutes)', ylabel='Pooled OOF MAE (score points)')
        ax.legend(fontsize=8)
    fig.suptitle('Median regression mostly reproduces the matching median dummy')
    fig.savefig(build / 'loss_comparison.png', dpi=150)
    plt.close(fig)
    q = tables['quality_diagnostics']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, measure, label in zip(axes, ['maximum_gap_including_edges_seconds', 'observed_span_seconds', 'latest_age_seconds'],
                                  ['Maximum internal/edge gap', 'Accepted-reading timestamp span', 'Freshness']):
        for policy in ['A', 'B', 'C']:
            f = q.loc[q.scope.eq('native') & q.policy.eq(policy) & q.group.eq('retained') & q.measure.eq(measure)].sort_values('lookback_minutes')
            ax.plot(f.lookback_minutes.astype(str), f['median'], 'o-', label=policy)
            ax.fill_between(range(len(f)), f.q25, f.q75, alpha=.1)
        ax.set(title=label, xlabel='Lookback (minutes)', ylabel='Seconds (log scale)', yscale='log')
        ax.legend(title='Policy', fontsize=8)
    fig.suptitle('Median and interquartile range: occupied bins are not continuous support')
    fig.savefig(build / 'recording_diagnostics.png', dpi=150)
    plt.close(fig)


def build_report(root, bundle):
    root, bundle = Path(root), Path(bundle)
    manifest = verify_bundle(bundle)
    inputs = {'run_manifest': sha256(bundle / 'manifest.json'), 'report_source': sha256(Path(__file__))}
    identifier = digest_object(inputs)[:16]
    parent = root / 'reports/hr_support_loss'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / identifier
    if destination.exists():
        verify_bundle(destination)
        return destination
    build = Path(tempfile.mkdtemp(prefix='.building_', dir=parent))
    for name in TABLES:
        shutil.copyfile(bundle / (name + '.json'), build / (name + '.json'))
    tables = {name: pd.DataFrame(json.loads((build / (name + '.json')).read_text())) for name in TABLES if name != 'occupancy_verification'}
    audits = json.loads((bundle / 'training_audit.json').read_text())
    fits = []
    for a in audits:
        if a['stage'] != 'loss_only':
            continue
        for f in a['models']:
            fits.append({**{k:a[k] for k in ['scope','lookback_minutes','fold']},
                         **{k:v for k,v in f.items() if k in ['model','converged','solver_iterations','warning_categories',
                            'nonzero_coefficients_at_1e_minus_8','training_pinball_plus_penalty','validation_prediction_sd',
                            'validation_prediction_minimum','validation_prediction_maximum','validation_unique_predictions_8dp',
                            'validation_equal_training_median_fraction']}})
    write_json(build / 'quantile_fit_diagnostics.json', fits)
    figures(build, tables)
    cohorts, metrics, intervals = tables['cohorts'], tables['metrics'], tables['paired_intervals']
    native = cohorts.loc[cohorts.scope.eq('native')].pivot(index='lookback_minutes', columns='policy', values=['responses','participants'])
    native.columns = ['_'.join(c) for c in native.columns]
    m = metrics.loc[metrics.fold.eq('pooled') & metrics.survey_category.eq('all')]
    support = m.loc[m.scope.eq('native') & m.policy.isin(['B','C']) & m.stage.isin(['saved_subset','support_refit'])
                    & m.model.isin(['median','context','context_hr'])]
    support = support.pivot(index=['policy','lookback_minutes'],columns=['stage','model'],values='mae')
    support.columns = ['_'.join(c) for c in support.columns]
    loss = m.loc[m.stage.eq('loss_only')].pivot(index=['scope','lookback_minutes'],columns='model',values='mae').reset_index()
    category = metrics.loc[metrics.stage.eq('loss_only') & metrics.scope.eq('native') & metrics.fold.eq('pooled')
                           & metrics.survey_category.ne('all') & metrics.lookback_minutes.isin([5,1440])
                           & metrics.model.isin(['median','context_hr','q_context_hr'])]
    category = category.pivot(index=['lookback_minutes','survey_category'],columns='model',values='mae').reset_index()
    delta = intervals.loc[intervals.stage.eq('support_refit') & intervals.scope.eq('native') & intervals.policy.isin(['B','C'])
                          & intervals.model.eq('context_hr') & intervals.baseline.eq('context'),
                          ['policy','lookback_minutes','mae_difference','mae_difference_low','mae_difference_high']]
    delta_loss = intervals.loc[intervals.stage.eq('loss_only') & intervals.scope.eq('native') & intervals.model.eq('q_context_hr')
                              & intervals.baseline.eq('context_hr'), ['lookback_minutes','mae_difference','mae_difference_low','mae_difference_high']]
    # Full human-readable pooled metric appendix, including small common cohorts and all three metrics.
    appendix = '# All pooled metrics — exploratory; see cohort and fold feasibility flags\n\n'
    for (stage, scope, policy), part in m.groupby(['stage','scope','policy']):
        appendix += f'## {stage}, {scope}, policy {policy}\n\n' + table(part[['lookback_minutes','model','responses','participants','mae','rmse','participant_mae']]) + '\n\n'
    (build / 'pooled_metrics.md').write_text(appendix)
    # Composition appendix makes retained/excluded category and outcome comparisons reviewable without private data.
    composition = tables['composition']
    composition = composition.loc[composition.scope.eq('native') & composition.policy.isin(['B','C']) & composition.group.ne('outside_reference')]
    (build / 'retained_excluded.md').write_text('# Retained / excluded from corresponding A cohort\n\nParticipants may occur in both groups at different responses. No significance tests or threshold selection.\n\n' + table(composition[['policy','lookback_minutes','group','survey_category','responses','participants','pss_mean','pss_median','pss_q25','pss_q75']]))
    text = f'''# Notebook 06 — recording-support and median-loss follow-up

**Denser/more recent recordings did not produce a consistent HR advantage. Median regression
mostly recovered the already competitive median dummy rather than adding HR information.**
This is a **prospectively specified exploratory follow-up informed by Notebook 05**, not an
independent preregistered confirmation. Timing questions remain unresolved: exploratory prediction
from recorded pre-EMA data is not validated real-time prediction. No optimal window or threshold
is declared.

## Questions and rules fixed before new comparisons

[The frozen specification](../../hr_support_loss_specification.md) asks whether denser/recent HR
helps, and whether a median-oriented loss improves MAE relative to squared-error Ridge. Continuous
three-item PSS coding, all eight lookbacks, HR mean/SD/latest/slope, context, original five folds and
all 3,706 linked rows are unchanged. No binary cutoff, imputation or new activity feature.

- **A:** existing complete-case HR support.
- **B:** A plus ≥80% of lookback minute bins containing accepted HR readings.
- **C:** B plus latest accepted reading ≤60 seconds before the cutoff.

These are fixed provisional sensitivity choices, not validated device standards. Existing occupancy
already counts accepted readings after clock/value/conflict rules. All **29,648 HR lookbacks** were
independently checked against the preserved accepted-reading ledger, using [left,right) bin searches
and Python integer offsets. Bins are anchored at cutoff−lookback; an observation at cutoff is
excluded. Features were not changed. Occupied bins do not establish continuous recording.

## Coverage and sample loss

{table(native.reset_index())}

At five minutes, B/C retain 96.0%/95.8% of the A responses. At 24 hours this falls to 1.0%/0.9%:
**35 responses from nine participants for B; 30 from eight for C**. The common intersection across
all eight lookbacks contains **22 responses from seven participants for both B and C**, versus
1,395/117 for A. Those intersections and the 24-hour B/C cohorts are explicitly flagged small;
their results are descriptive and cannot support a reliable lookback comparison. No folds were
redrawn and no threshold relaxed. All folds were numerically feasible, including these small fits.

![Retention](retention.png)

Every cohort/fold's training and validation counts are in `fold_counts.json`. All development EMAs
have explicit inclusion reasons in the private membership table, including those outside the
original complete-case cohort. `composition.json` and the [readable retained/excluded appendix](retained_excluded.md)
report response/participant counts, survey composition and outcome mean/SD/median/quartiles/range.
The denominator for a support loss is the matching A cohort, not the entire development set.
Participants may contribute responses to both retained and excluded groups.

## Saved predictions versus support-cohort refitting

`saved_subset` filters validation predictions saved by Notebook 05; its training population and
matching dummies/context models remain unchanged. `support_refit` retrains all five comparators on
the policy cohort inside each frozen fold. Within each comparison, training and validation membership
is identical across models. Saved common-subset predictions come from the original common-cohort
fits. These two stages must not be interpreted as the same intervention.

The following native-cohort MAEs show their separation. `context_hr` is Ridge with context and HR.
Every row compares models on the same validation responses. Different rows may contain different
people/responses; lower unpaired error across rows is not evidence of a better policy.

{table(support.reset_index())}

Filtering alone often lowers errors for **all** models, including the dummy. That is an evaluation-
population change, not evidence of stronger HR signal. Refit versus saved contrasts on the exact
same validation rows are in `retraining_intervals.json`. For C at 12 hours, context+HR refitting
changes MAE by −0.053 (95% interval −0.147 to +0.049); uncertainty includes no retraining gain.

The most favorable native context+HR point comparison is C at 12 hours: **MAE 2.013**, versus
matching median **2.024** and context **2.034**. Differences are **−0.012 [−0.141,+0.117]** versus
median and **−0.021 [−0.077,+0.036]** versus context. It retains only **246 responses / 57 participants**,
excluding **2,807 of 3,053** A responses (91.9%). It is an uncertain exploratory exception, not a
selected window or evidence that C is optimal. All native B/C context+HR-versus-context intervals
include zero:

{table(delta)}

![Support comparisons](support_comparisons.png)

Small common-cohort fits sometimes have lower point error than a comparator, but only seven
participants determine those results. [All pooled metrics](pooled_metrics.md) retain those results
with the cohort warnings; they are not used to choose a winner.

## Median-loss baseline, separately on unchanged A cohorts

QuantileRegressor(quantile=.5, alpha=.01, solver='highs') uses the same training-only numerical
scaling and categorical encoding as Ridge. No median model is fitted on B or C. The fixed alphas
are **not comparable penalties**: Ridge has squared loss/L2 regularization; QuantileRegressor has
average pinball loss/L1 regularization. At .5, pinball loss is half absolute error. This experiment
tests this specific loss-plus-penalty choice, not a pure isolated loss effect. Official
[QuantileRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html)
and the installed source are recorded in the specification; scikit-learn is 1.9.1.

All **240 median-regression fits converged**, with no convergence warnings. **234/240** have constant
validation predictions at eight decimal places; six have nonzero coefficient/prediction variation.
The constant predictions equal the matching training median, 6. No alpha was changed to force
variation. The six varying fits occur in common-cohort 60/120-minute HR models and native 24-hour
HR models. `quantile_fit_diagnostics.json` records iterations, objectives, variation and coefficient
counts, while `prediction_variability.json` gives pooled prediction ranges/SDs and dummy agreement.
All fitted objectives are no worse than the feasible constant-training-median solution.

{table(loss)}

For native 5–720-minute cohorts, the quantile models equal the median dummy to numerical precision.
They therefore improve context+HR Ridge MAE by about **0.031–0.049 points**, without an HR-specific
gain. At 24 hours quantile context+HR MAE is **2.212**, versus median **2.203**: difference
**+0.009 [−0.003,+0.022]**. In the common A cohort, varying 60/120-minute quantile models are about
0.005 points worse than median; other windows reproduce the dummy. No quantile context+HR result
provides a convincing advantage beyond the matching median/context-quantile baseline.

Native quantile context+HR minus matched context+HR Ridge differences:

{table(delta_loss)}

![Loss comparison](loss_comparison.png)

`metrics.json` includes pooled and fold **MAE, RMSE and participant-averaged MAE**. At five minutes,
quantile context+HR equals median: MAE 2.052, RMSE 2.563, participant-MAE 2.181; Ridge context+HR
has 2.092, 2.548, 2.220. Better primary MAE does not imply better secondary RMSE.

## Survey-category errors and recording diagnostics

Category errors for representative endpoints are below; all categories/stages/policies/lookbacks
are retained in `metrics.json`. No category-specific model was fitted.

{table(category)}

Morning items ask **since waking**; Midday/Endday refer to **since the last prompt**. Those recall
periods need not equal any of these fixed lookbacks. No waking time or preceding-prompt time was
inferred from completed surveys, and a survey label does not prove an end-of-day recall period.
Category differences are descriptive, not grounds for choosing an outcome definition.

![Recording diagnostics](recording_diagnostics.png)

`quality_diagnostics.json` retains accepted-reading counts, occupancy, freshness, span and maximum
internal/edge gaps for retained/excluded groups. Two readings do not establish representative
coverage, and even an occupied bin can contain only one point. B/C constrain density/freshness,
not physiological continuity, acquisition timing or absence of batching. Quality remains outside
the predictor matrix. No missing data are filled, and overlapping windows are not independent.

## Uncertainty and next experiment

Intervals use **2,000 paired participant-cluster bootstrap resamples** of fixed out-of-fold
predictions. All responses for a sampled participant move together. They omit model-refitting
uncertainty and are not external validation or multiplicity-adjusted tests. Many overlapping
lookbacks, policies, populations and models were examined after Notebook 05; isolated improvements
should not be treated as confirmations. Small-cohort intervals, particularly with seven participants,
are unstable even when numerical fitting is possible.

**Recommendation:** resolve timestamp/batching and intended survey recall periods before expanding
models. If explicit wake/notification times are available and their meaning is confirmed, specify
a development-only recall-aligned HR experiment using the same fixed median/Ridge comparators and
matched cohorts. Do not synthesize those times from previous completed surveys. The current results
do not support threshold/alpha searching, a winning lookback, expanded HRV or opening final-test data.
[Provider questions](../../hr_support_loss_provider_questions_draft.md) are drafted locally only,
not sent. The loss result chiefly explains why the simple median dummy is hard to beat on MAE;
it does not establish that regularization or recording support caused all weak HR results.

## Artifacts and verification

- Frozen specification: `reports/hr_support_loss_specification.md`; config: `config/hr_support_loss.json`.
- Private cohort inclusion: `{bundle.relative_to(root)}/membership.parquet` — {manifest['membership_rows']:,} rows.
- Private predictions: `{bundle.relative_to(root)}/predictions.parquet` — {manifest['prediction_rows']:,} cohort/EMA views.
- Verified support: `{bundle.relative_to(root)}/verified_hr_support.parquet` — 29,648 HR windows.
- Existing linked data remain unchanged at `private/ema_baseline/runs/640600af77404ff6/linked_development.parquet`.
- Notebook: `notebooks/06_hr_support_and_loss.ipynb`; executed copies belong under ignored `outputs/`.
- All aggregate tables, four figures and readable appendices: `{destination.relative_to(root)}`.

All 32 final-test participants remain sealed. Corrected provenance, raw files, previous bundles,
score coding and original assignments/folds are preserved. No new activity features, expanded HRV,
Garmin stress predictors, nonlinear models, imputation, feature selection, commit or push.
The separate verification record documents tests, notebook execution, privacy and integrity checks.
'''
    (build / 'report.md').write_text(text)
    publish_bundle(build, destination, {'version': identifier, 'inputs': inputs, 'aggregate_only': True})
    return destination
