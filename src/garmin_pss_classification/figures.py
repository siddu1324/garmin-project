"""Readable aggregate-only plots; no participant records or identifiers."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'dummy': '#707070', 'logistic': '#2980b9', 'xgboost': '#d35400', 'lightgbm': '#16836b'}
NAMES = {'dummy': 'Majority', 'logistic': 'Logistic', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM'}
VIEWS = {'context': 'Context', 'context_hr': 'Context + HR', 'context_hr_steps': 'Context + HR + steps',
         'context_hr_steps_history': '+ earlier HR reference', 'context_missingness': 'Context + missingness'}


def read(folder, name):
    return pd.DataFrame(json.loads((folder / f'{name}.json').read_text()))


def make_figures(folder):
    out = folder / 'figures'
    out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 130, 'savefig.dpi': 160})
    m = read(folder, 'metrics')
    def save(fig, name):
        fig.savefig(out / f'{name}.png', bbox_inches='tight', facecolor='white')
        plt.close(fig)
    d = read(folder, 'class_distribution')
    d = d[(d.task == 'three_class') & (d.survey_category == 'All') & (d.label != 'missing')]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(['Low: 3–4', 'Middle: 5–9', 'High: 10–15'], d.responses, color=['#5499c7', '#7f8c8d', '#bb5566'])
    ax.bar_label(bars, labels=[f'{n:,} ({n/d.responses.sum():.1%})' for n in d.responses], padding=4)
    ax.set(ylim=(0, d.responses.max()*1.18), ylabel='Development responses', title='Exploratory classes in the three-item composite')
    ax.text(0, -.22, '17 missing outcomes retained in accounting, excluded from fitting. No clinical cutoffs.', transform=ax.transAxes)
    save(fig, '01_class_distribution')

    primary = m[(m.experiment == 'three_class__60m__joint__complete') & (m.weighting == 'none')].copy()
    primary['label'] = [NAMES[r.model] + (' · ' + VIEWS.get(r.view, '') if r.model != 'dummy' else '') for r in primary.itertuples()]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(primary.label, primary.macro_f1, color=[COLORS[x] for x in primary.model])
    for y, v in enumerate(primary.macro_f1):
        ax.text(v+.005, y, f'{v:.3f}', va='center')
    ax.set(xlim=(0, max(.5, primary.macro_f1.max()+.08)), xlabel='Pooled out-of-fold macro F1',
           title='Primary 60-minute complete-case benchmark · unweighted')
    ax.invert_yaxis()
    ax.text(0, -.15, 'Identical training and validation membership: 1,355 responses / 113 participants.', transform=ax.transAxes)
    save(fig, '02_primary_models')

    cms = read(folder, 'confusion_matrices')
    fig, axes = plt.subplots(4, 2, figsize=(10, 14))
    for row, model in enumerate(['dummy', 'logistic', 'xgboost', 'lightgbm']):
        for col, weight in enumerate(['none', 'balanced']):
            selected = cms[(cms.experiment == 'three_class__60m__joint__complete') & (cms.model == model)
                           & (cms.view == ('dummy' if model == 'dummy' else 'context_hr_steps'))
                           & (cms.weighting == ('none' if model == 'dummy' else weight))].iloc[0]
            cm = np.asarray(selected.confusion)
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[row, col]
            ax.imshow(norm, vmin=0, vmax=1, cmap='Blues')
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{cm[i,j]}\n{norm[i,j]:.0%}', ha='center', va='center', color='white' if norm[i,j] > .55 else 'black')
            ax.set(xticks=range(3), yticks=range(3), xticklabels=['Low', 'Middle', 'High'], yticklabels=['Low', 'Middle', 'High'],
                   xlabel='Predicted class', ylabel='Observed class', title=NAMES[model] + ' · ' + ('unweighted' if col == 0 else 'balanced*'))
    fig.suptitle('60-minute complete cases · context + HR + steps\nCounts and row percentages; *majority dummy remains unweighted', y=1.01)
    fig.tight_layout()
    save(fig, '03_confusion_matrices')

    per = read(folder, 'per_class')
    per = per[(per.experiment == 'three_class__60m__joint__complete') & (per.view == 'context_hr_steps')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    labels = []
    for ax, metric in zip(axes, ['precision', 'recall', 'f1']):
        array = []
        labels = []
        for model in ['logistic', 'xgboost', 'lightgbm']:
            for weight in ['none', 'balanced']:
                array.append(per[(per.model == model) & (per.weighting == weight)].sort_values('label')[metric].to_numpy())
                labels.append(NAMES[model] + ' / ' + weight)
        a = np.array(array)
        ax.imshow(a, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
        for i in range(len(a)):
            for j in range(3):
                ax.text(j, i, f'{a[i,j]:.2f}', ha='center', va='center', color='white' if a[i,j] > .6 else 'black')
        ax.set(xticks=range(3), xticklabels=['Low', 'Middle', 'High'], yticks=range(len(labels)), yticklabels=labels, title=metric.title())
    fig.suptitle('Per-class performance: weighting changes the trade-off', y=1.04)
    fig.tight_layout()
    save(fig, '04_per_class')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for row, handling in enumerate(['complete', 'imputed']):
        for col, weight in enumerate(['none', 'balanced']):
            q = m[(m.task == 'three_class') & (m.cohort == 'joint') & (m.handling == handling) & (m.weighting == weight)]
            ax = axes[row, col]
            for model in ['logistic', 'xgboost', 'lightgbm']:
                for view, style in [('context_hr_steps', '-'), ('context', '--')]:
                    g = q[(q.model == model) & (q.view == view)].sort_values('minutes')
                    ax.plot(range(len(g)), g.macro_f1, style, marker='o' if style == '-' else None,
                            color=COLORS[model], label=NAMES[model] + (' + wearable' if style == '-' else ' context'))
            ax.set(xticks=range(8), xticklabels=['5', '15', '30', '60', '120', '360', '720', '1440'],
                   xlabel='Lookback (minutes)', ylabel='OOF macro F1', title=handling.title() + ' / ' + weight)
            ax.grid(axis='y', alpha=.2)
    axes[0, 1].legend(fontsize=8, ncol=2)
    fig.suptitle('Fixed lookbacks: native complete cohorts differ; imputed cohorts share all responses', y=1.01)
    fig.tight_layout()
    save(fig, '05_lookbacks')

    differences = read(folder, 'paired_differences')
    q = differences[(differences.experiment == 'three_class__60m__joint__imputed') &
                    (differences.view == 'context_hr_steps') & (differences.comparison.isin(['imputed_vs_complete_matched_validation', 'missingness_only']))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, comparison, title in zip(axes, ['imputed_vs_complete_matched_validation', 'missingness_only'],
                                    ['Imputed minus complete models\nsame complete validation responses; training differs', 'Physiological values plus flags minus flags alone\nall valid-outcome responses; same training cohort']):
        g = q[q.comparison == comparison]
        y = np.arange(len(g))
        ax.hlines(y, g.difference_low, g.difference_high, color='#29688c')
        ax.scatter(g.macro_f1_difference, y, color='#29688c')
        ax.axvline(0, color='gray', linestyle='--')
        ax.set(yticks=y, yticklabels=[NAMES[r.model] + ' / ' + r.weighting for r in g.itertuples()],
               xlabel='Difference in macro F1 (conditional 95% interval)', title=title)
    fig.tight_layout()
    save(fig, '06_missingness_matched')

    imp = read(folder, 'importance')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
    for ax, handling in zip(axes, ['complete', 'imputed']):
        q = imp[(imp.handling == handling) & (imp.view == 'context_hr_steps') & (imp.weighting == 'none')]
        features = sorted(q.feature.unique())
        for offset, model in [(-.15, 'xgboost'), (.15, 'lightgbm')]:
            g = q[q.model == model].copy()
            g['value'] = g.macro_f1_decreases.map(np.mean)
            stats = g.groupby('feature').value.agg(['mean', 'std']).reindex(features)
            ax.errorbar(stats['mean'], np.arange(len(features))+offset, xerr=stats['std'], fmt='o', color=COLORS[model], label=NAMES[model], capsize=2)
        ax.axvline(0, color='gray', linestyle='--')
        ax.set(yticks=range(len(features)), yticklabels=[x.replace('__60m','').replace('_',' ') for x in features],
               title=handling.title() + ' · unweighted', xlabel='Decrease in validation macro F1')
        ax.legend()
    fig.suptitle('Descriptive permutation importance at 60 minutes\nFive repeats per fold; bars show between-fold SD, not confidence intervals', y=1.03)
    fig.tight_layout()
    save(fig, '07_permutation_importance')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes, ['macro_f1', 'elevated_recall'], ['Macro F1', 'High-class recall']):
        modes = [('complete','none'), ('complete','balanced'), ('imputed','none'), ('imputed','balanced')]
        for i, model in enumerate(['logistic', 'xgboost', 'lightgbm']):
            vals = []
            for handling, weighting in modes:
                row = m[(m.task == 'three_class') & (m.minutes == 60) & (m.cohort == 'joint') & (m.handling == handling) & (m.weighting == weighting) & (m.view == 'context_hr_steps') & (m.model == model)].iloc[0]
                vals.append(row[metric])
            ax.bar(np.arange(4)+(i-1)*.24, vals, width=.24, color=COLORS[model], label=NAMES[model])
        ax.set(xticks=range(4), xticklabels=['Complete\nNone', 'Complete\nBalanced', 'Imputed\nNone', 'Imputed\nBalanced'],
               ylabel=title, title=title)
        ax.legend(fontsize=8)
    fig.suptitle('60-minute sensitivities · complete n=1,355, imputed n=3,689\nFull-cohort scores mix changes in training and evaluation population', y=1.08)
    fig.tight_layout()
    save(fig, '08_weighting_and_handling')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    make_figures(parser.parse_args().folder)
