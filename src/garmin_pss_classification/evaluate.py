"""Aggregate only reporting from fixed, participant-held-out predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_inventory.privacy import AuditError
from garmin_preprocessing.common import verify_bundle, write_json
from .data import TASKS, cohort_mask, experiments
from .metrics import paired_bootstrap, score

KEYS = ['experiment', 'view', 'model', 'weighting']


def records(frame):
    # JSON round-trip converts nullable/numpy scalars; only aggregate frames use this.
    return json.loads(frame.to_json(orient='records', double_precision=10))


def collect(run, config):
    pieces, settings, training, importance, skips = [], [], [], [], []
    for spec in experiments(config):
        for fold in range(5):
            folder = run / 'jobs' / f"{spec['experiment']}__fold{fold}"
            manifest = verify_bundle(folder)
            base = {k: v for k, v in spec.items() if k != 'views'} | {'fold': fold}
            if (folder / 'skipped.json').exists():
                skips.append(base | json.loads((folder / 'skipped.json').read_text()))
                continue
            p = pd.read_parquet(folder / 'predictions.parquet')
            if p.n_fold.nunique() != 1 or p.n_fold.iloc[0] != fold:
                raise AuditError('Prediction fold mismatch.')
            member = pd.read_parquet(folder / 'validation_membership.parquet')
            train = pd.read_parquet(folder / 'train_membership.parquet')
            if set(member.n_participant) & set(train.n_participant):
                raise AuditError('Outer split is not participant-separated.')
            for _, group in p.groupby(KEYS):
                if group.n_ema_id.duplicated().any() or set(group.n_ema_id) != set(member.n_ema_id):
                    raise AuditError('Models do not share validation membership.')
            pieces.append(p)
            for name, sink in [('settings', settings), ('training', training), ('importance', importance)]:
                sink.extend([base | row for row in json.loads((folder / f'{name}.json').read_text())])
            if manifest['training_responses'] != len(train) or manifest['validation_responses'] != len(member):
                raise AuditError('Membership manifest counts differ.')
    return pd.concat(pieces, ignore_index=True), settings, training, importance, skips


def metric_rows(p, specs):
    pooled, folds, classes, matrices, surveys = [], [], [], [], []
    for key, group in p.groupby(KEYS, sort=True):
        meta = dict(zip(KEYS, key)) | {k: specs[key[0]][k] for k in ['task', 'minutes', 'cohort', 'handling']}
        k = TASKS[meta['task']]
        if group.n_ema_id.duplicated().any():
            raise AuditError('Repeated OOF responses.')
        def measure(g):
            return score(g.y.to_numpy(int), g.prediction.to_numpy(int),
                         g[[f'p{i}' for i in range(k)]].to_numpy(float), g.n_participant.to_numpy(), k)
        values, cm, per = measure(group)
        counts = dict(responses=len(group), participants=group.n_participant.nunique(),
                      outer_folds=group.n_fold.nunique(), full_oof=group.n_fold.nunique() == 5)
        pooled.append(meta | counts | values)
        matrices.append(meta | dict(confusion=cm.tolist(), true_axis='rows', predicted_axis='columns'))
        for c in range(k):
            classes.append(meta | dict(label=c, support=int(per['support'][c]),
                precision=float(per['precision'][c]), recall=float(per['recall'][c]), f1=float(per['f1'][c])))
        for fold, g in group.groupby('n_fold'):
            folds.append(meta | dict(fold=int(fold), responses=len(g), participants=g.n_participant.nunique()) | measure(g)[0])
        for category, g in group.groupby('n_survey_category'):
            surveys.append(meta | dict(survey_category=category, responses=len(g), participants=g.n_participant.nunique()) | measure(g)[0])
    return {'metrics': pooled, 'fold_metrics': folds, 'per_class': classes, 'confusion_matrices': matrices,
            'survey_metrics': surveys}


def comparisons(p, specs, config):
    grouped = {key: g.set_index('n_ema_id').sort_index() for key, g in p.groupby(KEYS)}
    output = []
    pairs = set()
    for key in grouped:
        exp, view, model, weight = key
        if model == 'dummy':
            continue
        bases = [('dummy', (exp, 'dummy', 'dummy', 'none'))]
        if model != 'logistic':
            bases.append(('logistic', (exp, view, 'logistic', weight)))
        if view != 'context':
            bases.append(('context', (exp, 'context', model, weight)))
        if view == 'context_hr_steps':
            bases.append(('incremental_steps', (exp, 'context_hr', model, weight)))
        if view == 'context_hr_steps_history':
            bases.append(('incremental_history', (exp, 'context_hr_steps', model, weight)))
        if view == 'context_hr_steps' and specs[exp]['handling'] == 'imputed':
            bases.append(('missingness_only', (exp, 'context_missingness', model, weight)))
        if weight == 'balanced':
            bases.append(('class_weighting', (exp, view, model, 'none')))
        if specs[exp]['handling'] == 'imputed':
            bases.append(('imputed_vs_complete_matched_validation', (exp.replace('__imputed', '__complete'), view, model, weight)))
        for reason, base in bases:
            if base in grouped:
                pairs.add((key, base, reason))
    for key, base, reason in sorted(pairs):
        a, b = grouped[key], grouped[base]
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        if not len(a):
            continue
        if not np.array_equal(a.y, b.y) or not np.array_equal(a.n_participant, b.n_participant) or not np.array_equal(a.n_fold, b.n_fold):
            raise AuditError('Paired comparison has mismatched labels, identities or folds.')
        meta = dict(zip(KEYS, key)) | dict(comparison=reason, baseline_experiment=base[0],
            baseline_view=base[1], baseline_model=base[2], baseline_weighting=base[3],
            responses=len(a), participants=a.n_participant.nunique(),
            same_training_population=key[0] == base[0])
        output.append(meta | paired_bootstrap(a.y.to_numpy(int), a.prediction.to_numpy(int),
            b.prediction.to_numpy(int), a.n_participant.to_numpy(), TASKS[specs[key[0]]['task']],
            config['bootstrap_draws'], config['bootstrap_seed']))
    return output


def cohort_accounting(frame, config):
    rows, distributions = [], []
    for task in TASKS:
        for category, group in [('All', frame), *list(frame.groupby('n_survey_category'))]:
            for label in [*range(TASKS[task]), 'missing']:
                chosen = group[group['target_' + task].isna()] if label == 'missing' else group[group['target_' + task] == label]
                distributions.append(dict(task=task, survey_category=category, label=label,
                    responses=len(chosen), participants=chosen.n_participant.nunique()))
    for spec in experiments(config):
        mask = cohort_mask(frame, spec)
        for included in [True, False]:
            subgroup = frame[mask == included]
            for fold, f in [('All', subgroup), *list(subgroup.groupby('n_fold'))]:
                for category, g in [('All', f), *list(f.groupby('n_survey_category'))]:
                    y = g['target_' + spec['task']]
                    rows.append({k: v for k, v in spec.items() if k != 'views'} | dict(included=included,
                        fold=str(fold), survey_category=category, responses=len(g), participants=g.n_participant.nunique(),
                        valid_scores=int(y.notna().sum()), missing_scores=int(y.isna().sum()),
                        pss_mean=float(g.pss_combo.mean()) if g.pss_combo.notna().any() else None,
                        pss_median=float(g.pss_combo.median()) if g.pss_combo.notna().any() else None,
                        **{f'class_{c}': int((y == c).sum()) for c in range(TASKS[spec['task']])}))
    return rows, distributions


def common_lookback(p, frame, config):
    specs = [s for s in experiments(config) if s['task'] == 'three_class' and s['cohort'] == 'joint' and s['handling'] == 'complete']
    common = set(frame.n_ema_id)
    for spec in specs:
        common &= set(frame.loc[cohort_mask(frame, spec), 'n_ema_id'])
    mask = p.experiment.isin([s['experiment'] for s in specs]) & p.n_ema_id.isin(common)
    result = metric_rows(p[mask], {s['experiment']: s for s in specs})['metrics']
    return [r | dict(comparison_scope='common validation responses only; native training cohorts differ') for r in result]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    config = json.loads((run / 'prefit_specification.json').read_text())['config']
    out = root / 'reports/pss_classification' / run.name
    if out.exists():
        raise AuditError('Aggregate output exists; refusing to overwrite.')
    p, settings, training, importance, skips = collect(run, config)
    frame = pd.read_parquet(run / 'development.parquet')
    specs = {s['experiment']: s for s in experiments(config)}
    out.mkdir(parents=True)
    tables = metric_rows(p, specs)
    print('Scored aggregate/fold metrics.', flush=True)
    tables['paired_differences'] = comparisons(p, specs, config)
    print('Computed paired participant-bootstrap intervals.', flush=True)
    tables['cohort_accounting'], tables['class_distribution'] = cohort_accounting(frame, config)
    tables['common_lookback_metrics'] = common_lookback(p, frame, config)
    tables.update(settings=settings, training=training, importance=importance, skipped=skips)
    for name, rows in tables.items():
        write_json(out / f'{name}.json', records(pd.DataFrame(rows)))
    write_json(out / 'configuration.json', config)
    write_json(out / 'packages.json', json.loads((run / 'prefit_specification.json').read_text())['packages'])
    p.to_parquet(run / 'out_of_fold_predictions.parquet', index=False)
    print('Aggregate reports:', out.relative_to(root), flush=True)


if __name__ == '__main__':
    main()
