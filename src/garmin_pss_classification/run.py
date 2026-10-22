"""Versioned resumable classification entrypoint. Private folds never enter reports."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (
    digest_object, frozen_assignments, input_guard, publish_bundle, verify_bundle, write_json,
)
from .data import TASKS, cohort_mask, experiments, load_development
from .metrics import macro_f1
from .models import Transform, fit_model, inner_splits, probabilities, tune

REFERENCE = Path('/Users/siddharthareddy/SEA-RESEARCH/HEALTH-WEARABLE-PROJECT/Analysis/sea-garmin')


def fold_job(root_string, run_string, spec, fold, config):
    run = Path(run_string)
    destination = run / 'jobs' / f"{spec['experiment']}__fold{fold}"
    if destination.exists():
        verify_bundle(destination)
        return destination.name, 'reused'
    frame = pd.read_parquet(run / 'development.parquet')
    frame = frame.loc[cohort_mask(frame, spec)].copy()
    train = frame.loc[frame.n_fold != fold].reset_index(drop=True)
    valid = frame.loc[frame.n_fold == fold].reset_index(drop=True)
    target, k = 'target_' + spec['task'], TASKS[spec['task']]
    if set(train.n_participant) & set(valid.n_participant):
        raise AuditError('Participant leakage in outer folds.')
    build = Path(tempfile.mkdtemp(prefix='build-', dir=run))
    meta = dict(spec=spec, fold=fold, run=run.name, training_responses=len(train),
                validation_responses=len(valid), training_participants=train.n_participant.nunique(),
                validation_participants=valid.n_participant.nunique())
    splits = inner_splits(train, target, k, config) if len(train) and len(valid) and train[target].nunique() == k else None
    if splits is None:
        write_json(build / 'skipped.json', {'reason': 'Empty outer cohort or absent class in outer training / fixed inner training or validation.'})
        publish_bundle(build, destination, meta)
        return destination.name, 'skipped'
    train[['n_ema_id', 'n_participant', 'n_fold']].to_parquet(build / 'train_membership.parquet', index=False)
    valid[['n_ema_id', 'n_participant', 'n_fold']].to_parquet(build / 'validation_membership.parquet', index=False)
    inner_membership = np.full(len(train), -1)
    for j, (_, b) in enumerate(splits):
        inner_membership[b] = j
    membership = train[['n_ema_id', 'n_participant']].copy()
    membership['inner_fold'] = inner_membership
    membership.to_parquet(build / 'inner_membership.parquet', index=False)
    y, yv = train[target].to_numpy(int), valid[target].to_numpy(int)
    predictions, settings, training, importance = [], [], [], []

    def add_predictions(model_name, view, weighting, pv, pt):
        out = valid[['n_ema_id', 'n_participant', 'n_fold', 'n_survey_category']].copy()
        out['y'] = yv
        out['prediction'] = pv.argmax(axis=1)
        for c in range(k):
            out[f'p{c}'] = pv[:, c]
        out['experiment'], out['view'], out['model'], out['weighting'] = spec['experiment'], view, model_name, weighting
        predictions.append(out)
        training.append(dict(model=model_name, view=view, weighting=weighting,
            training_macro_f1=macro_f1(y, pt.argmax(axis=1), k),
            validation_macro_f1=macro_f1(yv, pv.argmax(axis=1), k)))

    majority = int(np.argmax(np.bincount(y, minlength=k)))
    add_predictions('dummy', 'dummy', 'none', np.eye(k)[np.full(len(valid), majority)],
                    np.eye(k)[np.full(len(train), majority)])
    with threadpool_limits(limits=1):
        for view in spec['views']:
            prepared = []
            for a, b in splits:
                transform = Transform(view, spec['minutes'], spec['handling']).fit(train.iloc[a])
                prepared.append((transform.transform(train.iloc[a]), y[a], transform.transform(train.iloc[b]), y[b]))
            transform = Transform(view, spec['minutes'], spec['handling']).fit(train)
            x, xv = transform.transform(train), transform.transform(valid)
            for weighting in ['none', 'balanced']:
                for family in ['logistic', 'xgboost', 'lightgbm']:
                    params, scores, best = tune(family, prepared, config[family + '_grid'], k, weighting, config['seed'])
                    model = fit_model(family, params, x, y, k, weighting, config['seed'])
                    pv, pt = probabilities(model, xv, k), probabilities(model, x, k)
                    add_predictions(family, view, weighting, pv, pt)
                    settings.append(dict(model=family, view=view, weighting=weighting, selected_index=best,
                        selected_params=params, inner_candidate_macro_f1=scores,
                        dropped_training_all_missing=transform.dropped, converged=True))
                    joblib.dump((transform, model), build / f'{view}__{family}__{weighting}.joblib', compress=3)
                    if spec['task'] == 'three_class' and spec['minutes'] == 60 and spec['cohort'] == 'joint' and view in ['context_hr_steps', 'context_missingness']:
                        rng = np.random.default_rng(config['seed'])
                        reference = macro_f1(yv, pv.argmax(axis=1), k)
                        for group, columns in transform.groups.items():
                            if not columns:
                                continue
                            decreases = []
                            for _ in range(config['permutation_repeats']):
                                permuted = xv.copy()
                                permuted[:, columns] = xv[rng.permutation(len(xv))][:, columns]
                                decreases.append(reference - macro_f1(yv, probabilities(model, permuted, k).argmax(axis=1), k))
                            importance.append(dict(view=view, model=family, weighting=weighting, feature=group,
                                                   macro_f1_decreases=decreases))
    pd.concat(predictions, ignore_index=True).to_parquet(build / 'predictions.parquet', index=False)
    write_json(build / 'settings.json', settings)
    write_json(build / 'training.json', training)
    write_json(build / 'importance.json', importance)
    publish_bundle(build, destination, meta)
    return destination.name, 'complete'


def prepare(root, config):
    input_guard(root, REFERENCE, root / 'data')
    assignments, _ = frozen_assignments(root, REFERENCE)
    frame = load_development(root, config, assignments)
    source = {str(p.relative_to(root)): sha256(p) for p in [root / 'src/garmin_pss_classification' / name for name in ['__init__.py', 'data.py', 'metrics.py', 'models.py', 'run.py']]}
    identity = dict(config=config, specification_sha256=sha256(root / 'reports/pss_classification/specification.md'),
        source_sha256=source, input_sha256={name: sha256(root / config[name] / 'manifest.json')
        for name in ['baseline_bundle', 'feature_bundle', 'history_bundle', 'eligibility_bundle']},
        assignment_sha256=sha256(root / 'private/frozen_participant_assignments.parquet'),
        packages={name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm']})
    run = root / 'private/pss_classification/runs' / digest_object(identity)[:16]
    if not run.exists():
        run.mkdir(parents=True)
        (run / 'jobs').mkdir()
        write_json(run / 'prefit_specification.json', identity)
        frame.to_parquet(run / 'development.parquet', index=False)
        write_json(run / 'development_checksum.json', {'sha256': sha256(run / 'development.parquet')})
        shutil.copy2(root / 'reports/pss_classification/specification.md', run / 'specification.md')
    else:
        if sha256(run / 'development.parquet') != json.loads((run / 'development_checksum.json').read_text())['sha256']:
            raise AuditError('Private input bundle changed.')
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / 'config/pss_classification.json').read_text())
    run = prepare(root, config)
    print('Versioned run:', run.relative_to(root), flush=True)
    if args.prepare_only:
        return
    os.environ['OMP_NUM_THREADS'] = '1'
    jobs = [(spec, fold) for spec in experiments(config) for fold in range(5)]
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        futures = [pool.submit(fold_job, str(root), str(run), spec, fold, config) for spec, fold in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            _, status = future.result()
            print(f'Fold jobs {number}/{len(jobs)}: {status}', flush=True)
    input_guard(root, REFERENCE, root / 'data')
    print('All fold jobs finished; raw/reference integrity verified.', flush=True)


if __name__ == '__main__':
    main()
