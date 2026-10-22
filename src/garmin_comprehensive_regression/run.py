"""Immutable, resumable nested regression jobs over the existing development matrix."""

import argparse
import importlib.metadata
import json
import tempfile
import shutil
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from garmin_comprehensive.extract import REFERENCE
from garmin_comprehensive.run import get_data
from garmin_ema_baseline.linkage import check_development
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import (
    digest_object, frozen_assignments, publish_bundle, verify_bundle, write_json,
)
from .models import TARGET, nested_fit, specifications

META = ['n_ema_id', 'n_participant', 'n_fold', 'n_survey_category', TARGET]


def json_parameters(value):
    """Record estimator defaults such as XGBoost missing=NaN without invalid JSON."""
    if isinstance(value, dict):
        return {k: json_parameters(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_parameters(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return {'nonfinite_default': str(value)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def data(root, cfg):
    folder = root / cfg['feature_bundle']
    verify_bundle(folder)
    frame, blocks = get_data(folder)
    assignment, dev = frozen_assignments(root, REFERENCE)
    check_development(frame, assignment, 3706)
    assert set(frame.n_participant) == dev
    assert len(blocks['full']) == 438 and len(blocks['pulseox_full']) == 678
    assert frame[TARGET].dropna().between(3, 15).all()
    return frame.sort_values('n_ema_id').reset_index(drop=True), blocks


def cohort(frame, blocks, name):
    mask = frame[TARGET].notna() & frame.context_available
    if name == 'complete60':
        mask &= frame[blocks['original60']].notna().all(axis=1)
    return frame.loc[mask].copy()


def basic(y, p):
    from scipy.stats import spearmanr
    residual = p - y
    total = np.sum((y - y.mean()) ** 2)
    return dict(mae=float(np.abs(residual).mean()), rmse=float(np.sqrt(np.mean(residual**2))),
                r2=float(1 - np.sum(residual**2) / total) if total > 0 else None,
                spearman=float(spearmanr(y, p).statistic) if np.ptp(y) and np.ptp(p) else None,
                prediction_sd=float(np.std(p)), prediction_min=float(p.min()),
                prediction_max=float(p.max()), signed_error=float(residual.mean()),
                outside_target_range=float(np.mean((p < 3) | (p > 15))))


def one_job(root, run, spec, fold, cfg):
    with threadpool_limits(limits=1):
        return _one_job(root, run, spec, fold, cfg)


def _one_job(root, run, spec, fold, cfg):
    destination = run / 'jobs' / f'{spec["name"]}__fold{fold}'
    if destination.exists():
        verify_bundle(destination)
        return 'cached'
    frame, blocks = data(root, cfg)
    f = cohort(frame, blocks, spec['cohort'])
    train, valid = f[f.n_fold.ne(fold)], f[f.n_fold.eq(fold)]
    build = Path(tempfile.mkdtemp(prefix='.building_', dir=run / 'jobs'))
    info = dict(spec=spec, fold=fold, training_responses=len(train), validation_responses=len(valid),
                training_participants=train.n_participant.nunique(),
                validation_participants=valid.n_participant.nunique())
    try:
        valid_x = valid.drop(columns=[c for c in valid if c == TARGET or c.startswith('target_')])
        fitted, best, trials, splits, warnings_seen = nested_fit(
            train, valid_x, spec, blocks[spec['block']], cfg)
        p, pt = fitted.predict(valid_x), fitted.predict(train)
        result, training = valid[META].copy(), train[META].copy()
        result['prediction'], training['prediction'] = p, pt
        for name in ['mean', 'median']:
            constant = float(getattr(train[TARGET], name)())
            result['dummy_' + name] = constant
            training['dummy_' + name] = constant
        result.to_parquet(build / 'predictions.parquet', index=False)
        training.to_parquet(build / 'training_predictions.parquet', index=False)
        membership = []
        for inner, (a, b) in enumerate(splits):
            for role, ix in [('train', a), ('validation', b)]:
                membership.append(train.iloc[ix][['n_ema_id', 'n_participant']].assign(
                    inner_fold=inner, role=role))
        pd.concat(membership).to_parquet(build / 'inner_membership.parquet', index=False)
        joblib.dump(fitted, build / 'estimator.joblib', compress=3)
        write_json(build / 'trials.json', trials)
        model = fitted.model
        coefficient = getattr(model, 'coef_', None)
        importance = getattr(model, 'feature_importances_', None)
        write_json(build / 'features.json', dict(
            names=fitted.names, dropped=fitted.transform.dropped,
            coefficient=coefficient.tolist() if coefficient is not None else None,
            importance=importance.tolist() if importance is not None else None))
        info.update(status='complete', best=best, warning_categories=warnings_seen,
                    train=basic(train[TARGET].to_numpy(), pt),
                    validation=basic(valid[TARGET].to_numpy(), p),
                    transformed_features=len(fitted.transform.names),
                    selected_features=len(fitted.names),
                    nonzero_coefficients=int(np.sum(np.abs(coefficient) > 1e-8))
                    if coefficient is not None else None,
                    estimator_parameters=json_parameters(model.get_params()))
    except Exception as exc:
        info.update(status='failed', error_type=type(exc).__name__)
        (build / 'error_private.txt').write_text(str(exc))
    write_json(build / 'summary.json', info)
    publish_bundle(build, destination, dict(spec=spec, fold=fold))
    return info['status']


def reuse_metadata_compatible(previous, destination, identity):
    """Reuse only completed jobs when the sole fit-path edit is JSON-safe metadata."""
    old = json.loads((previous/'run_identity.json').read_text())
    for key in ['config', 'specification', 'features', 'assignments', 'reused_transform', 'packages']:
        assert old[key] == identity[key], 'Incompatible reuse inputs.'
    assert old['code']['models.py'] == identity['code']['models.py']
    old_source = (previous/'executed_source/run.py').read_text()
    assert sha256(previous/'executed_source/run.py') == old['code']['run.py']
    old_source = old_source.replace('estimator_parameters=model.get_params())',
                                    'estimator_parameters=json_parameters(model.get_params()))')
    before = {x.name: ast.dump(x) for x in ast.parse(old_source).body if isinstance(x, ast.FunctionDef)}
    after = {x.name: ast.dump(x) for x in ast.parse(Path(__file__).read_text()).body
             if isinstance(x, ast.FunctionDef)}
    for name in ['data', 'cohort', 'basic', 'one_job', '_one_job']:
        assert before[name] == after[name], 'Fit-path change prevents cache reuse.'
    reused = []
    for p in sorted((previous/'jobs').glob('*/manifest.json')):
        verify_bundle(p.parent)
        target = destination/'jobs'/p.parent.name
        if not target.exists():
            shutil.copytree(p.parent, target)
        verify_bundle(target)
        reused.append(dict(job=p.parent.name, original_manifest_sha256=sha256(p)))
    write_json(destination/'metadata_compatible_reuse.json', dict(parent=str(previous), jobs=reused,
        reason='Strict JSON serialization of nonfinite estimator defaults only; fit path AST verified.'))


def run(root, workers=4, reuse_run=None):
    root = Path(root).resolve()
    cfg = json.loads((root / 'config/comprehensive_regression.json').read_text())
    frame, blocks = data(root, cfg)
    identity = dict(config=cfg, specification=sha256(root / 'reports/comprehensive_regression/specification.md'),
                    features=sha256(root / cfg['feature_bundle'] / 'manifest.json'),
                    assignments=sha256(root / 'private/frozen_participant_assignments.parquet'),
                    code={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')
                          if p.name in ['models.py', 'run.py']},
                    reused_transform=sha256(root / 'src/garmin_comprehensive/models.py'),
                    packages={p: importlib.metadata.version(p) for p in
                              ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'lightgbm', 'joblib']})
    version = digest_object(identity)[:16]
    destination = root / 'private/comprehensive_regression/runs' / version
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'jobs').mkdir(exist_ok=True)
    write_json(root / 'private/comprehensive_regression/current.json', dict(run=str(destination.relative_to(root))))
    if (destination / 'manifest.json').exists():
        verify_bundle(destination)
        return destination
    if not (destination / 'run_identity.json').exists():
        write_json(destination / 'run_identity.json', identity)
        source = destination / 'executed_source'
        source.mkdir()
        for name in identity['code']:
            (source / name).write_bytes((Path(__file__).parent / name).read_bytes())
        (source / 'specification.md').write_bytes((root / 'reports/comprehensive_regression/specification.md').read_bytes())
        accounting = frame[META].copy()
        accounting['outcome_available'] = frame[TARGET].notna()
        accounting['context_available'] = frame.context_available
        accounting['complete_original60'] = frame[blocks['original60']].notna().all(axis=1)
        accounting['included'] = accounting.outcome_available & accounting.context_available
        accounting['exclusion_reason'] = np.where(~accounting.outcome_available, 'missing_outcome',
                                                  np.where(~accounting.context_available, 'missing_context', 'included'))
        accounting.to_parquet(destination / 'ema_accounting.parquet', index=False)
    specs = specifications()
    write_json(destination / 'specifications.json', specs)
    if reuse_run:
        reuse_metadata_compatible(root/reuse_run, destination, identity)
    jobs = [(s, f) for s in specs for f in range(5)]
    print(f'Frozen {len(specs)} comparisons / {len(jobs)} outer jobs; version {version}', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one_job, root, destination, s, f, cfg): (s['name'], f) for s, f in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            status = future.result()
            name, fold = futures[future]
            print(f'{i}/{len(jobs)} {name} fold {fold}: {status}', flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reuse-run')
    args = parser.parse_args()
    print(run(Path.cwd(), args.workers, args.reuse_run))
