"""Audit fixed split membership, train-only fitted statistics and prediction replay."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import input_guard, frozen_assignments, verify_bundle, write_json
from .data import TASKS, cohort_mask, experiments, load_development
from .models import Transform, probabilities
from .run import REFERENCE


def verify(root, run):
    identity = json.loads((run / 'prefit_specification.json').read_text())
    config = identity['config']
    for file, digest in identity['source_sha256'].items():
        if sha256(root / file) != digest:
            raise AuditError('Training source changed after specification freeze.')
    if sha256(root / 'reports/pss_classification/specification.md') != identity['specification_sha256']:
        raise AuditError('Prespecified experiment was modified.')
    input_guard(root, REFERENCE, root / 'data')
    assignments, _ = frozen_assignments(root, REFERENCE)
    expected = load_development(root, config, assignments)
    frame = pd.read_parquet(run / 'development.parquet')
    pd.testing.assert_frame_equal(frame, expected)
    checked, folds, skipped = 0, 0, 0
    with threadpool_limits(limits=1):
        for spec in experiments(config):
            cohort = frame[cohort_mask(frame, spec)]
            for fold in range(5):
                folder = run / 'jobs' / f"{spec['experiment']}__fold{fold}"
                verify_bundle(folder)
                if (folder / 'skipped.json').exists():
                    skipped += 1
                    continue
                tr = cohort[cohort.n_fold != fold].reset_index(drop=True)
                va = cohort[cohort.n_fold == fold].reset_index(drop=True)
                saved_train = pd.read_parquet(folder / 'train_membership.parquet')
                saved_val = pd.read_parquet(folder / 'validation_membership.parquet')
                pd.testing.assert_frame_equal(tr[saved_train.columns], saved_train)
                pd.testing.assert_frame_equal(va[saved_val.columns], saved_val)
                inner = pd.read_parquet(folder / 'inner_membership.parquet')
                if set(inner.n_ema_id) != set(tr.n_ema_id) or inner.groupby('n_participant').inner_fold.nunique().max() != 1:
                    raise AuditError('Inner membership is incomplete or crosses participants.')
                if set(tr.n_participant) & set(va.n_participant):
                    raise AuditError('Outer membership crosses participants.')
                p = pd.read_parquet(folder / 'predictions.parquet')
                target = 'target_' + spec['task']
                majority = int(np.argmax(np.bincount(tr[target].to_numpy(int), minlength=TASKS[spec['task']])))
                dummy = p[p.model == 'dummy']
                if len(dummy) != len(va) or not (dummy.prediction == majority).all():
                    raise AuditError('Majority baseline did not use the matching training cohort.')
                for setting in json.loads((folder / 'settings.json').read_text()):
                    best = int(np.argmax(np.mean(setting['inner_candidate_macro_f1'], axis=1)))
                    if best != setting['selected_index'] or config[setting['model'] + '_grid'][best] != setting['selected_params']:
                        raise AuditError('Selected hyperparameters do not match frozen inner macro F1 selection.')
                for view in spec['views']:
                    expected_transform = Transform(view, spec['minutes'], spec['handling']).fit(tr)
                    for family in ['logistic', 'xgboost', 'lightgbm']:
                        for weighting in ['none', 'balanced']:
                            transform, model = joblib.load(folder / f'{view}__{family}__{weighting}.joblib')
                            pd.testing.assert_series_equal(transform.medians, expected_transform.medians)
                            np.testing.assert_array_equal(transform.scaler.mean_, expected_transform.scaler.mean_)
                            np.testing.assert_array_equal(transform.scaler.scale_, expected_transform.scaler.scale_)
                            assert transform.names == expected_transform.names
                            np.testing.assert_array_equal(transform.encoder.categories_[0], expected_transform.encoder.categories_[0])
                            pv = probabilities(model, transform.transform(va), TASKS[spec['task']])
                            saved = p[(p.view == view) & (p.model == family) & (p.weighting == weighting)]
                            if list(saved.n_ema_id) != list(va.n_ema_id):
                                raise AuditError('Prediction order changed.')
                            np.testing.assert_allclose(pv, saved[[f'p{i}' for i in range(TASKS[spec['task']])]], rtol=1e-7, atol=1e-8)
                            np.testing.assert_array_equal(saved.y, va['target_' + spec['task']])
                            checked += 1
                folds += 1
    snapshot = json.loads((root / 'private/pss_classification/initial_snapshot.json').read_text())
    allowed = {'README.md', 'pyproject.toml', 'uv.lock', 'src/garmin_inventory/privacy.py'}
    preserved = 0
    for file, digest in snapshot['files'].items():
        if file not in allowed:
            if sha256(root / file) != digest:
                raise AuditError('A prior checkpoint source or public artifact changed.')
            preserved += 1
    # Snapshot manifests protect previous bundle metadata; full input bundles verified above.
    manifest_map = snapshot.get('private_manifests', snapshot.get('manifests', {}))
    for file, digest in manifest_map.items():
        if sha256(root / file) != digest:
            raise AuditError('Previous private bundle manifest changed.')
    result = dict(development_rows=len(frame), development_participants=frame.n_participant.nunique(),
        final_test_participants_evaluated=0, completed_fold_jobs=folds, skipped_fold_jobs=skipped,
        replayed_estimators=checked, all_prediction_replays_agree=True,
        training_only_outer_transform_statistics_verified=True, participant_separation_verified=True,
        original_assignment_checksum_verified=True, corrected_input_bundles_verified=True,
        raw_reference_integrity_verified=True, preserved_prior_public_source_files=preserved,
        preserved_prior_manifests=len(manifest_map), missing_targets=int(frame.pss_combo.isna().sum()))
    destination = root / 'reports/pss_classification' / run.name / 'verification.json'
    if destination.exists():
        if json.loads(destination.read_text()) != result:
            raise AuditError('Existing verification differs; preserve the previous artifact.')
    else:
        write_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(Path.cwd(), args.run.resolve()), indent=2))


if __name__ == '__main__':
    main()
