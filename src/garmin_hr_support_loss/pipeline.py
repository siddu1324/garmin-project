"""Guarded development-only immutable follow-up; no support/loss combination search."""
import argparse
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from garmin_ema_baseline.linkage import check_development
from garmin_ema_baseline.pipeline import records, verified_inputs
from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (digest_object, frozen_assignments, input_guard,
                                         publish_bundle, verify_bundle, write_json)
from garmin_preprocessing.normalize import write_parquet
from .experiment import evaluate, run_comparisons
from .support import verify_quality


def run(root, reference):
    root, reference = Path(root), Path(reference)
    config = json.loads((root / 'config/hr_support_loss.json').read_text())
    frozen = json.loads((root / 'private/hr_support_loss/prefit_specification.json').read_text())
    baseline = root / config['baseline_bundle']
    if (sha256(root / 'config/hr_support_loss.json') != frozen['config_sha256']
            or sha256(root / 'reports/hr_support_loss_specification.md') != frozen['specification_sha256']
            or sha256(baseline / 'manifest.json') != frozen['baseline_manifest_sha256']):
        raise AuditError('Follow-up specification or baseline changed after freezing.')
    input_guard(root, reference, root / 'data')
    assignments, _ = frozen_assignments(root, reference)
    manifest = verify_bundle(baseline)
    prior_config = json.loads((baseline / 'frozen_specification.json').read_text())
    _, features = verified_inputs(root, prior_config)
    if (features != root / config['feature_bundle']
            or sha256(features / 'manifest.json') != manifest['inputs']['features_manifest']
            or sha256(root / 'private/frozen_participant_assignments.parquet') != manifest['inputs']['assignments_sha256']
            or sha256(root / 'reports/window_supersession.json') != manifest['inputs']['supersession_sha256']):
        raise AuditError('Notebook 05 input provenance no longer matches.')
    inputs = {'baseline_manifest': sha256(baseline / 'manifest.json'), 'feature_manifest': sha256(features / 'manifest.json'),
              'config_sha256': frozen['config_sha256'], 'specification_sha256': frozen['specification_sha256'],
              'freeze_record_sha256': sha256(root / 'private/hr_support_loss/prefit_specification.json'),
              'assignments_sha256': manifest['inputs']['assignments_sha256'],
              'supersession_sha256': manifest['inputs']['supersession_sha256'], 'uv_lock_sha256': sha256(root / 'uv.lock'),
              'code': {str(p.relative_to(root)): sha256(p) for folder in ['src/garmin_hr_support_loss', 'src/garmin_ema_baseline']
                       for p in sorted((root / folder).glob('*.py')) if p.name not in ['report.py', 'verify.py']},
              'versions': {p: version(p) for p in ['scikit-learn', 'scipy', 'numpy', 'pandas', 'pyarrow']}}
    identifier = digest_object(inputs)[:16]
    parent = root / 'private/hr_support_loss/runs'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / identifier
    if destination.exists():
        verify_bundle(destination)
        return destination
    linked = pd.read_parquet(baseline / 'linked_development.parquet')
    check_development(linked, assignments, 3706)
    if linked.n_participant.nunique() != 125:
        raise AuditError('Development accounting changed.')
    quality, occupancy_verification = verify_quality(linked, features)
    saved = pd.read_parquet(baseline / 'out_of_fold_predictions.parquet')
    results = run_comparisons(linked, quality, saved, config)
    results.update(evaluate(results['predictions'], config))
    build = Path(tempfile.mkdtemp(prefix='.building_', dir=parent))
    write_parquet(quality, build / 'verified_hr_support.parquet')
    for name, value in results.items():
        if name in ['predictions', 'membership']:
            write_parquet(value, build / (name + '.parquet'))
        else:
            write_json(build / (name + '.json'), value if name == 'training_audit' else records(value))
    write_json(build / 'occupancy_verification.json', occupancy_verification)
    write_json(build / 'frozen_configuration.json', config)
    verify_bundle(baseline)
    input_guard(root, reference, root / 'data')
    publish_bundle(build, destination, {'version': identifier, 'inputs': inputs,
                                      'preserved_ema_rows': len(linked), 'prediction_rows': len(results['predictions']),
                                      'membership_rows': len(results['membership']), 'development_only': True})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--reference', type=Path, default=os.environ.get('GARMIN_REFERENCE_PROJECT'))
    args = parser.parse_args()
    if args.reference is None:
        parser.error('Set GARMIN_REFERENCE_PROJECT or pass --reference.')
    print('Verified development follow-up bundle:', run(args.root, args.reference).relative_to(args.root))


if __name__ == '__main__':
    main()
