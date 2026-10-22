"""Deterministic immutable bundles and input guards."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from garmin_inventory.audit import validate_split
from garmin_inventory.privacy import AuditError, git_guard, sha256, tree_snapshot


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_config(root):
    p = root / 'config/ibi_eligibility.json'
    return json.loads(p.read_text())


def frozen_assignments(root, reference):
    p = root / 'private/frozen_participant_assignments.parquet'
    expected = json.loads((root / 'reports/inventory.json').read_text())['frozen_split']['assignment_sha256']
    if sha256(p) != expected or sha256(reference / 'data_interim/v2/v2_step13_participant_assignments.parquet') != expected:
        raise AuditError('Frozen assignments changed; stop without resplitting.')
    a = pd.read_parquet(p)
    dev = validate_split(a, set(a.participant))
    return a, dev


def input_guard(root, reference, data):
    git_guard(root)
    baseline = json.loads((root / 'private/integrity_baseline.json').read_text())
    if tree_snapshot(data) != baseline['data'] or tree_snapshot(reference) != baseline['reference']:
        raise AuditError('Raw/reference inputs differ from the verified inventory baseline.')
    return baseline


def artifact_hashes(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name != 'manifest.json'}


def verify_archive_bundle(folder):
    """Checksum-only historical inspection; never permission to use obsolete windows."""
    m = json.loads((folder / 'manifest.json').read_text())
    if not m.get('complete') or artifact_hashes(folder) != m['artifact_sha256']:
        raise AuditError('Existing version is incomplete or changed; never overwrite validated artifacts.')
    return m


def verify_bundle(folder):
    from .provenance import assert_current_windows
    m = json.loads((Path(folder) / "manifest.json").read_text())
    assert_current_windows(folder, m)
    return verify_archive_bundle(folder)


def publish_bundle(build, destination, metadata):
    metadata = dict(metadata, complete=True, artifact_sha256=artifact_hashes(build))
    write_json(build / 'manifest.json', metadata)
    if destination.exists():
        raise AuditError('Immutable destination exists; refusing to overwrite it.')
    build.rename(destination)
    return metadata
