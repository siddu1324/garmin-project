"""Verify sources, public artifacts, source notebook and an executed notebook copy."""
import argparse
import json
from pathlib import Path

import nbformat

from .privacy import (AuditError, contains_private_token, git_guard, scan_public,
                      sha256, tree_snapshot)


def verify(root, reference, data=None):
    root, reference = Path(root).resolve(), Path(reference).resolve()
    data = Path(data or root / 'data').resolve()
    git_guard(root)
    baseline = json.loads((root / 'private/integrity_baseline.json').read_text())
    if tree_snapshot(data) != baseline['data'] or tree_snapshot(reference) != baseline['reference']:
        raise AuditError('Original source/reference files differ from the audit baseline.')
    tokens = json.loads((root / 'private/privacy_tokens.json').read_text())
    count = scan_public(root, set(tokens['identifiers']), set(tokens['names']))
    nb_path = root / 'notebooks/00_data_inventory_and_understanding.ipynb'
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)
    executed = root / 'outputs/00_data_inventory_and_understanding.executed.ipynb'
    if not executed.exists():
        raise AuditError('Executed notebook copy is unavailable.')
    done = nbformat.read(executed, as_version=4)
    nbformat.validate(done)
    code = [c for c in done.cells if c.cell_type == 'code']
    if any(c.execution_count is None for c in code):
        raise AuditError('Notebook contains unexecuted cells.')
    if any(o.output_type == 'error' for c in code for o in c.outputs):
        raise AuditError('Notebook contains an execution error.')
    text = executed.read_text()
    if contains_private_token(text, tokens['identifiers'], tokens['names']):
        raise AuditError('Executed notebook contains a known private token; details withheld.')
    report = json.loads((root / 'reports/inventory.json').read_text())
    local_split = root / 'private/frozen_participant_assignments.parquet'
    if report['frozen_split']['available']:
        if sha256(local_split) != report['frozen_split']['assignment_sha256']:
            raise AuditError('Local assignment bytes differ from the original frozen split.')
        if sha256(reference / 'data_interim/v2/v2_step13_participant_assignments.parquet') != sha256(local_split):
            raise AuditError('Reference split differs from the preserved assignment.')
    result = dict(source_integrity=True, reference_integrity=True,
                  public_artifacts_scanned=count, source_notebook_outputs_empty=True,
                  notebook_executed_code_cells=len(code), executed_notebook_errors=0,
                  frozen_assignment_bytes_preserved=report['frozen_split']['available'])
    (root / 'reports/verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    try:
        verify(args.root, args.reference, args.data)
    except Exception as exc:
        print(str(exc) if isinstance(exc, AuditError) else f'Verification stopped ({type(exc).__name__}); private details withheld.')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
