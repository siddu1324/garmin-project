"""Bounded documentation discovery and immutable, aggregate evidence review.

No participant Parquet or CSV values are read here. Historical diagnostics are cited,
not recomputed. An interface/version cannot be inferred from a scalar CSV schema.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader

from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import (
    digest_object,
    frozen_assignments,
    input_guard,
    publish_bundle,
    verify_bundle,
    write_json,
)
from garmin_preprocessing.provenance import assert_current_windows

APP_SUFFIXES = {'.java', '.kt', '.swift', '.m', '.mm', '.gradle', '.dart', '.js',
                '.ts', '.tsx', '.cs', '.go', '.rs', '.cpp', '.h', '.c'}
BUILD_NAMES = {'AndroidManifest.xml', 'Package.resolved', 'Podfile.lock', 'build.gradle',
               'build.gradle.kts', 'pubspec.lock', 'package-lock.json', 'pom.xml'}
EXCLUDED = {'.git', '.venv', 'node_modules', '__pycache__', '.pytest_cache', '.ruff_cache',
            'data', 'private', 'outputs', 'data_raw', 'data_interim', 'data_processed',
            'models', 'TULIP Data', 'reports', 'site-packages'}


def source_candidates(paths):
    """Metadata classification only; returned names must remain private."""
    included = [Path(p) for p in paths if not EXCLUDED.intersection(Path(p).parts)]
    return {
        'application_sources': sorted(str(p) for p in included if p.suffix in APP_SUFFIXES),
        'build_manifests': sorted(str(p) for p in included if p.name in BUILD_NAMES),
    }


def discover_code(roots):
    paths = []
    for root in roots:
        command = ['rg', '--files', '--hidden', '--no-ignore', '-0', str(root)]
        for folder in sorted(EXCLUDED):
            command += ['-g', '!' + folder]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode not in (0, 1):
            raise AuditError('Bounded source discovery failed; private paths withheld.')
        paths += [Path(p.decode()) for p in result.stdout.split(b'\0') if p]
    candidates = source_candidates(paths)
    # Explicit count is evidence of bounded absence, not a claim about other locations.
    return dict(roots=[str(p) for p in roots], files_enumerated=len(set(paths)), **candidates)


def export_shape(schemas):
    """Report available raw fields without attributing upstream SDK semantics."""
    fields = sorted({f for schema in schemas for f in schema['fields']})
    known = {'ts', 'dte_tme', 'rsp_id', 'mbl_cod', 'data_type', 'val'}
    if set(fields) != known:
        raise AuditError('Raw export schema changed; review evidence before interpretation.')
    return dict(raw_fields=fields, schema_variants=len(schemas),
                sdk_or_version_identified=False, acquisition_clock_verified=False,
                original_payload_batch_shape='not recoverable from CSV field names',
                dedicated_source_quality_fields=[], derived_flags_are_device_annotations=False)


def synthetic_ambiguity():
    """Two upstream histories with the same exported scalar rows; no beat reconstruction."""
    scalar_rows = [{'ts': 10., 'val': 800.}, {'ts': 10., 'val': 800.}]
    return {
        'synthetic_only': True,
        'scalar_rows': scalar_rows,
        'history_a': 'One interval event retried twice, with its event clock preserved.',
        'history_b': 'Two equal intervals in a batch, flattened using one receipt clock.',
        'logical_exact_duplicate_rows_kept': 1,
        'physical_rows_preserved': 2,
        'distinguishable_from_export_alone': False,
        'needed': 'Original event/batch identity, interval index and documented clock mapping.',
    }


def evidence_spans(root):
    entries = [
        ('csv_container', 'src/garmin_inventory/sources.py', 'def discover(',
         'Reads ZIP/CSV exports; not collection-app or device parser code.'),
        ('raw_schema', 'reports/inventory.json', '"schemas":',
         'Three raw schemas; union has six fields and no explicit SDK/version/beat-quality field.'),
        ('clock_parse', 'src/garmin_preprocessing/normalize.py', "f['n_epoch_s'] =",
         'Parses Unix seconds without timezone shift; does not verify event-clock meaning.'),
        ('derived_quality', 'src/garmin_preprocessing/normalize.py', "f['q_invalid_value'] =",
         'q_ flags are analysis-derived; normal-beat status explicitly unverified.'),
        ('raw_duplicates', 'src/garmin_preprocessing/normalize.py', 'def annotate_duplicates(',
         'Exact raw/schema duplicates collapse only in logical view; physical copies survive.'),
        ('logical_order', 'src/garmin_preprocessing/normalize.py', "['n_epoch_s', 'n_source_id', 'n_source_row']",
         'Logical ordering uses exported clocks/source row, not a recovered beat sequence.'),
        ('interval_geometry', 'src/garmin_preprocessing/eligibility.py', 'def _masks(',
         'Whole-interval containment assumes either start or end anchoring.'),
        ('corrected_windows', 'src/garmin_preprocessing/eligibility.py', 'def window_ends(',
         'Current corrected generator; historical long-window results remain revoked.'),
        ('stale_versions', 'src/garmin_preprocessing/provenance.py', 'def assert_current_windows(',
         'Rejects known-invalid versions and manifest copies.'),
        ('precision', 'reports/hrv_input_validation/30a0bbe188d56b40/report.md', '## Precision, batching',
         'Reused stream-level precision evidence; no obsolete 12/24-hour counts reused.'),
        ('package_limits', 'reports/ibi_pilot/78af22e7f0ead75b/report.md', '## Package agreement',
         'Numerical agreement in limited pilot does not validate physiology.'),
        ('raw_hr_trace', 'reports/window_benchmark/e3e34a2ea014f38f/report.md', '## Does the implemented',
         'Existing independent raw trace and synthetic positive control; no new participant sample.'),
    ]
    rows = []
    for key, path, needle, conclusion in entries:
        lines = (root / path).read_text().splitlines()
        line = next((i + 1 for i, text in enumerate(lines) if needle in text), None)
        if line is None:
            raise AuditError('Evidence anchor moved; review citations before publishing.')
        rows.append(dict(key=key, path=path, line=line, sha256=sha256(root / path),
                         status='verified_local_implementation_or_export', conclusion=conclusion))
    return rows


def run(root, reference, documentation_root):
    root, reference, documentation_root = map(Path, (root, reference, documentation_root))
    input_guard(root, reference, root / 'data')
    assignments, dev = frozen_assignments(root, reference)
    if len(dev) != 125 or len(assignments) != 157:
        raise AuditError('Frozen cohort accounting changed.')
    config = json.loads((root / 'config/measurement_source.json').read_text())
    for path, expected in config['reviewed_sha256'].items():
        if sha256(root / path) != expected:
            raise AuditError('Reviewed evidence changed; create a new audit specification.')
    manifests = {}
    for path in config['current_bundles']:
        folder = root / path
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert_current_windows(folder, manifest)
        if not manifest.get('complete'):
            raise AuditError('An input bundle is incomplete.')
        manifests[path] = sha256(folder / 'manifest.json')
    # Full artifact checksum verification is recorded separately; this pins provenance.
    research = documentation_root / config['research_document']['basename']
    if sha256(research) != config['research_document']['sha256']:
        raise AuditError('Study document changed; review before drawing conclusions.')
    reader = PdfReader(research)
    excerpts = {str(page): reader.pages[page - 1].extract_text()
                for page in config['research_document']['pages']}
    if not all(term in excerpts['7'] for term in ['will integrate', 'Vivoactive 5', 'Software Development Kit']):
        raise AuditError('Planned-study evidence not found at the pinned source.')
    search = discover_code([root, documentation_root])
    inventory = json.loads((root / 'reports/inventory.json').read_text())
    public = dict(
        audit_scope='documentation, source and saved aggregate diagnostics only',
        development_participants=125, sealed_final_test_participants=32,
        new_participant_records_read=0, model_fits=0, real_hrv_calculations=0,
        code_search_roots=2,
        application_source_candidates=len(search['application_sources']),
        application_build_candidates=len(search['build_manifests']),
        collection_sdk_and_version='unidentified',
        study_document=dict(basename=research.name, sha256=sha256(research),
                            pages=[5, 7], proposed_device_is_not_export_provenance=True),
        raw_schema=export_shape(inventory['structural']['garmin']['schemas']),
        evidence=evidence_spans(root),
        sampled_raw_trace_reused=dict(sources=10, hr_readings=18153, feature_checks=320,
                                     mismatches=0, source='Notebook 09'),
        new_implementation_bug_identified=False,
    )
    inputs = dict(config_sha256=sha256(root / 'config/measurement_source.json'),
                  current_manifest_sha256=manifests,
                  code_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                  search_sha256=digest_object({key: search[key] for key in
                                              ['roots', 'application_sources', 'build_manifests']}),
                  document_sha256=sha256(research))
    version = digest_object(inputs)[:16]
    destination = root / 'private/measurement_source/runs' / version
    if destination.exists():
        verify_bundle(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix='.build-', dir=destination.parent))
    write_json(build / 'audit_summary.json', public)
    write_json(build / 'source_discovery_private.json', search)
    write_json(build / 'document_excerpts_private.json', dict(path=str(research), pages=excerpts))
    write_json(build / 'evidence_input_hashes.json', config['reviewed_sha256'])
    write_json(build / 'synthetic_ambiguity.json', synthetic_ambiguity())
    publish_bundle(build, destination, dict(version=version, inputs=inputs, development_only=True,
                                            final_test_access=False, no_new_calculations=True))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--documentation-root', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.reference, args.documentation_root)
    print('Evidence bundle:', result.relative_to(args.root))


if __name__ == '__main__':
    main()
