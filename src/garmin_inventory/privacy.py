"""Fail-closed Git checks, source integrity and public-artifact checks."""
import hashlib
import json
import re
import subprocess
from pathlib import Path


class AuditError(RuntimeError):
    """Only use messages that do not contain source records or private filenames."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git_guard(root):
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).split(b'\0')
    allowed = {'.gitignore', '.python-version', 'pyproject.toml', 'uv.lock', 'README.md'}
    bad = []
    for raw in filter(None, tracked):
        p = Path(raw.decode())
        safe = (str(p) in allowed or
                (p.parts[0] in {'src', 'tests'} and p.suffix == '.py') or
                (p.parts[0] == 'reports' and p.suffix in {'.md', '.json', '.svg', '.png'}) or
                (str(p) in {'config/ibi_eligibility.json', 'config/hrv_input_diagnostics.json', 'config/ibi_pilot.json', 'config/hr_activity.json', 'config/ema_baseline.json', 'config/hr_support_loss.json', 'config/recall_diagnostics.json', 'config/morning_recall.json', 'config/window_benchmark.json', 'config/past_hr_reference.json', 'config/hrv_input_diagnostics_corrected.json', 'config/measurement_source.json', 'config/pss_classification.json'}) or
                (p.parts[0] == 'notebooks' and p.name in {'00_data_inventory_and_understanding.ipynb', '01_preprocessing_and_ibi_eligibility.ipynb', '02_hrv_input_validation.ipynb', '03_ibi_variability_pilot.ipynb', '04_hr_activity_features.ipynb', '05_ema_outcome_baseline.ipynb', '06_hr_support_and_loss.ipynb', '07_recall_alignment_and_model_diagnostics.ipynb', '08_morning_recall_window_comparison.ipynb', '09_window_information_and_nonlinear_benchmark.ipynb', '10_within_person_signal_and_past_hr_reference.ipynb', '11_measurement_source_and_timestamp_audit.ipynb', '12_pss_classification.ipynb'}))
        if not safe:
            from garmin_publication.paths import allowed as publication_allowed
            safe = publication_allowed(root, str(p))
        if not safe:
            bad.append(str(p))
    if bad:
        # Intentionally do not print possibly identifying paths into a notebook.
        raise AuditError('Tracked/staged paths require a private review; no files untracked.')
    for probe in ['data/probe', 'private/probe', 'outputs/probe', 'models/probe',
                  'config.local.json', '.env', 'probe.csv', 'probe.zip']:
        if subprocess.run(['git', 'check-ignore', '-q', probe], cwd=root).returncode:
            raise AuditError('Required privacy ignore rule is missing.')


def tree_snapshot(root):
    """Hash all reference files, including legacy outputs, without executing its code."""
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): {'sha256': sha256(p), 'mtime_ns': p.stat().st_mtime_ns,
                                     'size': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file()
            and not any(x in p.parts for x in ['.git', '.venv', '__pycache__', '.pytest_cache'])}


def contains_private_token(text, identifiers, private_names):
    # Short numeric identifiers can occur by chance inside dependency/content hashes.
    # Match complete identifier tokens; still match filenames verbatim.
    return any(re.search(r'(?<![A-Za-z0-9])' + re.escape(str(t)) + r'(?![A-Za-z0-9])', text)
               for t in identifiers if len(str(t)) >= 4) or any(
                   str(t) in text for t in private_names if len(str(t)) >= 4)


def public_scan_text(path):
    """Exclude only schema-validated aggregate window counts from identifier matching.

    A five-digit count can equal a real identifier by coincidence. String values,
    keys and unexpected schemas remain subject to the ordinary fail-closed scan.
    """
    text = path.read_text()
    if path.name == 'attrition.json':
        rows = json.loads(text)
        keys = {'policy', 'scenario', 'lookback_minutes', 'survey_category', 'reason', 'windows'}
        if isinstance(rows, list) and all(isinstance(row, dict) and set(row) == keys
                and type(row['windows']) is int and row['windows'] >= 0 for row in rows):
            return json.dumps([{**row, 'windows': '[aggregate window count]'} for row in rows])
    if path.name == 'window_supersession.json':
        record = json.loads(text)
        rows = record.get('attrition_delta') if isinstance(record, dict) else None
        keys = {'policy', 'scenario', 'lookback_minutes', 'survey_category', 'reason',
                'windows', 'obsolete_windows', 'delta_windows'}
        if isinstance(rows, list) and all(isinstance(row, dict) and set(row) == keys
                and all(type(row[k]) is int for k in ['windows', 'obsolete_windows', 'delta_windows'])
                and row['windows'] >= 0 and row['obsolete_windows'] >= 0
                and row['delta_windows'] == row['windows'] - row['obsolete_windows'] for row in rows):
            record['attrition_delta'] = [{**row, **{k: '[validated aggregate count]'
                for k in ['windows', 'obsolete_windows', 'delta_windows']}} for row in rows]
            return json.dumps(record)
    return text


def scan_public(root, identifiers, private_names):
    """Scan exact known keys/names and executable notebook output state; no values printed."""
    checked = 0
    paths = [p for folder in ['reports', 'notebooks', 'src', 'tests', 'config']
             for p in (root / folder).rglob('*') if p.is_file()
             and p.suffix in {'.json', '.md', '.ipynb', '.py', '.svg'}]
    paths += [root / name for name in ['README.md', '.gitignore', 'pyproject.toml', 'uv.lock']
              if (root / name).exists()]
    for p in paths:
        text = public_scan_text(p)
        if contains_private_token(text, identifiers, private_names):
            raise AuditError('Public artifact contains a known private token; details withheld.')
        if p.suffix == '.ipynb':
            nb = json.loads(text)
            validate_notebook_output(p, nb)
        checked += 1
    return checked


def validate_notebook_output(path, nb):
    """Notebooks 12–21 are explicitly authorized to retain aggregate execution output.

    Known-token scanning still applies to all source and output text. This exception
    is not permission for participant tables; its cells are audited aggregate readers.
    """
    executed = path.name in {'12_pss_classification.ipynb',
                             '13_comprehensive_physio_features.ipynb',
                             '14_comprehensive_pss_regression.ipynb',
                             '15_binary_pss_6hour_experiments.ipynb',
                             '16_binary_stress_improvement.ipynb',
                             '17_temporal_personalized_stress.ipynb',
                             '18_root_cause_diagnostics.ipynb',
                             '19_measurement_alignment.ipynb',
                             '20_complete_pipeline_for_review.ipynb',
                             '21_central_research_overview.ipynb'} and nb.get('metadata', {}).get(
        'garmin_public_aggregate_notebook') is True
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        if not executed:
            if cell.get('outputs') or cell.get('execution_count') is not None:
                raise AuditError('Source notebook must have empty outputs.')
        else:
            if type(cell.get('execution_count')) is not int:
                raise AuditError('Authorized presentation notebook is not fully executed.')
            for output in cell.get('outputs', []):
                if output.get('output_type') not in {'display_data', 'execute_result', 'stream'}:
                    raise AuditError('Presentation notebook contains an error or unknown output.')
                if set(output.get('data', {})) - {'text/plain', 'text/html', 'text/markdown', 'image/png'}:
                    raise AuditError('Presentation notebook contains an unreviewed MIME output.')
