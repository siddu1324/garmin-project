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
                (p.parts[0] == 'reports' and p.suffix in {'.md', '.json'}) or
                (p.parts[0] == 'notebooks' and p.name == '00_data_inventory_and_understanding.ipynb'))
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


def scan_public(root, identifiers, private_names):
    """Scan exact known keys/names and executable notebook output state; no values printed."""
    checked = 0
    paths = [p for folder in ['reports', 'notebooks', 'src', 'tests']
             for p in (root / folder).rglob('*') if p.is_file()
             and p.suffix in {'.json', '.md', '.ipynb', '.py'}]
    paths += [root / name for name in ['README.md', '.gitignore', 'pyproject.toml', 'uv.lock']
              if (root / name).exists()]
    for p in paths:
        text = p.read_text()
        if contains_private_token(text, identifiers, private_names):
            raise AuditError('Public artifact contains a known private token; details withheld.')
        if p.suffix == '.ipynb':
            nb = json.loads(text)
            if any(c.get('outputs') or c.get('execution_count') is not None
                   for c in nb['cells'] if c['cell_type'] == 'code'):
                raise AuditError('Source notebook must have empty outputs.')
        checked += 1
    return checked
