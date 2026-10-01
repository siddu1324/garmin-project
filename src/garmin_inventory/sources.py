"""Read-only ZIP discovery. No extraction, basename linkage, or pickle loading."""
import hashlib
import io
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .privacy import AuditError

MAX_MEMBER_BYTES = 1_500_000_000
MAX_CONTAINER_BYTES = 12_000_000_000


def safe_members(z):
    members = []
    total = 0
    for m in z.infolist():
        p = PurePosixPath(m.filename)
        if p.is_absolute() or '..' in p.parts or '\\' in m.filename:
            raise AuditError('Unsafe archive member path; names withheld.')
        if m.is_dir() or '__MACOSX' in p.parts or p.name == '.DS_Store':
            continue
        if m.flag_bits & 1 or ((m.external_attr >> 16) & 0o170000) == 0o120000:
            raise AuditError('Encrypted or symlink archive member is unsupported.')
        total += m.file_size
        if m.file_size > MAX_MEMBER_BYTES or total > MAX_CONTAINER_BYTES:
            raise AuditError('Archive exceeds inspection size budget.')
        members.append(m)
    return members


@dataclass(frozen=True)
class Source:
    archive: Path
    member: str
    nested: str | None = None

    @contextmanager
    def open(self):
        with zipfile.ZipFile(self.archive) as outer:
            if self.nested is None:
                with outer.open(self.member) as f:
                    yield f
            else:
                with zipfile.ZipFile(io.BytesIO(outer.read(self.nested))) as inner:
                    with inner.open(self.member) as f:
                        yield f

    def content_hash(self):
        h = hashlib.sha256()
        with self.open() as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    def exact_tokens(self):
        return set(Path(self.member).stem.split('.')) | set(Path(self.nested or '').stem.split('.'))


def discover(data):
    ema, garmin, pdf = [], [], []
    inventory, names = [], set()
    for p in sorted(data.rglob('*.zip')):
        names.add(p.name)
        with zipfile.ZipFile(p) as z:
            members = safe_members(z)
            inventory.append({'archive': f'archive_{len(inventory) + 1:02}',
                              'bytes': p.stat().st_size, 'members': len(members),
                              'uncompressed_bytes': sum(m.file_size for m in members),
                              'extensions': dict(Counter(Path(m.filename).suffix for m in members))})
            for m in members:
                names.add(Path(m.filename).name)
                suffix = Path(m.filename).suffix.lower()
                if suffix == '.zip':
                    with zipfile.ZipFile(io.BytesIO(z.read(m))) as inner:
                        for n in safe_members(inner):
                            names.add(Path(n.filename).name)
                            if n.filename.lower().endswith('.csv'):
                                garmin.append(Source(p, n.filename, m.filename))
                elif suffix == '.csv':
                    import pandas as pd
                    with z.open(m) as f:
                        columns = set(pd.read_csv(f, nrows=0).columns)
                    if {'actual_start_local', 'scheduled_start_local', 'mbl_cod'} <= columns:
                        ema.append(Source(p, m.filename))
                    elif {'ts', 'dte_tme', 'data_type', 'val'} <= columns:
                        garmin.append(Source(p, m.filename))
                    else:
                        raise AuditError('Unrecognized local CSV schema; no legacy substitution.')
                elif suffix == '.pdf':
                    pdf.append(Source(p, m.filename))
    if not ema or not garmin:
        raise AuditError('Corrected EMA and Garmin archives are required.')
    return ema, garmin, pdf, inventory, names
