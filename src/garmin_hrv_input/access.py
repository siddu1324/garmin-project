"""Read-only reconstruction of a contract view from immutable reference slices."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_inventory.privacy import AuditError, sha256
from .contract import window_input
from garmin_preprocessing.provenance import assert_current_windows


def load_indexed_window(normalized, diagnostics, directory, window_row, policy, scenario):
    """Load one approved window with actual records and run boundaries; never HRV.

    `directory` is a private catalog key, not a public participant identifier.
    The caller verifies bundle hashes once per session before iterating windows.
    """
    normalized, diagnostics = Path(normalized), Path(diagnostics)
    manifest = json.loads((diagnostics / "manifest.json").read_text())
    assert_current_windows(diagnostics, manifest)
    if sha256(normalized / "manifest.json") != manifest["inputs"]["normalized_manifest"]:
        raise AuditError("Contract source bundle does not match the indexed version.")
    people = json.loads((normalized / "participant_directories.json").read_text())
    matches = [p for p in people if p["directory"] == directory]
    if len(matches) != 1:
        raise AuditError("Contract directory is outside the development catalog.")
    index = pd.read_parquet(diagnostics / "window_index" / f"{directory}.parquet")
    row = index.iloc[window_row]
    path = normalized / "logical" / directory / "ibi.parquet"
    if not path.exists():
        return None  # Index and prior window diagnostic still account for the empty window.
    frame = pd.read_parquet(path)
    if not frame.n_partition.eq("development").all() or not frame.n_fold.eq(row.n_fold).all():
        raise AuditError("Contract participant/fold separation failed.")
    expected = np.searchsorted(
        frame.n_epoch_s.to_numpy(), [row.n_window_start_epoch, row.n_window_end_epoch]
    )
    if tuple(expected) != (row.logical_slice_start, row.logical_slice_stop):
        raise AuditError("Contract window membership differs from its immutable index.")
    return window_input(frame, float(row.n_window_end_epoch), policy, scenario)
