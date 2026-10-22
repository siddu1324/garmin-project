"""Independent original-source spot checks and all-row clock/label verification."""

import csv
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_inventory.sources import discover
from garmin_inventory.privacy import sha256
from garmin_preprocessing.eligibility import window_ends
from garmin_preprocessing.common import write_json
from garmin_window_benchmark.trace import trace_sample
from .data import independent_hr, HR


def run(root, f, cfg, out):
    root, out = Path(root), Path(out)
    normalized, ledger = root / cfg["normalized_bundle"], root / cfg["ledger_bundle"]
    ema = pd.read_parquet(normalized / "ema/development_ema.parquet")
    idx = ema.set_index("n_ema_id").loc[f.n_ema_id]
    assert np.array_equal(idx.n_participant, f.n_participant)
    assert np.array_equal(idx.n_fold, f.n_fold)
    expected = []
    for row in ema.itertuples():
        local = None
        for fmt in ["%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"]:
            try:
                local = datetime.strptime(row.actual_start_local, fmt)
                break
            except ValueError:
                pass
        if local is None:
            raise ValueError("Unrecognized source clock; details withheld")
        utc = (local - timedelta(minutes=float(row.timezone_offset))).replace(tzinfo=timezone.utc)
        cutoff = utc - timedelta(seconds=60 if row.n_start_precision == "minute" else 0)
        assert cutoff.timestamp() == row.n_cutoff_epoch
        values = []
        for key in ["PSS_NERVOUS", "PSS_COPE", "PSS_CONTROL"]:
            try:
                v = float(getattr(row, key))
            except (ValueError, TypeError):
                v = np.nan
            values.append(v if v in [1, 2, 3, 4, 5] else np.nan)
        expected.append(values[0] + 12 - values[1] - values[2])
    score = pd.Series(expected, index=ema.n_ema_id).loc[f.n_ema_id]
    np.testing.assert_allclose(score, f.pss_combo, equal_nan=True)
    # Known overflow edges, every generated step, all eight lookbacks, scalar datetime reference.
    checks = 0
    for c in [0.0, 1700000035.0]:  # synthetic clocks only
        for minutes in [5, 15, 30, 60, 120, 360, 720, 1440]:
            ends = window_ends(c, minutes)
            for k, end in enumerate(ends):
                ref = (datetime.fromtimestamp(c, timezone.utc) - timedelta(minutes=k)).timestamp()
                assert end == ref and end - (end - 300) == 300
                assert c - minutes * 60 <= end - 300 < end <= c
                checks += 1
    # Re-execute existing independent ORIGINAL raw measurement trace: 10 people / all folds.
    prior_cfg = json.loads((root / "config/window_benchmark.json").read_text())
    trace, result = trace_sample(root, normalized, root / prior_cfg["morning_bundle"], ledger, prior_cfg)
    trace.to_parquet(out / "private_source_trace.parquet", index=False)
    # Original answer records: parse selected development rows only; never parse final-test answers.
    sources = discover(root / "data")[0]
    counts = 0
    for number, source in enumerate(sources):
        e = ema[ema.n_source_id.eq(f"ema_{number:02}")].sort_values("n_source_row").head(2)
        assert source.content_hash() == e.n_source_sha256.iloc[0]
        wanted = e.set_index("n_source_row")
        with source.open() as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            columns = next(reader)
            for n, fields in enumerate(reader, 1):
                if n not in wanted.index:
                    continue
                row = dict(zip(columns, fields))
                saved = wanted.loc[n]
                for key in ["PSS_NERVOUS", "PSS_COPE", "PSS_CONTROL", "actual_start_local", "timezone_offset", "mbl_cod"]:
                    assert row[key] == saved[key]
                assert row["mbl_cod"].strip() == saved.n_participant
                counts += 1
    # All-eight-window HR summaries independently recalculated on deterministic observed rows.
    mapping = json.loads((normalized / "participant_directories.json").read_text())
    by_person = {p["participant"]: p for p in mapping}
    traced = []
    for fold in range(5):
        choices = f[f.n_fold.eq(fold) & f.matched].sort_values("n_ema_id").head(2)
        for r in choices.itertuples():
            directory = by_person[r.n_participant]["directory"]
            h = pd.read_parquet(ledger / "reading_dispositions" / directory / "hr.parquet")
            h = h[h.a_first & h.a_acceptable]
            for m in [5, 15, 30, 60, 120, 360, 720, 1440]:
                w = h[h.n_epoch_s.ge(r.n_cutoff_epoch - int(m) * 60) & h.n_epoch_s.lt(r.n_cutoff_epoch)]
                values = independent_hr(w.n_epoch_s, w.n_value)
                for key, value in values.items():
                    name = key.replace("__60m", f"__{m}m")
                    np.testing.assert_allclose(value, getattr(r, name), equal_nan=True, atol=1e-10)
                traced.append(dict(n_ema_id=r.n_ema_id, minutes=m, checked=len(HR)))
    pd.DataFrame(traced).to_parquet(out / "private_all_window_trace.parquet", index=False)
    evidence = {
        "all_ema_clock_checks": len(ema), "all_ema_score_checks": len(ema),
        "original_answer_rows": counts, "original_sensor_trace": result,
        "independent_window_boundaries": checks, "eight_window_feature_values": len(traced) * 4,
        "new_confirmed_defects": [], "final_test_values_parsed": 0,
        "normalized_ema_sha256": sha256(normalized / "ema/development_ema.parquet"),
        "limits": "Bounded original sensor/answer trace plus all-development score and clock checks; not proof of acquisition semantics or exhaustive raw-row verification",
    }
    write_json(out / "integrity_audit.json", evidence)
    return evidence
