"""Independent bounded trace from original development CSVs to saved features."""

import hashlib
import json
import statistics
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

from garmin_inventory.privacy import AuditError
from garmin_inventory.sources import discover
from garmin_morning_recall.data import columns_for
from .information import WINDOWS


def scalar_features(t, v):
    if len(v) < 3 or t[-1] - t[0] < 60:
        raise AuditError("Trace sample unexpectedly lacks feature support.")
    x = [(a - t[0]) / 60 for a in t]
    xm = statistics.mean(x)
    vm = statistics.mean(v)
    slope = sum((a - xm) * (b - vm) for a, b in zip(x, v)) / sum((a - xm) ** 2 for a in x)
    return [vm, statistics.stdev(v), v[-1], slope]


def trace_sample(root, normalized, morning, features, config):
    f = pd.read_parquet(morning / "features.parquet")
    membership = pd.read_parquet(morning / "membership.parquet")
    keys = membership.loc[
        membership.included
        & membership.scenario.eq("reported_minute_start")
        & membership.cohort.eq("common"),
        "n_ema_id",
    ]
    base = f[f.n_ema_id.isin(keys) & f.scenario.eq("reported_minute_start")].copy()
    base["order"] = base.n_ema_id.map(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
    chosen = []
    for _, part in base.groupby("n_fold"):
        chosen.extend(
            part.sort_values("order")
            .drop_duplicates("n_participant")
            .head(config["trace_participants_per_fold"])
            .n_ema_id.tolist()
        )
    selected = f[f.n_ema_id.isin(chosen)]
    ema = pd.read_parquet(normalized / "ema/development_ema.parquet").set_index("n_ema_id")
    lineage = pd.read_parquet(features / "window_lineage.parquet")
    lineage = lineage[lineage.sensor.eq("hr")].set_index(["n_ema_id", "lookback_minutes"])
    wake_lineage = pd.read_parquet(morning / "lineage.parquet").set_index(["scenario", "n_ema_id"])
    catalog = {
        x["source_id"]: x
        for x in json.loads((normalized / "source_catalog.json").read_text())["garmin"]
    }
    sources = discover(root / "data")[1]
    directory_map = {
        x["directory"]: x
        for x in json.loads((normalized / "participant_directories.json").read_text())
    }
    trace_rows = []
    raw_rows = 0
    archives_checked = 0
    for directory, part in selected.groupby("directory"):
        physical = pd.read_parquet(normalized / "physical" / directory / "hr.parquet")
        ledger = pd.read_parquet(features / "reading_dispositions" / directory / "hr.parquet")
        person = part.n_participant.iloc[0]
        fold = int(part.n_fold.iloc[0])
        assert (
            directory_map[directory]["participant"] == person
            and directory_map[directory]["fold"] == fold
        )
        assert (
            physical.n_participant.eq(person).all()
            and physical.n_partition.eq("development").all()
            and physical.n_fold.eq(fold).all()
        )
        assert physical.n_mapping_method.notna().all()
        # Independent acceptance: exact-record logical keep, chronological first source row,
        # then reject whole timestamp group on conflicting values or any bad clock/value.
        logical = (
            physical[physical.n_logical_keep]
            .sort_values(["n_epoch_s", "n_source_id", "n_source_row"], kind="stable")
            .copy()
        )
        logical["bad"] = (
            logical[
                [
                    "q_invalid_epoch",
                    "q_invalid_text_clock",
                    "q_clock_discrepancy",
                    "q_unusual_offset",
                ]
            ].any(axis=1)
            | ~np.isfinite(logical.n_epoch_s)
            | ~logical.n_value.between(30, 240)
            | ~np.isfinite(logical.n_value)
            | logical.q_conflicting_timestamp_value
        )
        groups = logical.groupby("n_epoch_s", dropna=False)
        logical["bad_group"] = groups.bad.transform("any") | groups.n_value.transform(
            "nunique", dropna=False
        ).gt(1)
        accepted = logical[~logical.bad_group & ~logical.n_epoch_s.duplicated()]
        g = ledger[ledger.a_first & np.isfinite(ledger.n_epoch_s)].reset_index(drop=True)
        wanted = []
        for _, row in part.iterrows():
            e = ema.loc[row.n_ema_id]
            assert e.n_participant == person and int(e.n_fold) == fold
            actual = pd.Timestamp(e.actual_start_local).to_pydatetime()
            utc = (actual - timedelta(minutes=float(e.timezone_offset))).replace(
                tzinfo=timezone.utc
            )
            cutoff = (
                utc - timedelta(seconds=60 if row.n_start_precision == "minute" else 0)
            ).timestamp()
            assert cutoff == row.end_epoch == row.n_cutoff_epoch
            for name in WINDOWS:
                start = (
                    float(row.start_epoch)
                    if name == "wake"
                    else (
                        utc
                        - timedelta(
                            seconds=(60 if row.n_start_precision == "minute" else 0)
                            + 60 * int(name[6:])
                        )
                    ).timestamp()
                )
                w = accepted[accepted.n_epoch_s.ge(start) & accepted.n_epoch_s.lt(cutoff)]
                values = scalar_features(w.n_epoch_s.tolist(), w.n_value.tolist())
                assert np.allclose(
                    values, row[columns_for(name)].to_numpy(float), rtol=1e-10, atol=1e-9
                )
                lin = (
                    wake_lineage.loc[(row.scenario, row.n_ema_id)]
                    if name == "wake"
                    else lineage.loc[(row.n_ema_id, int(name[6:]))]
                )
                assert lin.directory == directory
                sl = g.iloc[int(lin.group_slice_start) : int(lin.group_slice_stop)]
                sl = sl[sl.a_acceptable]
                assert list(zip(w.n_source_sha256, w.n_source_row)) == list(
                    zip(sl.n_source_sha256, sl.n_source_row)
                )
                wanted.append(w)
                trace_rows.append(
                    dict(
                        n_ema_id=row.n_ema_id,
                        n_participant=person,
                        n_fold=fold,
                        scenario=row.scenario,
                        window=name,
                        accepted_readings=len(w),
                        features_verified=4,
                        source_lineage_verified=True,
                        clock_verified=True,
                    )
                )
        wanted = pd.concat(wanted).drop_duplicates(["n_source_id", "n_source_row"])
        for sid, need in wanted.groupby("n_source_id"):
            meta = catalog[sid]
            assert meta["participant"] == person and meta["fold"] == fold
            source = sources[int(sid.removeprefix("garmin_"))]
            assert source.content_hash() == meta["content_sha256"]
            assert need.n_source_sha256.eq(meta["content_sha256"]).all()
            lookup = need.set_index("n_source_row")
            found = 0
            offset = 1
            with source.open() as stream:
                for chunk in pd.read_csv(
                    stream, dtype=str, keep_default_na=False, chunksize=100000
                ):
                    chunk.index = np.arange(offset, offset + len(chunk))
                    offset += len(chunk)
                    match = chunk.index.intersection(lookup.index)
                    if not len(match):
                        continue
                    raw = chunk.loc[match]
                    saved = lookup.loc[match]
                    for col in ["ts", "dte_tme", "rsp_id", "mbl_cod", "data_type", "val"]:
                        if col in raw:
                            assert np.array_equal(raw[col].to_numpy(), saved[col].to_numpy())
                        else:
                            assert saved[col].isna().all()
                            assert all(col not in json.loads(s) for s in saved.n_source_columns)
                    assert np.allclose(pd.to_numeric(raw.ts), saved.n_epoch_s, atol=0, rtol=0)
                    assert np.allclose(pd.to_numeric(raw.val), saved.n_value, atol=0, rtol=0)
                    found += len(match)
            assert found == len(need)
            raw_rows += found
            archives_checked += 1
    result = dict(
        sample_responses=len(chosen),
        sample_participants=selected.n_participant.nunique(),
        window_views=len(trace_rows),
        feature_values_verified=4 * len(trace_rows),
        unique_original_hr_rows_verified=raw_rows,
        development_sources_verified=archives_checked,
        failures=0,
        final_test_sources_parsed=0,
    )
    return pd.DataFrame(trace_rows), result
