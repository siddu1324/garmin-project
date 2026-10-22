"""Read-only pre-fit source audit and aggregate recording diagnostics."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from garmin_ema_baseline.linkage import score_items
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import write_json, verify_bundle
from garmin_binary_6hour.diagnostics import quantiles
from .data import data
from .design import WINDOWS


def audit(root):
    root = Path(root)
    out = root / "reports/binary_improvement/audit"
    if (out / "checks.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    f, blocks, cfg = data(root)
    folder = root / cfg["feature_bundle"]
    verify_bundle(folder)
    ema = pd.read_parquet(
        root / "private/preprocessing/normalized/31e51c934bcdeffa/ema/development_ema.parquet",
        columns=[
            "n_ema_id",
            "n_participant",
            "n_fold",
            "n_partition",
            "PSS_NERVOUS",
            "PSS_COPE",
            "PSS_CONTROL",
            "actual_start_local",
            "timezone_offset",
            "n_start_precision",
            "n_cutoff_epoch",
        ],
    )
    ema = ema.set_index("n_ema_id").loc[f.n_ema_id].reset_index()
    assert ema.n_partition.eq("development").all()
    np.testing.assert_array_equal(ema.n_participant, f.n_participant)
    np.testing.assert_array_equal(ema.n_fold, f.n_fold)
    scored = score_items(ema)
    np.testing.assert_allclose(scored.pss_combo, f.pss_combo, equal_nan=True)
    for row in ema.itertuples():
        value = str(row.actual_start_local)
        actual = None
        for fmt in [
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S.%f",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
        ]:
            try:
                actual = datetime.strptime(value, fmt)
                break
            except ValueError:
                pass
        assert actual is not None, "Unsupported raw start format; private value withheld."
        utc = (actual - timedelta(minutes=float(row.timezone_offset))).replace(tzinfo=timezone.utc)
        cutoff = utc - timedelta(seconds=60 if row.n_start_precision == "minute" else 0)
        assert abs(cutoff.timestamp() - row.n_cutoff_epoch) < 1e-5
    lineage = pd.read_parquet(folder / "lineage.parquet")
    lin = lineage[lineage.lookback_minutes.isin(WINDOWS)]
    ends = f.set_index("n_ema_id").n_cutoff_epoch
    assert lin.end_epoch.eq(lin.n_ema_id.map(ends)).all()
    for row in lin.itertuples():
        end = datetime.fromtimestamp(row.end_epoch, timezone.utc)
        start = end - timedelta(minutes=int(row.lookback_minutes))
        assert abs(start.timestamp() - row.start_epoch) < 1e-5
        assert row.group_start <= row.group_stop
    # Nine deterministic EMA keys (three per survey), only aggregate assertions leave this function.
    sample = f.groupby("n_survey_category", sort=True).head(3)
    checks = 0
    selected = lin[lin.sensor.eq("hr") & lin.n_ema_id.isin(sample.n_ema_id)]
    for directory, part in selected.groupby("directory"):
        p = (
            root
            / "private/hr_activity/features/76f5885b3955d5b0/reading_dispositions"
            / directory
            / "hr.parquet"
        )
        ledger = pd.read_parquet(p)
        groups = ledger[ledger.a_first & np.isfinite(ledger.n_epoch_s)].reset_index(drop=True)
        for row in part.itertuples():
            w = groups.iloc[row.group_start : row.group_stop]
            assert w.n_epoch_s.ge(row.start_epoch).all() and w.n_epoch_s.lt(row.end_epoch).all()
            expected = f.set_index("n_ema_id").loc[row.n_ema_id]
            assert (
                w.n_participant.eq(expected.n_participant).all()
                and w.n_fold.eq(expected.n_fold).all()
            )
            v = w[w.a_acceptable]
            mu = (
                v.n_value.mean()
                if len(v) >= 2 and v.n_epoch_s.iloc[-1] - v.n_epoch_s.iloc[0] >= 1
                else np.nan
            )
            np.testing.assert_allclose(
                mu, expected[f"hr__mean__{row.lookback_minutes}m"], equal_nan=True
            )
            checks += 1
    sources = json.loads((folder / "source_partitions.json").read_text())
    # Source hashes are already sealed; hash only nine sampled HR ledgers here, no final-test partitions.
    checked_sources = 0
    for directory in selected.directory.unique():
        suffix = f"/reading_dispositions/{directory}/hr.parquet"
        names = [p for p in sources if p.endswith(suffix)]
        assert len(names) == 1 and sha256(root / names[0]) == sources[names[0]]
        checked_sources += 1
    keep = f.pss_combo.notna() & f.context_available
    valid = f[keep]
    q = pd.read_parquet(folder / "quality.parquet")
    q = q[q.sensor.eq("hr") & q.lookback_minutes.isin(WINDOWS)].merge(
        valid[["n_ema_id", "n_participant", "n_survey_category", "target_A", "target_B"]],
        on="n_ema_id",
        validate="many_to_one",
    )
    qualities, missing, distributions, composition = [], [], [], []
    for task in ["A", "B"]:
        for cls in [0, 1]:
            part = valid[valid[f"target_{task}"].eq(cls)]
            distributions.append(
                dict(
                    task=task,
                    class_id=cls,
                    responses=len(part),
                    participants=part.n_participant.nunique(),
                    fraction=len(part) / len(valid),
                )
            )
        for m in WINDOWS:
            for survey in ["All", "Morning", "Midday", "Endday"]:
                g = q[q.lookback_minutes.eq(m)]
                if survey != "All":
                    g = g[g.n_survey_category.eq(survey)]
                for metric in [
                    "recorded_groups",
                    "accepted_readings",
                    "occupied_minutes",
                    "occupied_fraction",
                    "span_seconds",
                    "latest_age_seconds",
                    "maximum_gap_seconds",
                ]:
                    qualities.append(
                        dict(
                            task=task,
                            window=m,
                            survey=survey,
                            metric=metric,
                            **quantiles(g[metric]),
                        )
                    )
            for col in blocks[f"full{m}"]:
                missing.append(
                    dict(
                        task=task,
                        window=m,
                        feature=col,
                        observed=int(valid[col].notna().sum()),
                        unique_values=int(valid[col].nunique()),
                        missing_fraction=float(valid[col].isna().mean()),
                        class0_missing=float(
                            valid.loc[valid[f"target_{task}"].eq(0), col].isna().mean()
                        ),
                        class1_missing=float(
                            valid.loc[valid[f"target_{task}"].eq(1), col].isna().mean()
                        ),
                    )
                )
        for name in ["supported", "retro_nominal", "retro_later"]:
            for state in [True, False]:
                for category in ["All", "Morning", "Midday", "Endday"]:
                    g = valid[valid[name].eq(state)]
                    if category != "All":
                        g = g[g.n_survey_category.eq(category)]
                    composition.append(
                        dict(
                            task=task,
                            cohort=name,
                            included=state,
                            survey=category,
                            responses=len(g),
                            participants=g.n_participant.nunique(),
                            class0=int(g[f"target_{task}"].eq(0).sum()),
                            class1=int(g[f"target_{task}"].eq(1).sum()),
                        )
                    )
    results = dict(
        all_development_rows=len(f),
        valid_score_context_rows=len(valid),
        missing_outcomes=int(f.pss_combo.isna().sum()),
        participants=valid.n_participant.nunique(),
        score_checks=len(ema),
        independent_clock_checks=len(ema),
        independent_lineage_boundaries=len(lin),
        sampled_reading_traces=checks,
        sampled_source_hashes=checked_sources,
        implementation_bug_found=False,
        original_artifacts_preserved=True,
        raw_reextracted=False,
        final_test_evaluated=False,
    )
    for name, value in [
        ("quality", qualities),
        ("missingness", missing),
        ("classes", distributions),
        ("composition", composition),
        ("checks", results),
    ]:
        write_json(out / f"{name}.json", value)
    return out


if __name__ == "__main__":
    print(audit(Path.cwd()))
