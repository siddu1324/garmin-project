"""Source-identity comparisons and observed-minute weighting diagnostics, never predictors."""

from itertools import combinations

import numpy as np
import pandas as pd

from garmin_inventory.privacy import AuditError
from garmin_morning_recall.data import columns_for

WINDOWS = ["wake", "fixed_5", "fixed_60", "fixed_120"]
FEATURES = ["mean", "sd", "latest", "slope"]


def density(readings, start, end):
    if not len(readings):
        return dict(
            readings=0, occupied_minutes=0, possible_minutes=int(np.ceil((end - start) / 60))
        )
    bins = np.floor((readings.n_epoch_s.to_numpy() - start) / 60).astype(int)
    per = (
        pd.DataFrame({"bin": bins, "value": readings.n_value.to_numpy()})
        .groupby("bin")
        .value.agg(["count", "mean", "sum"])
    )
    top = per.sort_values(["count", "bin"], ascending=[False, True]).head(
        int(np.ceil(len(per) * 0.1))
    )
    mean = float(readings.n_value.mean())
    return dict(
        readings=len(readings),
        occupied_minutes=len(per),
        possible_minutes=int(np.ceil((end - start) / 60)),
        occupied_fraction=len(per) / np.ceil((end - start) / 60),
        count_cv=float(per["count"].std(ddof=0) / per["count"].mean()),
        count_min=int(per["count"].min()),
        count_median=float(per["count"].median()),
        count_max=int(per["count"].max()),
        dense_decile_reading_share=float(top["count"].sum() / len(readings)),
        dense_decile_mean_contribution_bpm=float(top["sum"].sum() / len(readings)),
        reading_mean=mean,
        observed_minute_mean=float(per["mean"].mean()),
        reading_minus_minute_mean=mean - float(per["mean"].mean()),
        absolute_weighting_difference=abs(mean - float(per["mean"].mean())),
    )


def reading_ids(frame):
    return set(zip(frame.n_source_sha256, frame.n_source_row))


def compare(a, b, va, vb, tolerances):
    sa, sb = reading_ids(a), reading_ids(b)
    complete = bool(np.isfinite(va).all() and np.isfinite(vb).all())
    delta = np.asarray(va) - np.asarray(vb)
    both = bool(len(a) and len(b))
    latest_identity = both and (a.n_source_sha256.iloc[-1], a.n_source_row.iloc[-1]) == (
        b.n_source_sha256.iloc[-1],
        b.n_source_row.iloc[-1],
    )
    return dict(
        readings_a=len(sa),
        readings_b=len(sb),
        shared=len(sa & sb),
        only_a=len(sa - sb),
        only_b=len(sb - sa),
        identical_sets=sa == sb,
        jaccard=len(sa & sb) / len(sa | sb) if sa | sb else None,
        both_have_readings=both,
        identical_latest_identity=latest_identity,
        identical_latest_value=both and a.n_value.iloc[-1] == b.n_value.iloc[-1],
        complete_vectors=complete,
        identical_vector=complete and bool(np.all(np.abs(delta) <= 1e-10)),
        near_identical_vector=complete and bool(np.all(np.abs(delta) <= tolerances)),
        **{f"{k}_a": float(v) for k, v in zip(FEATURES, va)},
        **{f"{k}_b": float(v) for k, v in zip(FEATURES, vb)},
    )


def run_information(morning, features, config):
    f = pd.read_parquet(morning / "features.parquet")
    membership = pd.read_parquet(morning / "membership.parquet")
    q = pd.read_parquet(morning / "quality.parquet").set_index(["scenario", "n_ema_id", "interval"])
    pairs, densities = [], []
    relevant = membership[membership.included & membership.cohort.isin(config["audit_cohorts"])]
    key_groups = relevant.groupby(["scenario", "cohort"]).n_ema_id.apply(set).to_dict()
    selected = f[f.n_ema_id.isin(relevant.n_ema_id)]
    for directory, part in selected.groupby("directory"):
        ledger = pd.read_parquet(features / "reading_dispositions" / directory / "hr.parquet")
        if not ledger.n_partition.eq("development").all() or not set(ledger.n_participant) == set(
            part.n_participant
        ):
            raise AuditError("Window ledger identity mismatch.")
        accepted = ledger[ledger.a_analysis_keep].sort_values("n_epoch_s")
        times = accepted.n_epoch_s.to_numpy()
        for _, row in part.iterrows():
            groups = [
                c for c in config["audit_cohorts"] if row.n_ema_id in key_groups[(row.scenario, c)]
            ]
            if not groups:
                continue
            windows = {}
            for name in WINDOWS:
                qual = q.loc[(row.scenario, row.n_ema_id, name)]
                start, end = qual.start_epoch, qual.end_epoch
                left, right = np.searchsorted(times, [start, end])
                readings = accepted.iloc[left:right]
                if len(readings) != qual.valid_readings:
                    raise AuditError("Accepted source count differs from saved quality.")
                windows[name] = readings
                for cohort in groups:
                    densities.append(
                        dict(
                            n_ema_id=row.n_ema_id,
                            n_participant=row.n_participant,
                            scenario=row.scenario,
                            cohort=cohort,
                            window=name,
                            **density(readings, start, end),
                        )
                    )
            for a, b in combinations(WINDOWS, 2):
                result = compare(
                    windows[a],
                    windows[b],
                    row[columns_for(a)].to_numpy(float),
                    row[columns_for(b)].to_numpy(float),
                    config["near_feature_tolerances"],
                )
                for cohort in groups:
                    pairs.append(
                        dict(
                            n_ema_id=row.n_ema_id,
                            n_participant=row.n_participant,
                            scenario=row.scenario,
                            cohort=cohort,
                            window_a=a,
                            window_b=b,
                            **result,
                        )
                    )
    return pd.DataFrame(pairs), pd.DataFrame(densities)


def aggregate_information(pairs, densities):
    overlaps, differences, density_rows = [], [], []
    for keys, f in pairs.groupby(["scenario", "cohort", "window_a", "window_b"]):
        tags = dict(zip(["scenario", "cohort", "window_a", "window_b"], keys))
        row = dict(
            **tags,
            responses=len(f),
            participants=f.n_participant.nunique(),
            complete_vectors=int(f.complete_vectors.sum()),
        )
        for name in [
            "identical_sets",
            "both_have_readings",
            "identical_latest_identity",
            "identical_latest_value",
            "identical_vector",
            "near_identical_vector",
        ]:
            row[name] = int(f[name].sum())
        for name in ["shared", "only_a", "only_b", "jaccard", "readings_a", "readings_b"]:
            row[name + "_median"] = float(f[name].median())
        overlaps.append(row)
        for feature in FEATURES:
            a, b = f[feature + "_a"], f[feature + "_b"]
            valid = np.isfinite(a) & np.isfinite(b)
            a, b = a[valid], b[valid]
            d = a - b
            differences.append(
                dict(
                    **tags,
                    feature=feature,
                    responses=len(d),
                    mean_signed_difference=float(d.mean()),
                    median_absolute_difference=float(d.abs().median()),
                    q95_absolute_difference=float(d.abs().quantile(0.95)),
                    pearson=float(a.corr(b)) if a.nunique() > 1 and b.nunique() > 1 else None,
                    spearman=float(a.corr(b, method="spearman"))
                    if a.nunique() > 1 and b.nunique() > 1
                    else None,
                )
            )
    for keys, f in densities.groupby(["scenario", "cohort", "window"]):
        row = dict(
            zip(["scenario", "cohort", "window"], keys),
            responses=len(f),
            participants=f.n_participant.nunique(),
        )
        for name in densities.select_dtypes(include="number").columns:
            if name not in ["n_participant"]:
                for suffix, quant in [("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("q95", 0.95)]:
                    row[name + "_" + suffix] = float(f[name].quantile(quant))
        density_rows.append(row)
    return {
        "window_overlap": pd.DataFrame(overlaps),
        "feature_differences": pd.DataFrame(differences),
        "density_summary": pd.DataFrame(density_rows),
    }
