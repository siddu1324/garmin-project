"""Development-only descriptive diagnostics; no new prediction or clock repair."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from garmin_inventory.sources import discover
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import verify_bundle, write_json
from garmin_comprehensive_regression.evaluate import finite


def availability_indicator(g):
    """Counters expose accepted adjacent pairs, not the HR-only accepted_readings field."""
    counters = g.sensor.isin(["steps", "floors"])
    return g.accepted_readings.gt(0).where(~counters, g.supported_pairs.gt(0))


def median(s):
    return float(s.median()) if s.notna().any() else np.nan


def summary(g):
    counters = g.sensor.isin(["steps", "floors"]).all()
    return dict(
        responses=len(g),
        participants=g.n_participant.nunique(),
        availability_definition="supported_counter_pair" if counters else "accepted_reading",
        any_recorded=int(g.recorded_groups.gt(0).sum()),
        any_accepted=int(availability_indicator(g).sum()),
        occupied_fraction_median=median(g.occupied_fraction),
        accepted_readings_median=median(g.accepted_readings),
        gap_median_seconds=median(g.maximum_gap_seconds),
        freshness_median_seconds=median(g.latest_age_seconds),
        supported_seconds_median=median(g.supported_seconds),
    )


def perturb(t, cutoff, minutes, shift):
    """Stored post-cutoff records are excluded even for the negative-shift diagnostic."""
    end = min(cutoff, cutoff - shift)
    a, b = np.searchsorted(t, [cutoff - minutes * 60 - shift, end], side="left")
    z = t[a:b] + shift
    return len(z), len(np.unique(np.floor((z - (cutoff - minutes * 60)) / 60)))


def availability(g, seed, draws):
    """Positive-minus-negative availability; standardize to overlapping survey/time strata."""
    g = g.dropna(subset=["label"]).copy()
    g["stratum"] = g.n_survey_category.astype(str) + "/" + g.time_bin.astype(str)
    strata = [s for s, z in g.groupby("stratum") if z.label.nunique() == 2]
    h = g[g.stratum.isin(strata)].copy()
    people, inv = np.unique(g.n_participant, return_inverse=True)
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(len(people), np.ones(len(people)) / len(people), draws)
    weights = h.groupby("stratum").size() / len(h) if len(h) else pd.Series(dtype=float)

    y = g.label.to_numpy()
    v = g.available.to_numpy(float)
    masks = {s: g.stratum.eq(s).to_numpy() for s in weights.index}

    def estimate(w):
        rates = [
            np.sum(w[y == k] * v[y == k]) / np.sum(w[y == k]) if np.sum(w[y == k]) else np.nan
            for k in [0, 1]
        ]
        adjusted = 0.0
        for s, sw in weights.items():
            mask = masks[s]
            r = []
            for k in [0, 1]:
                ix = mask & (y == k)
                r.append(np.sum(w[ix] * v[ix]) / np.sum(w[ix]) if np.sum(w[ix]) else np.nan)
            adjusted += sw * (r[1] - r[0])
        return rates[1] - rates[0], adjusted if len(weights) else np.nan

    point = estimate(np.ones(len(g)))
    boot = np.array([estimate(w[inv]) for w in multiplicities])
    result = dict(
        responses=len(g),
        participants=len(people),
        overlap_strata=len(weights),
        standardized_responses=len(h),
    )
    for i, key in enumerate(["crude_difference", "survey_time_standardized_difference"]):
        z = boot[:, i]
        z = z[np.isfinite(z)]
        result.update(
            {
                key: point[i],
                key + "_low": np.quantile(z, 0.025) if len(z) else np.nan,
                key + "_high": np.quantile(z, 0.975) if len(z) else np.nan,
                key + "_draws": len(z),
            }
        )
    return result


def run(root, cfg, private, public):
    root, private, public = map(Path, [root, private, public])
    for key in ["normalized_bundle", "feature_bundle", "development_bundle", "ledger_bundle"]:
        verify_bundle(root / cfg[key])
    ema = pd.read_parquet(root / cfg["normalized_bundle"] / "ema/development_ema.parquet")
    cols = [
        "n_ema_id",
        "n_participant",
        "n_partition",
        "n_fold",
        "n_survey_category",
        "n_cutoff_epoch",
        "n_actual_epoch",
        "pss_combo",
        "target_composite5",
        "target_composite10",
        "context_available",
        "local_time_sin",
        "local_time_cos",
    ]
    dev = pd.read_parquet(root / cfg["development_bundle"] / "development.parquet", columns=cols)
    assert (
        len(dev) == 3706
        and dev.n_ema_id.is_unique
        and dev.n_participant.nunique() == 125
        and dev.n_partition.eq("development").all()
    )
    assert ema.n_ema_id.tolist() == dev.n_ema_id.tolist()
    local = pd.to_datetime(ema.n_actual_local)
    meta = dev.copy()
    meta["time_bin"] = (local.dt.hour // 4 * 4).astype("Int64").astype(str)
    delay = (
        pd.to_datetime(ema.n_actual_utc, utc=True) - pd.to_datetime(ema.n_scheduled_utc, utc=True)
    ).dt.total_seconds() / 60
    timing = meta[["n_ema_id", "n_participant", "n_fold", "n_survey_category", "time_bin"]].copy()
    timing["delay_minutes"] = delay
    timing["precision"] = ema.n_start_precision
    timing.to_parquet(private / "timing.parquet", index=False)
    records = []
    for cat, g in timing.groupby("n_survey_category"):
        z = g.delay_minutes.dropna()
        records.append(
            dict(
                survey=cat,
                responses=len(g),
                missing_start=int(ema.loc[g.index, "n_actual_utc"].isna().sum()),
                missing_schedule=int(ema.loc[g.index, "n_scheduled_utc"].isna().sum()),
                minute_resolution=int(g.precision.eq("minute").sum()),
                second_resolution=int(g.precision.eq("second").sum()),
                delay_available=len(z),
                negative_delay=int(z.lt(0).sum()),
                delay_median_minutes=z.median(),
                delay_p25_minutes=z.quantile(0.25),
                delay_p75_minutes=z.quantile(0.75),
                delay_p95_minutes=z.quantile(0.95),
                **{
                    f"delay_gt_{m}_minutes": int(z.gt(m).sum())
                    for m in cfg["diagnostic_delay_minutes"]
                },
            )
        )
    write_json(public / "timing.json", finite(records))
    hist = (
        timing.assign(
            delay_band=pd.cut(
                timing.delay_minutes, [-np.inf, 0, 5, 15, 60, 360, np.inf], right=True
            ).astype(str)
        )
        .groupby(["n_survey_category", "delay_band"])
        .size()
        .reset_index(name="responses")
    )
    hist.to_csv(public / "delay_histogram.csv", index=False)
    # Field availability is counted only in the development-normalized records.
    fields = {
        "actual_start_local": (
            "verified",
            "actual survey start, not delivery/submission",
            "minute/second",
            "2030-01-01 09:22",
        ),
        "scheduled_start_local": (
            "verified",
            "exported schedule, not verified notification",
            "minute/second",
            "2030-01-01 09:20",
        ),
        "timezone_offset": ("verified", "local minus UTC in minutes", "numeric minutes", "-240"),
        "instance_id": (
            "provisional",
            "export instance key; not a delivered-prompt event log",
            "identifier",
            "SYNTHETIC_INSTANCE",
        ),
        "rsp_id": (
            "provisional",
            "response identifier; mapping preserved",
            "identifier",
            "SYNTHETIC_RESPONSE",
        ),
        "mbl_cod": (
            "verified",
            "participant identity via frozen verified crosswalk",
            "identifier",
            "SYNTHETIC_PERSON",
        ),
        "MORNING_SURVEY_COMPLETED": (
            "provisional",
            "status field, not completion timestamp",
            "categorical",
            "synthetic status",
        ),
        "SLEEP_WAKETIME": (
            "provisional",
            "current-response waking answer, retrospective only",
            "clock text",
            "07:30",
        ),
        "Location Timestamp": (
            "unsupported",
            "location clock, not survey delivery/completion",
            "unverified",
            "synthetic location time",
        ),
        "n_actual_utc": (
            "verified",
            "actual_start_local minus row offset",
            "derived",
            "2030-01-01T13:22Z",
        ),
        "n_cutoff_epoch": (
            "verified",
            "actual UTC minus 60 seconds if minute resolution",
            "derived Unix seconds",
            "synthetic epoch",
        ),
        "n_source_row": ("verified", "original CSV data row, header excluded", "integer", "1"),
        "n_source_sha256": (
            "verified",
            "original uncompressed CSV hash",
            "SHA256",
            "synthetic hash",
        ),
    }
    audit = []
    for field, (status, meaning, precision, example) in fields.items():
        s = ema[field]
        missing = s.isna() | s.astype(str).str.strip().isin(["", "nan", "None"])
        audit.append(
            dict(
                field=field,
                status=status,
                meaning=meaning,
                precision=precision,
                synthetic_example=example,
                development_missing=int(missing.sum()),
                denominator=len(ema),
                source=f"{cfg['normalized_bundle']}/ema/development_ema.parquet",
            )
        )
    for field in [
        "delivered_prompt",
        "answer_submitted",
        "answer_received",
        "answer_version",
        "independent_wake",
        "sensor_acquisition",
        "sensor_callback",
        "sensor_upload",
        "device_quality",
        "wear_status",
    ]:
        audit.append(
            dict(
                field=field,
                status="missing",
                meaning="No verified event-time/quality contract in local exports",
                development_missing=len(ema),
                denominator=len(ema),
            )
        )
    write_json(public / "field_audit.json", audit)
    # Exact original names remain private because archive/member paths may carry identity.
    sources = discover(root / "data")
    catalog = []
    for category, items in zip(["ema", "garmin", "questionnaire"], sources[:3]):
        for index, s in enumerate(items):
            catalog.append(
                dict(
                    category=category,
                    source_index=index,
                    archive=str(s.archive.relative_to(root)),
                    member=s.member,
                    nested=s.nested,
                )
            )
    write_json(private / "exact_source_catalog.json", catalog)
    q = pd.read_parquet(root / cfg["feature_bundle"] / "quality.parquet")
    q = q[q.sensor.isin(["hr", "steps", "floors", "pulseox"])].merge(
        meta, on="n_ema_id", validate="many_to_one"
    )
    assert not q.duplicated(["n_ema_id", "sensor", "lookback_minutes"]).any()
    for sensor, g in q.groupby("sensor"):
        assert len(g) == 3706 * 8
    coverage = []
    for dimensions in [
        ["sensor", "lookback_minutes"],
        ["sensor", "lookback_minutes", "n_survey_category"],
        ["sensor", "lookback_minutes", "time_bin"],
    ]:
        for keys, g in q.groupby(dimensions):
            coverage.append(
                dict(grouping="/".join(dimensions), **dict(zip(dimensions, keys)), **summary(g))
            )
    pd.DataFrame(coverage).to_csv(public / "coverage.csv", index=False)
    people = []
    for keys, g in q.groupby(["n_participant", "sensor", "lookback_minutes"]):
        people.append(
            dict(zip(["n_participant", "sensor", "lookback_minutes"], keys), **summary(g))
        )
    person = pd.DataFrame(people)
    person.to_parquet(private / "participant_coverage.parquet", index=False)
    person.groupby(["sensor", "lookback_minutes"]).agg(
        participants=("n_participant", "size"),
        participants_any_accepted=("any_accepted", lambda s: int(s.gt(0).sum())),
        median_responses_with_readings=("any_accepted", "median"),
        median_occupied_fraction=("occupied_fraction_median", "median"),
    ).reset_index().to_csv(public / "participant_coverage.csv", index=False)
    labelrows = []
    adjusted = []
    for sensor, g in q[q.lookback_minutes.eq(60)].groupby("sensor"):
        for threshold in [5, 10]:
            g = g.assign(
                label=g[f"target_composite{threshold}"], available=availability_indicator(g)
            )
            for label, z in g.dropna(subset=["label"]).groupby("label"):
                labelrows.append(
                    dict(sensor=sensor, threshold=threshold, label=int(label), **summary(z))
                )
            adjusted.append(
                dict(
                    sensor=sensor,
                    threshold=threshold,
                    **availability(g, cfg["seed"], cfg["bootstrap_draws"]),
                )
            )
    write_json(public / "availability_by_label.json", finite(labelrows))
    write_json(public / "availability_associations.json", finite(adjusted))
    # Private per-response gate includes every EMA; missing metadata is not failed HR physiology.
    gate = meta[["n_ema_id", "n_participant", "n_fold", "n_survey_category"]].copy()
    gate["status"] = "BLOCKED"
    gate["reason"] = "missing_approved_acquisition_contract_and_verified_recall_boundary"
    gate["history_reason"] = "missing_exact_answer_version_availability"
    gate.to_parquet(private / "recall_gate.parquet", index=False)
    directories = json.loads(
        (root / cfg["normalized_bundle"] / "participant_directories.json").read_text()
    )
    sensor_fields = []
    cadence = []
    sens = []
    counter = []
    for person in directories:
        directory = person["directory"]
        p = person["participant"]
        em = meta[meta.n_participant.eq(p)]
        for sensor in ["hr", "steps"]:
            path = root / cfg["normalized_bundle"] / "logical" / directory / f"{sensor}.parquet"
            if not path.exists():
                continue
            z = pd.read_parquet(path)
            assert z.n_partition.eq("development").all() and z.n_participant.eq(p).all()
            fields = ["ts", "dte_tme", "rsp_id", "mbl_cod", "data_type", "val"]
            rawpath = root / cfg["normalized_bundle"] / "physical" / directory / f"{sensor}.parquet"
            for batch in pq.ParquetFile(rawpath).iter_batches(batch_size=100000, columns=fields):
                physical = batch.to_pandas()
                for field in fields:
                    missing = physical[field].isna() | physical[field].astype(str).str.strip().isin(
                        ["", "nan", "None"]
                    )
                    sensor_fields.append(
                        dict(
                            sensor=sensor,
                            field=field,
                            records=len(physical),
                            missing=int(missing.sum()),
                        )
                    )
            t = z.n_epoch_s.to_numpy(float)
            v = z.n_value.to_numpy(float)
            valid = np.isfinite(t)
            order = np.argsort(t[valid], kind="stable")
            tt = t[valid][order]
            vv = v[valid][order]
            dt = np.diff(tt)
            source_order = z.sort_values(["n_source_sha256", "n_source_row"])
            reverse = int((source_order.groupby("n_source_sha256").n_epoch_s.diff() < 0).sum())
            cadence.append(
                dict(
                    n_participant=p,
                    sensor=sensor,
                    records=len(z),
                    finite_clocks=int(valid.sum()),
                    timestamp_repeat_pairs=int((dt == 0).sum()),
                    source_order_reversals=reverse,
                    repeated_value_adjacent=int((np.diff(vv) == 0).sum()),
                    adjacent_pairs=len(dt),
                    fractional_timestamps=int((tt != np.floor(tt)).sum()),
                    median_gap_seconds=float(np.median(dt)) if len(dt) else np.nan,
                    **{
                        f"gap_gt_{b}_seconds": int((dt > b).sum())
                        for b in cfg["diagnostic_gap_seconds"]
                    },
                )
            )
            if sensor == "steps":
                pairs = pd.read_parquet(
                    root / cfg["ledger_bundle"] / "counter_pairs" / directory / "steps.parquet"
                )
                counter.append(
                    dict(
                        n_participant=p,
                        records=len(pairs),
                        accepted_pairs=int(pairs.a_pair_accepted.sum()),
                        negative_reset_pairs=int(pairs.a_pair_delta.lt(0).sum()),
                        zero_change_accepted_pairs=int(
                            (pairs.a_pair_accepted & pairs.a_pair_delta.eq(0)).sum()
                        ),
                        positive_accepted_pairs=int(
                            (pairs.a_pair_accepted & pairs.a_pair_delta.gt(0)).sum()
                        ),
                    )
                )
        hp = root / cfg["ledger_bundle"] / "reading_dispositions" / directory / "hr.parquet"
        if hp.exists():
            h = pd.read_parquet(hp)
            t = np.sort(h.loc[h.a_first & h.a_acceptable, "n_epoch_s"].dropna().to_numpy(float))
        else:
            t = np.array([])
        for row in em.itertuples():
            for minutes in cfg["lookbacks_minutes"]:
                for shift in cfg["sensor_clock_sensitivity_seconds"]:
                    count, bins = perturb(t, row.n_cutoff_epoch, minutes, shift)
                    sens.append(
                        dict(
                            n_ema_id=row.n_ema_id,
                            n_participant=p,
                            lookback_minutes=minutes,
                            shift_seconds=shift,
                            accepted_readings=count,
                            occupied_fraction=bins / minutes,
                        )
                    )
    pd.DataFrame(sensor_fields).groupby(["sensor", "field"])[
        ["records", "missing"]
    ].sum().reset_index().to_csv(public / "sensor_field_availability.csv", index=False)
    c = pd.DataFrame(cadence)
    c.to_parquet(private / "cadence.parquet", index=False)
    aggregate = []
    for sensor, g in c.groupby("sensor"):
        sums = {
            col: int(g[col].sum())
            for col in g.select_dtypes("number")
            if col != "median_gap_seconds"
        }
        aggregate.append(
            dict(
                sensor=sensor,
                participants=len(g),
                **sums,
                median_of_participant_median_gap_seconds=g.median_gap_seconds.median(),
            )
        )
    write_json(public / "cadence.json", finite(aggregate))
    counters = pd.DataFrame(counter)
    counters.to_parquet(private / "counter_diagnostics.parquet", index=False)
    write_json(
        public / "counters.json",
        {k: int(v) for k, v in counters.drop(columns="n_participant").sum().items()},
    )
    sensitivity = pd.DataFrame(sens)
    sensitivity.to_parquet(private / "timing_sensitivity.parquet", index=False)
    sensitivity.groupby(["lookback_minutes", "shift_seconds"]).agg(
        responses=("n_ema_id", "size"),
        responses_any_accepted=("accepted_readings", lambda x: int(x.gt(0).sum())),
        median_readings=("accepted_readings", "median"),
        mean_occupied_fraction=("occupied_fraction", "mean"),
        median_occupied_fraction=("occupied_fraction", "median"),
    ).reset_index().to_csv(public / "timing_sensitivity.csv", index=False)
    old = root / "reports/root_cause_analysis/c62e4b329f41c611/sensor_summary.json"
    write_json(
        public / "previous_hr_evidence.json",
        dict(
            source=str(old.relative_to(root)),
            source_sha256=sha256(old),
            evidence=json.loads(old.read_text()),
        ),
    )
    return dict(
        development_responses=len(dev),
        development_participants=dev.n_participant.nunique(),
        blocked_aligned=len(gate),
        blocked_history=len(gate),
        new_real_predictive_fits=0,
        metadata_supplied=False,
    )
