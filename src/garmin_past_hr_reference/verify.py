"""Independent history arithmetic, model replay and preservation checks."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import artifact_hashes, digest_object, verify_bundle
from .models import feature_inputs, fit_predict


def verify_reference(root, bundle, config):
    features = root / config["feature_bundle"]
    refs = pd.read_parquet(bundle / "reference_features.parquet").set_index("n_ema_id")
    quality = pd.read_parquet(bundle / "reference_quality.parquet").set_index("n_ema_id")
    lineage = pd.read_parquet(bundle / "reference_lineage.parquet")
    base = pd.read_parquet(
        root / config["baseline_bundle"] / "linked_development.parquet"
    ).set_index("n_ema_id")
    windows = 0
    accepted_total = 0
    for directory, part in lineage.groupby("directory", dropna=False):
        path = features / "reading_dispositions" / str(directory) / "hr.parquet"
        ledger = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        if len(ledger):
            assert ledger.n_partition.eq("development").all()
            assert set(ledger.n_participant) == set(part.n_participant)
            assert set(ledger.n_fold) == set(part.n_fold)
            accepted = ledger[ledger.a_analysis_keep]
        else:
            accepted = pd.DataFrame(columns=["n_epoch_s", "n_value"])
        for row in part.itertuples():
            r = refs.loc[row.n_ema_id]
            q = quality.loc[row.n_ema_id]
            b = base.loc[row.n_ema_id]
            assert (
                r.n_participant == b.n_participant
                and r.n_fold == b.n_fold
                and r.n_cutoff_epoch == b.n_cutoff_epoch
            )
            if not np.isfinite(b.n_cutoff_epoch):
                assert q.reason == "invalid_cutoff"
                continue
            cutoff = datetime.fromtimestamp(float(b.n_cutoff_epoch), timezone.utc)
            start = (cutoff - timedelta(hours=25)).timestamp()
            end = (cutoff - timedelta(hours=1)).timestamp()
            assert (
                q.start_epoch == start
                and q.end_epoch == end
                and end - start == 86400
                and b.n_cutoff_epoch - end == 3600
            )
            w = accepted[accepted.n_epoch_s.ge(start) & accepted.n_epoch_s.lt(end)]
            times = pd.to_datetime(w.n_epoch_s, unit="s", utc=True)
            minute = (
                pd.Series(w.n_value.to_numpy(float), index=pd.DatetimeIndex(times).floor("min"))
                .groupby(level=0)
                .mean()
            )
            hours = pd.DatetimeIndex(times).floor("h").nunique()
            assert (
                len(minute) == q.occupied_utc_minutes
                and hours == q.occupied_utc_hours
                and len(w) == q.accepted_readings
            )
            sufficient = len(minute) >= 30 and hours >= 3
            assert sufficient == q.reference_available
            if len(w):
                assert np.isclose(
                    q.latest_reading_age_seconds, b.n_cutoff_epoch - w.n_epoch_s.max(), atol=1e-9
                )
                assert w.n_epoch_s.max() < end <= b.n_cutoff_epoch
            if sufficient:
                independent = sum(float(x) for x in minute) / len(minute)
                assert np.isclose(r.reference_mean_bpm, independent, atol=1e-10, rtol=1e-12)
                for col, current in [
                    ("hr_mean_deviation_bpm", "hr_mean_bpm__60m"),
                    ("hr_latest_deviation_bpm", "hr_latest_bpm__60m"),
                ]:
                    assert np.isclose(
                        r[col], b[current] - independent, atol=1e-10, rtol=1e-12, equal_nan=True
                    )
            else:
                assert (
                    r[["reference_mean_bpm", "hr_mean_deviation_bpm", "hr_latest_deviation_bpm"]]
                    .isna()
                    .all()
                )
            windows += 1
            accepted_total += len(w)
    return dict(independent_reference_windows=windows, accepted_reading_window_views=accepted_total)


def verify(root, bundle):
    root, bundle = Path(root), Path(bundle)
    verify_bundle(bundle)
    config = json.loads((bundle / "frozen_configuration.json").read_text())
    reference_checks = verify_reference(root, bundle, config)
    accounting = pd.read_parquet(bundle / "accounting.parquet")
    all_inputs = pd.read_parquet(bundle / "predictive_inputs.parquet")
    old = pd.read_parquet(root / config["baseline_bundle"] / "cohort_membership.parquet")
    old = (
        old[(old.scope == "native") & (old.block == "hr") & (old.lookback_minutes == 60)]
        .set_index("n_ema_id")
        .loc[accounting.n_ema_id]
    )
    assert np.array_equal(old.included, accounting.native_60m_hr)
    assert len(all_inputs) == 3706 and all_inputs.n_ema_id.is_unique
    permitted = {
        "n_ema_id",
        "n_participant",
        "n_partition",
        "n_fold",
        "n_survey_category",
        "n_cutoff_epoch",
        "pss_combo",
        "context_available",
    }
    for view in ["B_context", "C_raw_hr", "D_reference", "E_reference_deviations"]:
        permitted.update(feature_inputs(view)[0])
    assert set(all_inputs) == permitted
    linked = all_inputs[accounting.eligible].sort_values("n_ema_id")
    predictions = pd.read_parquet(bundle / "out_of_fold_predictions.parquet")
    training = pd.read_parquet(bundle / "training_predictions.parquet")
    audits = json.loads((bundle / "training_audit.json").read_text())
    outer_count = 0
    inner_count = 0
    for a in audits:
        fold = a["fold"]
        train = linked[linked.n_fold.ne(fold)]
        valid = linked[linked.n_fold.eq(fold)]
        assert not set(train.n_participant) & set(valid.n_participant)
        assert a["training_key_sha256"] == digest_object(train.n_ema_id.tolist()) and a[
            "validation_key_sha256"
        ] == digest_object(valid.n_ema_id.tolist())
        family, view = a["model"].split("__")
        numeric, category = feature_inputs(view)
        assert a["input_columns"] == numeric + category
        assert np.allclose(a["training_mean"], train[numeric].mean(), atol=1e-12, rtol=1e-12)
        splits = list(GroupKFold(3).split(train, groups=train.n_participant))
        for b in a["inner"]:
            it, iv = splits[b["inner_fold"]]
            tr, va = train.iloc[it], train.iloc[iv]
            assert not set(tr.n_participant) & set(va.n_participant)
            assert not (set(tr.n_participant) | set(va.n_participant)) & set(valid.n_participant)
            assert b["training_key_sha256"] == digest_object(tr.n_ema_id.tolist()) and b[
                "validation_key_sha256"
            ] == digest_object(va.n_ema_id.tolist())
            assert np.allclose(b["training_mean"], tr[numeric].mean(), atol=1e-12, rtol=1e-12)
            assert b["training_count"] == len(tr) and b["categories"] == sorted(
                tr.n_survey_category.unique()
            )
            inner_count += 1
        by_candidate = pd.DataFrame(a["inner"]).groupby("candidate").mae.mean()
        grid = (
            [{"alpha": v} for v in config["ridge_alphas"]]
            if family == "tuned_ridge"
            else config["hgb_grid"]
        )
        assert a["setting"] == grid[int(by_candidate.idxmin())]
        observed = (
            predictions.set_index("n_ema_id")
            .loc[valid.n_ema_id, "prediction_" + a["model"]]
            .to_numpy()
        )
        if family == "tuned_ridge":
            means = train[numeric].mean().to_numpy()
            scale = train[numeric].std(ddof=0).to_numpy()
            scale[scale == 0] = 1
            assert np.allclose(a["training_scale"], scale, atol=1e-12, rtol=1e-12)
            cats = sorted(train.n_survey_category.unique())

            def design(f):
                return np.column_stack(
                    [
                        (f[numeric].to_numpy() - means) / scale,
                        *[f.n_survey_category.eq(c).to_numpy(float) for c in cats],
                    ]
                )

            x, z = design(train), design(valid)
            xm = x.mean(axis=0)
            ym = train.pss_combo.mean()
            xx = x - xm
            coef = np.linalg.solve(
                xx.T @ xx + a["setting"]["alpha"] * np.eye(x.shape[1]),
                xx.T @ (train.pss_combo.to_numpy() - ym),
            )
            assert np.allclose(observed, (z - xm) @ coef + ym, atol=1e-9, rtol=1e-9)
        else:
            replay, _, _ = fit_predict(train, valid, view, family, a["setting"], config)
            assert np.array_equal(observed, replay)
        outer_count += 1
    metrics = pd.read_json(bundle / "metrics.json")
    metric_checks = 0
    for r in metrics.itertuples():
        f = predictions if r.stage == "oof" else training
        if str(r.fold) != "pooled":
            f = f[f.outer_fold.eq(int(r.fold))]
        if r.survey_category != "All":
            f = f[f.n_survey_category.eq(r.survey_category)]
        error = f["prediction_" + r.model] - f.pss_combo
        mae = error.abs().mean()
        rmse = np.sqrt(np.square(error).mean())
        pmae = error.abs().groupby(f.n_participant).mean().mean()
        assert np.allclose([mae, rmse, pmae], [r.mae, r.rmse, r.participant_mae], atol=1e-10)
        metric_checks += 1
    bootstrap_checks = 0
    for r in pd.read_json(bundle / "paired_intervals.json").itertuples():
        f = predictions
        d = (f["prediction_" + r.model] - f.pss_combo).abs() - (
            f["prediction_" + r.baseline] - f.pss_combo
        ).abs()
        arrays = [d[f.n_participant.eq(p)].to_numpy() for p in sorted(f.n_participant.unique())]
        sums = np.array([v.sum() for v in arrays])
        counts = np.array([len(v) for v in arrays])
        means = sums / counts
        ix = np.random.default_rng(config["bootstrap_seed"]).integers(
            0, len(sums), size=(config["bootstrap_draws"], len(sums))
        )
        a = np.array([sums[j].sum() / counts[j].sum() for j in ix])
        b = np.array([means[j].mean() for j in ix])
        assert np.allclose(
            [r.mae_difference, r.participant_mae_difference], [d.mean(), means.mean()], atol=1e-10
        )
        assert np.allclose(
            [r.mae_difference_low, r.mae_difference_high],
            np.quantile(a, [0.025, 0.975]),
            atol=1e-10,
        )
        assert np.allclose(
            [r.participant_mae_difference_low, r.participant_mae_difference_high],
            np.quantile(b, [0.025, 0.975]),
            atol=1e-10,
        )
        bootstrap_checks += 1
    snapshot = json.loads((root / "private/past_hr_reference/initial_snapshot.json").read_text())
    for name, h in snapshot["files"].items():
        if (
            name not in ["README.md", "src/garmin_inventory/privacy.py"]
            and sha256(root / name) != h
        ):
            raise AuditError("Existing public/source file changed.")
    for name, h in snapshot["prior_manifests"].items():
        p = root / name
        assert sha256(p) == h
        assert artifact_hashes(p.parent) == json.loads(p.read_text())["artifact_sha256"]
    return {
        **reference_checks,
        "outer_models_verified": outer_count,
        "inner_split_transform_checks": inner_count,
        "metric_rows_recomputed": metric_checks,
        "bootstrap_contrasts_recomputed": bootstrap_checks,
        "prior_private_manifests_preserved": len(snapshot["prior_manifests"]),
        "prior_public_files_preserved": len(snapshot["files"]) - 2,
        "predictor_allowlist_verified": True,
    }


def verify_retrospective_diagnostics(root, bundle):
    """Check full-record diagnostics separately; return aggregates only."""
    root, bundle = Path(root), Path(bundle)
    config = json.loads((bundle / "frozen_configuration.json").read_text())
    base = pd.read_parquet(root / config["baseline_bundle"] / "linked_development.parquet")
    accounting = pd.read_parquet(bundle / "accounting.parquet").set_index("n_ema_id")
    native = base[base.n_ema_id.map(accounting.native_60m_hr)]
    variance = pd.read_json(bundle / "variation.json")
    checked = 0
    for row in variance.itertuples():
        f = (base if row.scope == "all_development" else native)[
            ["n_participant", row.measure]
        ].dropna()
        values = f[row.measure].to_numpy()
        grand = sum(values) / len(values)
        within = 0.0
        between = 0.0
        person_means = []
        for _, g in f.groupby("n_participant"):
            y = g[row.measure].tolist()
            mean = sum(y) / len(y)
            person_means.append(mean)
            within += sum((v - mean) ** 2 for v in y)
            between += len(y) * (mean - grand) ** 2
        assert np.allclose(
            [row.within_ss, row.between_ss, row.total_ss],
            [within, between, sum((v - grand) ** 2 for v in values)],
            atol=1e-8,
            rtol=1e-12,
        )
        checked += 1
    for row in pd.read_json(bundle / "retrospective_associations.json").itertuples():
        if row.mode == "between_person_means":
            pairs = native.groupby("n_participant")[[row.feature, "pss_combo"]].mean()
            x, y = pairs[row.feature].to_numpy(), pairs.pss_combo.to_numpy()
        else:
            valid = native[native.groupby("n_participant").n_ema_id.transform("size").ge(2)]
            centered = valid[[row.feature, "pss_combo"]] - valid.groupby("n_participant")[
                [row.feature, "pss_combo"]
            ].transform("mean")
            x, y = centered[row.feature].to_numpy(), centered.pss_combo.to_numpy()
        slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1]
        assert np.allclose(
            [row.correlation, row.slope], [np.corrcoef(x, y)[0, 1], slope], atol=1e-10
        )
        checked += 1
    items = base[["PSS_NERVOUS_value", "PSS_COPE_value", "PSS_CONTROL_value"]]
    complete = items.notna().all(axis=1)
    assert np.array_equal(base.pss_combo.notna(), complete)
    assert np.allclose(
        base.loc[complete, "pss_combo"],
        items.loc[complete, "PSS_NERVOUS_value"]
        + 12
        - items.loc[complete, "PSS_COPE_value"]
        - items.loc[complete, "PSS_CONTROL_value"],
    )
    return dict(retrospective_scalar_checks=checked, unchanged_score_coding_verified=True)
