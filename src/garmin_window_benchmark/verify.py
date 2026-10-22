"""Independent arithmetic and split/provenance verification for Notebook 09."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from garmin_ema_baseline.models import model_columns
from garmin_inventory.privacy import AuditError, sha256
from garmin_preprocessing.common import artifact_hashes, digest_object, verify_bundle


def verify(root, bundle, replay_models=False):
    root, bundle = Path(root), Path(bundle)
    manifest = verify_bundle(bundle)
    config = json.loads((bundle / "frozen_configuration.json").read_text())
    source = pd.read_parquet(root / config["baseline_bundle"] / "linked_development.parquet")
    predictions = pd.read_parquet(bundle / "out_of_fold_predictions.parquet")
    training = pd.read_parquet(bundle / "training_predictions.parquet")
    membership = pd.read_parquet(bundle / "membership.parquet")
    audit = json.loads((bundle / "training_audit.json").read_text())
    members = {
        b: source.set_index("n_ema_id")
        .loc[f.loc[f.included, "n_ema_id"]]
        .reset_index()
        .sort_values("n_ema_id")
        for b, f in membership.groupby("block")
    }
    settings = pd.read_json(bundle / "selected_settings.json")
    inner_checks = 0
    outer_checks = 0
    for record in audit:
        cohort = members[record["block"]]
        fold = record["fold"]
        tr, va = cohort[cohort.n_fold.ne(fold)], cohort[cohort.n_fold.eq(fold)]
        assert not set(tr.n_participant) & set(va.n_participant)
        assert record["training_key_sha256"] == digest_object(tr.n_ema_id.tolist())
        assert record["validation_key_sha256"] == digest_object(va.n_ema_id.tolist())
        family, view = record["model"].split("__")
        numeric, category = model_columns(view, 60)
        assert record["input_columns"] == numeric + category
        assert np.allclose(record["training_mean"], tr[numeric].mean(), rtol=1e-12, atol=1e-12)
        scale = np.sqrt(tr[numeric].var(ddof=0))
        scale[scale == 0] = 1
        assert np.allclose(record["training_scale"], scale, atol=1e-12, rtol=1e-12)
        assert record["categories"] == sorted(tr.n_survey_category.unique())
        inner = list(GroupKFold(3).split(tr, groups=tr.n_participant))
        for check in record["inner"]:
            a, b = inner[check["inner_fold"]]
            it, iv = tr.iloc[a], tr.iloc[b]
            assert not set(it.n_participant) & set(iv.n_participant)
            assert not (set(it.n_participant) | set(iv.n_participant)) & set(va.n_participant)
            assert check["training_key_sha256"] == digest_object(it.n_ema_id.tolist())
            assert check["validation_key_sha256"] == digest_object(iv.n_ema_id.tolist())
            assert np.allclose(check["training_mean"], it[numeric].mean(), rtol=1e-12, atol=1e-12)
            assert check["categories"] == sorted(it.n_survey_category.unique())
            assert check["training_count"] == len(it)
            inner_checks += 1
        if record["inner"]:
            mean_scores = pd.DataFrame(record["inner"]).groupby("candidate").mae.mean()
            best = int(mean_scores.idxmin())
            grid = (
                [{"alpha": a} for a in config["ridge_alphas"]]
                if family == "tuned_ridge"
                else config["hgb_grid"]
            )
            assert record["setting"] == grid[best]
            selection = settings[
                (settings.block == record["block"])
                & (settings.fold == fold)
                & (settings.model == record["model"])
            ].iloc[0]
            assert np.isclose(selection.inner_mae, mean_scores.loc[best], atol=1e-11)
        actual = (
            predictions[(predictions.block == record["block"]) & (predictions.n_fold == fold)]
            .set_index("n_ema_id")
            .loc[va.n_ema_id, "prediction_" + record["model"]]
            .to_numpy()
        )
        if family != "hgb":
            # Independent closed-form ridge; center to leave the intercept unpenalized.
            cats = sorted(tr.n_survey_category.unique())

            def matrix(f):
                numeric_x = (
                    f[numeric].to_numpy() - tr[numeric].mean().to_numpy()
                ) / scale.to_numpy()
                return np.column_stack(
                    [numeric_x, *[f.n_survey_category.eq(c).to_numpy(float) for c in cats]]
                )

            x, z = matrix(tr), matrix(va)
            xm = x.mean(axis=0)
            ym = tr.pss_combo.mean()
            center = x - xm
            coef = np.linalg.solve(
                center.T @ center + record["setting"]["alpha"] * np.eye(x.shape[1]),
                center.T @ (tr.pss_combo.to_numpy() - ym),
            )
            independent = (z - xm) @ coef + ym
            assert np.allclose(actual, independent, atol=1e-9, rtol=1e-9)
        elif replay_models:
            from .models import fit_predict

            replay, _, _ = fit_predict(tr, va, view, family, record["setting"], config)
            assert np.array_equal(actual, replay)
        outer_checks += 1
    score_checks = 0
    scores = pd.read_json(bundle / "metrics.json")
    for _, row in scores.iterrows():
        f = predictions if row.stage == "oof" else training
        f = f[f.block == row.block]
        if str(row.fold) != "pooled":
            f = f[f.outer_fold.eq(int(row.fold))]
        if row.survey_category != "All":
            f = f[f.n_survey_category.eq(row.survey_category)]
        error = (f["prediction_" + row.model] - f.pss_combo).to_numpy()
        absolute = np.abs(error)
        people = {
            p: absolute[f.n_participant.to_numpy() == p].mean() for p in f.n_participant.unique()
        }
        assert np.allclose(
            [row.mae, row.rmse, row.participant_mae],
            [absolute.mean(), np.sqrt(np.mean(error * error)), np.mean(list(people.values()))],
            atol=1e-10,
            rtol=1e-10,
        )
        score_checks += 1
    bootstrap_checks = 0
    for row in pd.read_json(bundle / "paired_intervals.json").itertuples():
        f = predictions[predictions.block == row.block]
        delta = np.abs(f["prediction_" + row.model] - f.pss_combo) - np.abs(
            f["prediction_" + row.baseline] - f.pss_combo
        )
        person_values = [
            delta[f.n_participant.eq(p)].to_numpy() for p in sorted(f.n_participant.unique())
        ]
        sums = np.array([a.sum() for a in person_values])
        sizes = np.array([len(a) for a in person_values])
        means = sums / sizes
        draws = np.random.default_rng(config["bootstrap_seed"]).integers(
            0, len(sums), size=(config["bootstrap_draws"], len(sums))
        )
        response = np.array([sums[ix].sum() / sizes[ix].sum() for ix in draws])
        person = np.array([means[ix].mean() for ix in draws])
        assert np.allclose(
            [row.mae_difference, row.participant_mae_difference],
            [delta.mean(), means.mean()],
            atol=1e-10,
        )
        assert np.allclose(
            [row.mae_difference_low, row.mae_difference_high],
            np.quantile(response, [0.025, 0.975]),
            atol=1e-10,
        )
        assert np.allclose(
            [row.participant_mae_difference_low, row.participant_mae_difference_high],
            np.quantile(person, [0.025, 0.975]),
            atol=1e-10,
        )
        bootstrap_checks += 1
    snapshot = json.loads((root / "private/window_benchmark/initial_snapshot.json").read_text())
    for name, h in snapshot["files"].items():
        if (
            name not in ["README.md", "src/garmin_inventory/privacy.py"]
            and sha256(root / name) != h
        ):
            raise AuditError("Pre-existing public file changed.")
    for name, h in snapshot["prior_manifests"].items():
        path = root / name
        if sha256(path) != h:
            raise AuditError("Prior manifest changed.")
        old = json.loads(path.read_text())
        if artifact_hashes(path.parent) != old["artifact_sha256"]:
            raise AuditError("Prior immutable artifacts changed.")
    return dict(
        outer_learned_fits_verified=outer_checks,
        inner_split_transform_checks=inner_checks,
        independent_metric_rows=score_checks,
        independent_bootstrap_contrasts=bootstrap_checks,
        hgb_outer_predictions_replayed=replay_models,
        prior_manifests_preserved=len(snapshot["prior_manifests"]),
        prior_public_files_preserved=len(snapshot["files"]) - 2,
        development_only=manifest["development_only"],
    )
