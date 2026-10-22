"""Read-only checks of all candidates, exact formulas, preprocessing and provenance."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from garmin_preprocessing.common import verify_bundle, write_json
from garmin_ema_baseline.linkage import check_development, score_items
from garmin_comprehensive.run import get_data
from garmin_comprehensive.models import CONTEXT
from garmin_comprehensive.features import observed_features, activity_features, coobserved
from garmin_comprehensive.augment import past_reference, history_name
from garmin_hr_activity.features import prepare, empty_frame
from garmin_pss_classification.data import targets, TASKS
from garmin_comprehensive_regression.evaluate import finite


def load(root):
    root = Path(root)
    cfg = json.loads((root / "config/complete_pipeline_review.json").read_text())
    for key in ["feature_bundle", "normalized_bundle", "ledger_bundle"]:
        verify_bundle(root / cfg[key])
    frame, blocks = get_data(root / cfg["feature_bundle"])
    entries = json.loads((root / cfg["feature_bundle"] / "dictionary.json").read_text())
    assignments = pd.read_parquet(root / "private/frozen_participant_assignments.parquet")
    check_development(frame, assignments, 3706)
    assert len(entries) == 678 and len(blocks["full"]) == 438 and len(blocks["pulseox_full"]) == 678
    assert len(set(d["name"] for d in entries)) == 678
    assert sum(d["modality"] == "pulseox" for d in entries) == 240
    assert not set(blocks["full"]) & {d["name"] for d in entries if d["modality"] == "pulseox"}
    assert not any(c.startswith("quality__") for c in blocks["full"])
    assert not any("stress" in c.lower() or "ibi" in c.lower() for c in blocks["full"])
    ema = (
        pd.read_parquet(root / cfg["normalized_bundle"] / "ema/development_ema.parquet")
        .set_index("n_ema_id")
        .loc[frame.n_ema_id]
    )
    scores = score_items(ema)
    np.testing.assert_allclose(scores.pss_combo, frame.pss_combo, equal_nan=True)
    for task in TASKS:
        np.testing.assert_allclose(
            targets(frame.pss_combo, task), frame["target_" + task], equal_nan=True
        )
    return cfg, frame, blocks, entries


def trace(root, cfg, frame, entries):
    root = Path(root)
    feature_cfg = json.loads((root / "config/comprehensive_features.json").read_text())
    hr_cfg = json.loads((root / "config/hr_activity.json").read_text())
    people = json.loads(
        (root / cfg["normalized_bundle"] / "participant_directories.json").read_text()
    )
    by_person = {p["participant"]: p for p in people}
    sample = frame.groupby("n_fold", group_keys=False).apply(
        lambda g: g.sort_values("n_ema_id").head(cfg["sample_responses_per_fold"]),
        include_groups=True,
    )
    stats = []
    checks = []
    physical_checks = 0
    for row in sample.itertuples():
        person = by_person[row.n_participant]
        directory = person["directory"]
        streams = {}
        for sensor, raw in [
            ("hr", "hr"),
            ("steps", "steps"),
            ("floors", "floorsClimbed"),
            ("pulseox", "pulseOx"),
        ]:
            if sensor == "hr":
                h = pd.read_parquet(
                    root / cfg["ledger_bundle"] / "reading_dispositions" / directory / "hr.parquet"
                )
                streams[sensor] = (
                    h[h.a_first & np.isfinite(h.n_epoch_s)].copy().reset_index(drop=True)
                )
                path = root / cfg["normalized_bundle"] / "physical" / directory / "hr.parquet"
                if path.exists():
                    physical = pd.read_parquet(
                        path,
                        columns=[
                            "ts",
                            "val",
                            "n_epoch_s",
                            "n_value",
                            "n_source_sha256",
                            "n_source_row",
                        ],
                    )
                    p = physical.head(3)
                    np.testing.assert_allclose(pd.to_numeric(p.ts), p.n_epoch_s, equal_nan=True)
                    np.testing.assert_allclose(pd.to_numeric(p.val), p.n_value, equal_nan=True)
                    physical_checks += len(p)
            elif sensor in ["steps", "floors"]:
                streams[sensor] = pd.read_parquet(
                    root / cfg["ledger_bundle"] / "counter_pairs" / directory / f"{raw}.parquet"
                )
            else:
                p = root / cfg["normalized_bundle"] / "logical" / directory / "pulseOx.parquet"
                f = pd.read_parquet(p) if p.exists() else empty_frame()
                streams[sensor] = prepare(f, "hr", {**hr_cfg, "hr_min_bpm": 70, "hr_max_bpm": 100})[
                    0
                ]
        recomputed = {}
        for m in feature_cfg["lookbacks_minutes"]:
            end = float(row.n_cutoff_epoch)
            start = end - 60 * int(m)
            for sensor, g in streams.items():
                values = (
                    observed_features(g, start, end, feature_cfg)[0]
                    if sensor in ["hr", "pulseox"]
                    else activity_features(g, start, end)[0]
                )
                recomputed.update({f"{sensor}__{k}__{m}m": v for k, v in values.items()})
            recomputed.update(
                {
                    f"interaction__{k}__{m}m": v
                    for k, v in coobserved(streams["hr"], streams["steps"], start, end).items()
                }
            )
            if m < 1440:
                earlier = observed_features(streams["hr"], start - 60 * int(m), start, feature_cfg)[
                    0
                ]
                for metric in ["mean", "sd"]:
                    recomputed[f"multiscale__adjacent_{metric}_difference__{m}m"] = (
                        recomputed[f"hr__{metric}__{m}m"] - earlier[metric]
                    )
            history, _ = past_reference(
                streams["hr"],
                end,
                m,
                recomputed[f"hr__mean__{m}m"],
                recomputed[f"hr__latest__{m}m"],
            )
            recomputed.update({history_name(k, m): v for k, v in history.items()})
        assert set(recomputed) == {d["name"] for d in entries}
        for d in entries:
            actual = recomputed[d["name"]]
            saved = getattr(row, d["name"])
            assert np.isclose(actual, saved, equal_nan=True, rtol=1e-10, atol=1e-10)
            checks.append(
                dict(
                    n_ema_id=row.n_ema_id,
                    n_participant=row.n_participant,
                    n_fold=row.n_fold,
                    feature=d["name"],
                    checked=True,
                    observed=bool(np.isfinite(saved)),
                )
            )
        stats.append(
            dict(
                fold=row.n_fold,
                checked=len(recomputed),
                observed=sum(np.isfinite(v) for v in recomputed.values()),
            )
        )
    return (
        pd.DataFrame(checks),
        dict(
            response_checks=len(sample),
            feature_value_checks=len(checks),
            physical_clock_value_checks=physical_checks,
            all_678_candidates_checked=True,
        ),
        pd.DataFrame(stats).groupby("fold").sum().reset_index(),
    )


def transform_audit(transform, train, valid, entries, checkpoint, task, fold):
    full = [d["name"] for d in entries if d["modality"] != "pulseox"]
    assert transform.columns == full
    np.testing.assert_allclose(transform.medians, train[transform.kept].median())
    filled = train[transform.kept].fillna(transform.medians)
    np.testing.assert_allclose(transform.scaler.mean_, filled.mean(), atol=1e-10)
    assert transform.encoder.categories_[0].tolist() == sorted(train.n_survey_category.unique())
    assert transform.flag_columns == [c for c in full if train[c].isna().nunique() > 1]
    expected = train[CONTEXT + full]
    expected_kept = [c for c in expected if expected[c].notna().any() and expected[c].nunique() > 1]
    assert transform.kept == expected_kept
    x = transform.transform(train)
    xv = transform.transform(valid)
    assert np.isfinite(x).all() and np.isfinite(xv).all()
    assert not set(train.n_participant) & set(valid.n_participant)
    names = transform.names
    assert len(names) == len(transform.kept) + len(transform.encoder.get_feature_names_out()) + len(
        transform.flag_columns
    )
    schema = []
    for name in full:
        schema.append(
            dict(
                checkpoint=checkpoint,
                task=task,
                fold=fold,
                feature=name,
                retained_numeric=name in transform.kept,
                dropped_reason=transform.dropped.get(name, ""),
                missingness_input="missing__" + name if name in transform.flag_columns else "",
                numeric_position=names.index(name) if name in names else None,
                missing_position=names.index("missing__" + name)
                if "missing__" + name in names
                else None,
            )
        )
    info = dict(
        checkpoint=checkpoint,
        task=task,
        fold=fold,
        candidate_physiology=438,
        provisional_used=0,
        retained_physiology=len(set(transform.kept) & set(full)),
        context_numeric=len(set(transform.kept) & set(CONTEXT)),
        survey_encoded=len(transform.encoder.get_feature_names_out()),
        missing_indicators=len(transform.flag_columns),
        actual_inputs=x.shape[1],
        training_rows=len(train),
        validation_rows=len(valid),
        training_participants=train.n_participant.nunique(),
        validation_participants=valid.n_participant.nunique(),
        dropped_numeric=len(transform.dropped),
    )
    encoded = [
        dict(
            checkpoint=checkpoint,
            task=task,
            fold=fold,
            position=i,
            name=n,
            kind="missingness"
            if n.startswith("missing__")
            else "physiological"
            if n in full
            else "context",
        )
        for i, n in enumerate(names)
    ]
    return info, schema, encoded


def check_inner(job, train, valid):
    m = pd.read_parquet(job / "inner_membership.parquet")
    for _, g in m.groupby("inner_fold"):
        a, b = g[g.role.eq("train")], g[g.role.eq("validation")]
        assert not set(a.n_participant) & set(b.n_participant)
        assert set(a.n_ema_id) | set(b.n_ema_id) == set(train.n_ema_id)
        assert not set(g.n_ema_id) & set(valid.n_ema_id)
    return m.inner_fold.nunique()


def model_replays(root, cfg, frame, entries):
    root = Path(root)
    transforms = []
    schemas = []
    encoded = []
    replays = 0
    inner = 0
    predictions = []
    supports = []
    settings = []
    for checkpoint, tasks, views in [
        (13, list(TASKS), ["context", "full"]),
        (15, ["A", "B"], ["context_u", "multiscale_u"]),
    ]:
        folder = root / cfg["classification_run" if checkpoint == 13 else "binary_run"]
        for task in tasks:
            target = "target_" + (
                task if checkpoint == 13 else "at_least_5" if task == "A" else "at_least_10"
            )
            population = frame[frame[target].notna() & frame.context_available]
            for view in views:
                for fold in range(5):
                    job = folder / "jobs" / f"{task}__{view}__fold{fold}"
                    verify_bundle(job)
                    info = json.loads((job / "summary.json").read_text())
                    assert info["status"] == "complete"
                    train = population[population.n_fold.ne(fold)]
                    valid = population[population.n_fold.eq(fold)]
                    obj = joblib.load(job / "estimator.joblib")
                    transform, model = (
                        (obj["transform"], obj["model"])
                        if checkpoint == 13
                        else (obj.transform, obj.model)
                    )
                    with threadpool_limits(limits=1):
                        probs = model.predict_proba(transform.transform(valid))
                    saved = pd.read_parquet(
                        job
                        / (
                            "predictions.parquet"
                            if checkpoint == 13
                            else "validation_predictions.parquet"
                        )
                    )
                    assert saved.n_ema_id.tolist() == valid.n_ema_id.tolist()
                    expected = (
                        saved[[f"p{k}" for k in range(probs.shape[1])]].to_numpy()
                        if checkpoint == 13
                        else saved.probability.to_numpy()
                    )
                    np.testing.assert_allclose(
                        probs if checkpoint == 13 else probs[:, 1], expected, atol=1e-10, rtol=1e-10
                    )
                    np.testing.assert_array_equal(saved.target, valid[target])
                    pred = saved[["n_ema_id", "n_participant", "n_fold", "target"]].copy()
                    for k in range(probs.shape[1]):
                        pred[f"p{k}"] = probs[:, k]
                    pred["checkpoint"] = checkpoint
                    pred["task"] = task
                    pred["view"] = view
                    predictions.append(pred)
                    for role, z in [("training", train), ("validation", valid)]:
                        for k in range(probs.shape[1]):
                            supports.append(
                                dict(
                                    checkpoint=checkpoint,
                                    task=task,
                                    view=view,
                                    fold=fold,
                                    role=role,
                                    label=k,
                                    responses=int(z[target].eq(k).sum()),
                                    participants=z.loc[z[target].eq(k), "n_participant"].nunique(),
                                )
                            )
                    settings.append(
                        dict(
                            checkpoint=checkpoint,
                            task=task,
                            view=view,
                            fold=fold,
                            selected=json.dumps(info["best"], sort_keys=True),
                        )
                    )
                    if view in ["full", "multiscale_u"]:
                        counts, mapping, names = transform_audit(
                            transform, train, valid, entries, checkpoint, task, fold
                        )
                        transforms.append(counts)
                        schemas.extend(mapping)
                        encoded.extend(names)
                    inner += check_inner(job, train, valid)
                    replays += 1
    return dict(
        transforms=pd.DataFrame(transforms),
        mapping=pd.DataFrame(schemas),
        encoded=pd.DataFrame(encoded),
        predictions=pd.concat(predictions, ignore_index=True),
        support=pd.DataFrame(supports),
        settings=pd.DataFrame(settings),
        verification=dict(classification_model_replays=replays, inner_separation_checks=inner),
    )


def execute(root):
    root = Path(root)
    public = root / "reports/complete_pipeline_review"
    private = root / "private/pipeline_review"
    cfg, frame, blocks, entries = load(root)
    checks, verification, fold_trace = trace(root, cfg, frame, entries)
    checks.to_parquet(private / "feature_trace.parquet", index=False)
    with threadpool_limits(limits=1):
        results = model_replays(root, cfg, frame, entries)
    for key in ["transforms", "mapping", "encoded", "support", "settings"]:
        results[key].to_csv(public / f"{key}.csv", index=False)
    results["predictions"].to_parquet(private / "classification_replays.parquet", index=False)
    fold_trace.to_csv(public / "feature_trace_checks.csv", index=False)
    dictionary = pd.DataFrame(entries)
    dictionary.to_csv(public / "feature_dictionary.csv", index=False)
    count = dictionary.groupby(["modality", "status"]).size().reset_index(name="candidate_columns")
    count.to_csv(public / "feature_counts.csv", index=False)
    saved = pd.read_parquet(root / cfg["feature_bundle"] / "development_features.parquet")
    shapes = pd.DataFrame(
        [
            dict(dataset="all development candidates", rows=len(saved), columns=678),
            dict(dataset="supported candidate block", rows=len(saved), columns=438),
            dict(dataset="provisional PulseOx block", rows=len(saved), columns=240),
            dict(
                dataset="saved feature frame including metadata/context/targets",
                rows=len(saved),
                columns=saved.shape[1],
            ),
            dict(
                dataset="valid outcome/context modeling rows",
                rows=int((frame.pss_combo.notna() & frame.context_available).sum()),
                columns=438,
            ),
        ]
    )
    shapes.to_csv(public / "dataset_shapes.csv", index=False)
    coverage = pd.DataFrame(
        json.loads((root / cfg["feature_bundle"] / "coverage.json").read_text())
    )
    coverage.to_csv(public / "feature_availability.csv", index=False)
    verification.update(
        results["verification"],
        all_outcome_checks=len(frame),
        all_label_checks=len(frame) * 3,
        development_participants=frame.n_participant.nunique(),
        final_test_evaluated=False,
        new_model_searches=0,
    )
    write_json(public / "audit.json", finite(verification))
    return verification
