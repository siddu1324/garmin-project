"""Assemble verified existing outputs; no raw preprocessing or model fitting."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from garmin_comprehensive.run import get_data
from garmin_ema_baseline.linkage import check_development, score_items
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import verify_bundle
from garmin_pss_classification.data import TASKS, targets
from garmin_pss_classification.metrics import score

PUBLIC = "reports/central_research_overview"
REVIEW = "reports/complete_pipeline_review"


def read_json(path):
    return json.loads(Path(path).read_text())


def check_seal(root):
    seal = read_json(root / REVIEW / "completion.json")
    assert seal["complete"] and not seal["final_test_evaluated"]
    for name, expected in seal["sha256"].items():
        assert sha256(root / name) == expected, f"Prior review changed: {name}"
    return len(seal["sha256"])


def missing_summary(dictionary, frame, groups):
    """Count missing feature cells, not unique missing responses."""
    rows = []
    for key, entries in dictionary.groupby(groups, observed=True):
        keys = key if isinstance(key, tuple) else (key,)
        missing = int(frame[entries.name].isna().sum().sum())
        cells = len(frame) * len(entries)
        rows.append(
            dict(zip(groups, keys))
            | dict(
                features=len(entries),
                responses=len(frame),
                feature_cells=cells,
                missing_cells=missing,
                missing_pct=100 * missing / cells,
            )
        )
    return pd.DataFrame(rows)


def existing_results(root):
    cfg = read_json(root / "config/complete_pipeline_review.json")
    old = pd.read_json(root / "reports/pss_classification/8970bf5ebcc47bfd/metrics.json")
    old = old[
        old.task.eq("three_class")
        & old.minutes.eq(60)
        & old.cohort.eq("joint")
        & old.handling.eq("complete")
        & old.weighting.eq("none")
        & old.view.isin(["context_hr_steps", "dummy"])
    ].copy()
    current = pd.read_json(root / cfg["classification_report"] / "metrics.json")
    comparisons = current[
        current.task.eq("three_class")
        & current.cohort.eq("all_valid")
        & current.view.isin(
            [
                "dummy_majority",
                "context",
                "original60",
                "full",
                "full_corr",
                "full_l1",
                "full_elastic",
            ]
        )
    ].copy()
    paired = pd.read_json(root / cfg["classification_report"] / "paired_differences.json")
    paired = paired[paired.model.eq("full") & paired.baseline.eq("context")]
    return comparisons, old, paired


def prepare(root):
    root = Path(root)
    check_seal(root)
    out = root / PUBLIC
    out.mkdir(exist_ok=True)
    cfg = read_json(root / "config/complete_pipeline_review.json")
    verify_bundle(root / cfg["feature_bundle"])
    frame, blocks = get_data(root / cfg["feature_bundle"])
    dictionary = pd.DataFrame(read_json(root / cfg["feature_bundle"] / "dictionary.json"))
    assert len(dictionary) == 678 and dictionary.name.is_unique
    assert len(blocks["full"]) == 438 and len(blocks["pulseox_full"]) == 678
    assert set(dictionary.loc[dictionary.modality.eq("pulseox"), "name"]).isdisjoint(blocks["full"])
    assignments = pd.read_parquet(root / "private/frozen_participant_assignments.parquet")
    check_development(frame, assignments, 3706)
    inventory = read_json(root / "reports/inventory.json")
    assert (
        sha256(root / "private/frozen_participant_assignments.parquet")
        == inventory["frozen_split"]["assignment_sha256"]
    )
    ema = pd.read_parquet(root / cfg["normalized_bundle"] / "ema/development_ema.parquet")
    ema = ema.set_index("n_ema_id").loc[frame.n_ema_id]
    np.testing.assert_allclose(score_items(ema).pss_combo, frame.pss_combo, equal_nan=True)
    for task in TASKS:
        np.testing.assert_allclose(
            targets(frame.pss_combo, task), frame["target_" + task], equal_nan=True
        )
    normalized = read_json(root / cfg["normalized_bundle"] / "manifest.json")
    assert normalized["final_test_rows_written"] == 0
    sensor_rows = []
    usage = dict(
        hr="438 supported block",
        steps="438 supported block",
        floorsClimbed="438 supported block",
        pulseOx="240 provisional; sensitivity only",
        ibi="Limited pilot only; no HRV predictors",
        calories="Excluded",
        intensityMinutes="Excluded",
        stress="Excluded",
    )
    for sensor, physical in normalized["sensor_rows"].items():
        paths = list((root / cfg["normalized_bundle"] / "logical").glob(f"*/{sensor}.parquet"))
        logical = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
        sensor_rows.append(
            dict(
                modality=sensor,
                physical_rows=physical,
                logical_rows=logical,
                duplicate_excess=physical - logical,
                participants=len(paths),
                use=usage[sensor],
            )
        )
    sensors = pd.DataFrame(sensor_rows)
    assert sensors.logical_rows.sum() == normalized["counts"]["logical_sensor_rows"]
    dictionary["responses"] = len(frame)
    dictionary["missing_count"] = [int(frame[c].isna().sum()) for c in dictionary.name]
    dictionary["missing_pct"] = dictionary.missing_count / len(frame) * 100
    saved = (
        pd.read_csv(root / REVIEW / "feature_availability.csv")
        .query('survey == "All"')
        .set_index("feature")
    )
    np.testing.assert_array_equal(dictionary.missing_count, saved.loc[dictionary.name, "missing"])
    assert saved.responses.eq(len(frame)).all()
    dictionary.to_csv(out / "complete_feature_inventory.csv", index=False)
    missing_modality = missing_summary(dictionary, frame, ["modality"])
    missing_windows = missing_summary(dictionary, frame, ["modality", "lookback_minutes"])
    for name, table in [
        ("sensor_inventory", sensors),
        ("missingness_modality", missing_modality),
        ("missingness_lookback", missing_windows),
    ]:
        table.to_csv(out / f"{name}.csv", index=False)
    valid = frame[frame.pss_combo.notna() & frame.context_available]
    assert len(valid) == 3689
    assert inventory["structural"]["ema_rows"] == 4680
    assert inventory["structural"]["ema_participants"] == 157
    folds = (
        pd.read_csv(root / REVIEW / "transforms.csv")
        .query('checkpoint == 13 and task == "three_class"')
        .copy()
    )
    for row in folds.itertuples():
        tr, va = valid[valid.n_fold.ne(row.fold)], valid[valid.n_fold.eq(row.fold)]
        assert not set(tr.n_participant) & set(va.n_participant)
        assert (len(tr), len(va)) == (row.training_rows, row.validation_rows)
        assert (tr.n_participant.nunique(), va.n_participant.nunique()) == (
            row.training_participants,
            row.validation_participants,
        )
    support = pd.read_csv(root / REVIEW / "support.csv").query(
        'checkpoint == 13 and task == "three_class" and view == "full"'
    )
    support = support.pivot(index="fold", columns=["role", "label"], values="responses")
    for fold in range(5):
        for role, cohort in [
            ("training", valid[valid.n_fold.ne(fold)]),
            ("validation", valid[valid.n_fold.eq(fold)]),
        ]:
            for label in range(3):
                assert support.loc[fold, (role, label)] == cohort.target_three_class.eq(label).sum()
    support.columns = [f"{role} class {label}" for role, label in support.columns]
    settings = pd.read_csv(root / REVIEW / "settings.csv").query(
        'checkpoint == 13 and task == "three_class" and view == "full"'
    )
    folds["selected_C"] = settings.selected.map(lambda v: json.loads(v)["C"]).to_numpy()
    predictions = pd.read_parquet(root / "private/pipeline_review/classification_replays.parquet")
    metrics = pd.read_csv(root / REVIEW / "classification_metrics.csv")
    checked = 0
    for (checkpoint, task, view), group in predictions.groupby(["checkpoint", "task", "view"]):
        assert group.n_ema_id.is_unique and set(group.n_ema_id) == set(valid.n_ema_id)
        linked = valid.set_index("n_ema_id").loc[group.n_ema_id]
        assert linked.n_participant.tolist() == group.n_participant.tolist()
        np.testing.assert_array_equal(linked.n_fold, group.n_fold)
        task_name = task if checkpoint == 13 else "at_least_5" if task == "A" else "at_least_10"
        np.testing.assert_array_equal(linked["target_" + task_name], group.target)
        k = 3 if task == "three_class" else 2
        prob = group[[f"p{i}" for i in range(k)]].to_numpy()
        actual, _, _ = score(
            group.target.to_numpy(int), prob.argmax(axis=1), prob, group.n_participant.to_numpy(), k
        )
        ref = metrics[
            metrics.checkpoint.eq(checkpoint) & metrics.task.eq(task) & metrics.view.eq(view)
        ].iloc[0]
        for metric in ["roc_auc", "macro_f1", "balanced_accuracy", "accuracy", "elevated_recall"]:
            np.testing.assert_allclose(actual[metric], ref[metric], atol=1e-12)
        checked += 1
    encoded = pd.read_csv(root / REVIEW / "encoded.csv").query(
        'checkpoint == 13 and task == "three_class"'
    )
    encoded.to_csv(out / "actual_model_inputs_by_fold.csv", index=False)
    comparisons, earlier, paired = existing_results(root)
    coverage = pd.read_csv(root / "reports/measurement_alignment/e5359ea4cc4b4efa/coverage.csv")
    coverage = coverage[coverage.grouping.eq("sensor/lookback_minutes") & coverage.sensor.eq("hr")]
    shapes = pd.read_csv(root / REVIEW / "dataset_shapes.csv")
    provenance = pd.DataFrame(
        [
            ["Notebooks reviewed", "00–20"],
            ["Features", cfg["feature_bundle"]],
            ["Classification", cfg["classification_report"]],
            ["Latest full-block binary", cfg["binary_report"]],
            ["Timing / recording audit", "reports/measurement_alignment/e5359ea4cc4b4efa"],
            ["Execution", "Read-only validation and summaries; no new fits or raw preprocessing"],
        ],
        columns=["source", "value"],
    )
    return dict(
        root=root,
        dictionary=dictionary,
        sensors=sensors,
        normalized=normalized,
        inventory=inventory,
        shapes=shapes,
        folds=folds,
        support=support,
        missing_modality=missing_modality,
        missing_windows=missing_windows,
        metrics=metrics,
        comparisons=comparisons,
        earlier=earlier,
        paired=paired,
        coverage=coverage,
        provenance=provenance,
        checks=checked,
        complete_supported=int(frame[blocks["full"]].notna().all(axis=1).sum()),
        missing_scores=int(frame.pss_combo.isna().sum()),
    )


def missing_plot(data):
    values = data["missing_windows"].pivot(
        index="modality", columns="lookback_minutes", values="missing_pct"
    )
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(values, vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
    ax.set(
        xticks=range(len(values.columns)),
        xticklabels=values.columns,
        yticks=range(len(values)),
        yticklabels=values.index,
        xlabel="Lookback (minutes)",
        title="Missing feature cells (%) · 3,706 responses · — = not generated",
    )
    for (i, j), v in np.ndenumerate(values.to_numpy()):
        ax.text(
            j,
            i,
            f"{v:.0f}" if np.isfinite(v) else "—",
            ha="center",
            va="center",
            color="white" if v > 70 else "black",
        )
    fig.colorbar(im, ax=ax, label="Missing (%)")
    fig.tight_layout()
    return fig


def confusion_plot(data):
    matrices = read_json(data["root"] / REVIEW / "confusion_matrices.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, view in zip(axes, ["context", "full"]):
        m = next(
            x
            for x in matrices
            if x["checkpoint"] == 13 and x["task"] == "three_class" and x["view"] == view
        )
        cm = np.array(m["matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), n in np.ndenumerate(cm):
            ax.text(
                j,
                i,
                str(n),
                ha="center",
                va="center",
                color="white" if n > cm.max() / 2 else "black",
            )
        ax.set(
            title="Context" if view == "context" else "Context + 438 supported candidates",
            xlabel="Predicted class",
            ylabel="Observed class",
            xticks=range(3),
            yticks=range(3),
        )
    fig.suptitle("Notebook 13 · matched OOF responses (n=3,689); classes 0 / 1 / 2")
    fig.tight_layout()
    return fig
