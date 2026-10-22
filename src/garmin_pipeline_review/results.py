"""Recompute published scores from saved estimators without searching or changing cohorts."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from garmin_preprocessing.common import verify_bundle, write_json
from garmin_pss_classification.metrics import score
from garmin_comprehensive_regression.run import basic
from garmin_comprehensive_regression.evaluate import finite
from .audit import load


def run(root):
    root = Path(root)
    cfg, frame, _, _ = load(root)
    public = root / "reports/complete_pipeline_review"
    private = root / "private/pipeline_review"
    figures = public / "figures"
    figures.mkdir(exist_ok=True)
    p = pd.read_parquet(private / "classification_replays.parquet")
    metrics = []
    matrices = []
    per_class = []
    reference13 = pd.read_json(root / cfg["classification_report"] / "metrics.json")
    reference15 = pd.read_json(root / cfg["binary_report"] / "metrics.json")
    for (checkpoint, task, view), g in p.groupby(["checkpoint", "task", "view"]):
        k = 3 if task == "three_class" else 2
        probabilities = g[[f"p{i}" for i in range(k)]].to_numpy()
        y = g.target.to_numpy(int)
        result, cm, per = score(
            y, probabilities.argmax(axis=1), probabilities, g.n_participant.to_numpy(), k
        )
        if checkpoint == 13:
            ref = reference13[reference13.task.eq(task) & reference13.view.eq(view)].iloc[0]
            for name in ["roc_auc", "macro_f1", "balanced_accuracy"]:
                np.testing.assert_allclose(result[name], ref[name], atol=1e-12)
        else:
            ref = reference15[reference15.task.eq(task) & reference15.model.eq(view)].iloc[0]
            for name, other in [("roc_auc", "auroc"), ("balanced_accuracy", "balanced_accuracy")]:
                np.testing.assert_allclose(result[name], ref[other], atol=1e-12)
        metrics.append(
            dict(
                checkpoint=checkpoint,
                task=task,
                view=view,
                responses=len(g),
                participants=g.n_participant.nunique(),
                **result,
            )
        )
        matrices.append(dict(checkpoint=checkpoint, task=task, view=view, matrix=cm.tolist()))
        for i in range(k):
            per_class.append(
                dict(
                    checkpoint=checkpoint,
                    task=task,
                    view=view,
                    label=i,
                    support=int(per["support"][i]),
                    precision=per["precision"][i],
                    recall=per["recall"][i],
                    f1=per["f1"][i],
                )
            )
        # Same validation response keys for each comparison, within each task/checkpoint.
        baseline = p[
            p.checkpoint.eq(checkpoint)
            & p.task.eq(task)
            & p.view.eq("context" if checkpoint == 13 else "context_u")
        ]
        assert set(baseline.n_ema_id) == set(g.n_ema_id)
    m = pd.DataFrame(metrics)
    m.to_csv(public / "classification_metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(public / "per_class.csv", index=False)
    write_json(public / "confusion_matrices.json", finite(matrices))
    # All seven regression families, each with matching context and full features.
    population = frame[frame.pss_combo.notna() & frame.context_available]
    reg = []
    regpred = []
    replays = 0
    saved_metrics = pd.read_json(root / cfg["regression_report"] / "metrics.json")
    for family in ["ridge", "lasso", "elastic", "huber", "hgb", "xgb", "lgbm"]:
        for view in ["context", "full"]:
            parts = []
            for fold in range(5):
                job = (
                    root
                    / cfg["regression_run"]
                    / "jobs"
                    / f"all_valid__{view}__{family}__fold{fold}"
                )
                verify_bundle(job)
                fitted = joblib.load(job / "estimator.joblib")
                valid = population[population.n_fold.eq(fold)]
                train = population[population.n_fold.ne(fold)]
                saved = pd.read_parquet(job / "predictions.parquet")
                assert saved.n_ema_id.tolist() == valid.n_ema_id.tolist()
                assert not set(train.n_participant) & set(valid.n_participant)
                np.testing.assert_allclose(
                    fitted.transform.medians, train[fitted.transform.kept].median()
                )
                with threadpool_limits(limits=1):
                    actual = fitted.predict(valid)
                np.testing.assert_allclose(actual, saved.prediction, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(saved.dummy_mean, train.pss_combo.mean())
                np.testing.assert_allclose(saved.dummy_median, train.pss_combo.median())
                parts.append(saved)
                replays += 1
            joined = pd.concat(parts, ignore_index=True).sort_values("n_ema_id")
            regpred.append(joined.assign(view=view, family=family))
            result = basic(joined.pss_combo.to_numpy(), joined.prediction.to_numpy())
            ref = saved_metrics[
                saved_metrics.cohort.eq("all_valid")
                & saved_metrics.view.eq(view)
                & saved_metrics.family.eq(family)
            ].iloc[0]
            for metric in ["mae", "rmse", "r2", "spearman"]:
                np.testing.assert_allclose(result[metric], ref[metric], atol=1e-12)
            reg.append(
                dict(
                    checkpoint=14,
                    view=view,
                    family=family,
                    responses=len(joined),
                    participants=joined.n_participant.nunique(),
                    **result,
                    mae_low=ref.mae_low,
                    mae_high=ref.mae_high,
                )
            )
    for dummy in ["mean", "median"]:
        example = regpred[0]
        ref = saved_metrics[
            saved_metrics.cohort.eq("all_valid")
            & saved_metrics.model.str.contains("dummy_" + dummy)
        ]
        result = basic(example.pss_combo.to_numpy(), example["dummy_" + dummy].to_numpy())
        reg.append(
            dict(
                checkpoint=14,
                view="dummy",
                family=dummy,
                responses=len(example),
                participants=example.n_participant.nunique(),
                **result,
            )
        )
    regression = pd.DataFrame(reg)
    regression.to_csv(public / "regression_metrics.csv", index=False)
    pd.concat(regpred, ignore_index=True).to_parquet(
        private / "regression_replays.parquet", index=False
    )
    # Reuse all previously published matched uncertainty tables, retaining cohort labels.
    for label, path in [
        ("classification", cfg["classification_report"]),
        ("regression", cfg["regression_report"]),
        ("binary", cfg["binary_report"]),
    ]:
        for name in ["paired_differences.json", "fold_metrics.json"]:
            source = root / path / name
            if source.exists():
                (public / f"{label}_{name}").write_bytes(source.read_bytes())
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140}
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    z = m[m.checkpoint.eq(13) & m.task.eq("three_class")]
    for ax, key in zip(axes, ["roc_auc", "macro_f1", "balanced_accuracy"]):
        ax.bar(z.view, z[key], color=["#9caeb8", "#277b85"])
        ax.set_title(key.replace("_", " "))
        ax.set_ylim(0, 0.65)
    fig.suptitle("Notebook 13 · matched 3,689 responses · three-class logistic")
    fig.tight_layout()
    fig.savefig(figures / "classification.png", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, view in zip(axes, ["context", "full"]):
        cm = np.array(
            next(
                r["matrix"]
                for r in matrices
                if r["checkpoint"] == 13 and r["task"] == "three_class" and r["view"] == view
            )
        )
        ax.imshow(cm, cmap="Blues")
        for (i, j), n in np.ndenumerate(cm):
            ax.text(
                j,
                i,
                str(n),
                ha="center",
                va="center",
                color="white" if n > cm.max() / 2 else "#18313e",
            )
        ax.set(
            title=view,
            xlabel="Predicted class",
            ylabel="Observed class",
            xticks=range(3),
            yticks=range(3),
        )
    fig.tight_layout()
    fig.savefig(figures / "confusion.png", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, task in zip(axes, ["A", "B"]):
        z = m[m.checkpoint.eq(15) & m.task.eq(task)]
        ax.bar(z.view, z.roc_auc, color=["#9caeb8", "#277b85"])
        ax.axhline(0.5, color="gray", ls="--")
        ax.set_ylim(0.35, 0.65)
        ax.set(title=f"Notebook 15 · PSS ≥{5 if task == 'A' else 10}", ylabel="AUROC")
    fig.tight_layout()
    fig.savefig(figures / "binary.png", bbox_inches="tight")
    plt.close(fig)
    z = regression[regression.view.isin(["full", "dummy"])]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(z.family, z.mae, color=["#277b85"] * 7 + ["#9caeb8"] * 2)
    ax.set(ylabel="OOF MAE", title="Notebook 14 · same 3,689 responses · full 438 candidates")
    ax.axhline(z.loc[z.family.eq("median"), "mae"].iloc[0], color="#b05745", ls="--")
    fig.tight_layout()
    fig.savefig(figures / "regression.png", bbox_inches="tight")
    plt.close(fig)
    # Aggregate score-band predictions; never display participant-level scatter points.
    allreg = pd.concat(regpred, ignore_index=True)
    allreg["score_band"] = pd.cut(allreg.pss_combo, [2, 4, 9, 15], labels=["3–4", "5–9", "10–15"])
    strata = (
        allreg.groupby(["view", "family", "score_band"], observed=True)
        .apply(
            lambda g: pd.Series(
                dict(
                    responses=len(g),
                    actual_mean=g.pss_combo.mean(),
                    predicted_mean=g.prediction.mean(),
                    mae=np.abs(g.prediction - g.pss_combo).mean(),
                )
            ),
            include_groups=False,
        )
        .reset_index()
    )
    strata.to_csv(public / "regression_score_bands.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    for family in ["ridge", "elastic", "xgb", "lgbm"]:
        z = strata[strata.view.eq("full") & strata.family.eq(family)]
        ax.plot(z.score_band.astype(str), z.predicted_mean, "o-", label=family)
    ax.plot(z.score_band.astype(str), z.actual_mean, "k--", label="Observed mean")
    ax.set(
        ylabel="Mean composite score",
        title="Aggregate score bands · predictions remain near the middle",
    )
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(figures / "score_bands.png", bbox_inches="tight")
    plt.close(fig)
    write_json(
        public / "result_verification.json",
        finite(
            dict(
                regression_model_replays=replays,
                classification_metric_groups=len(m),
                regression_metric_groups=len(regression),
                new_model_searches=0,
            )
        ),
    )
    return dict(regression_model_replays=replays, classification_metric_groups=len(m))
