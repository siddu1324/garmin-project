"""Self-contained presentation and aggregate-only executable teaching notebook."""

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import verify_bundle, write_json

LABELS = {
    "context": "Context",
    "original": "Original HR4",
    "hr": "Expanded HR",
    "activity": "HR + activity",
    "full": "Full single-window",
    "sparse": "Sparse logistic",
    "corr": "Correlation filter",
    "availability": "Availability filter",
    "no_flags": "No missingness flags",
    "missing_only": "Missingness only",
    "quality": "Quality only",
}
COLORS = ["#198a8b", "#d18a35", "#6057b6", "#65809d", "#9c576c"]


def pretty(name):
    if name == "majority":
        return "Majority dummy"
    text = name
    if text.startswith("complete_"):
        text = "Complete cohort · " + pretty(text.removeprefix("complete_"))
        return text
    suffix = " · balanced" if text.endswith("_w") else ""
    text = text.removesuffix("_u").removesuffix("_w")
    if text == "multiscale":
        return "Separate multiscale · 438"
    for m in [360, 120, 60]:
        if text.endswith(str(m)):
            return LABELS.get(text[: -len(str(m))], text) + f" · {m}m" + suffix
    return LABELS.get(text, text) + suffix


def table(frame, cols=None):
    f = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    if cols:
        f = f[cols]
    if "model" in f:
        f["model"] = f.model.map(pretty)
    if "baseline" in f:
        f["baseline"] = f.baseline.map(pretty)
    return (
        '<div class="table">'
        + f.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}", na_rep="—")
        + "</div>"
    )


def markdown(frame):
    f = frame.copy().map(lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))
    return (
        "| "
        + " | ".join(f.columns)
        + " |\n| "
        + " | ".join(["---"] * len(f.columns))
        + " |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in f.to_numpy())
    )


def render(root):
    root = Path(root)
    run = root / json.loads((root / "private/binary_6hour/current.json").read_text())["run"]
    out = root / "reports/binary_6hour"
    public = out / run.name
    if (out / "completion.json").exists():
        seal = json.loads((out / "completion.json").read_text())
        assert seal["run"] == run.name
        assert all(sha256(root / p) == h for p, h in seal["presentation_sha256"].items())
        verify_bundle(public)
        return out

    def load(name):
        return json.loads((public / f"{name}.json").read_text())

    metrics, diff = pd.DataFrame(load("metrics")), pd.DataFrame(load("paired_differences"))
    fold_stability = (
        pd.DataFrame(load("fold_metrics"))
        .groupby(["task", "model"])
        .auroc.agg(fold_auroc_mean="mean", fold_auroc_min="min", fold_auroc_max="max")
        .reset_index()
    )
    metrics = metrics.merge(fold_stability, on=["task", "model"], validate="one_to_one")
    write_json(public / "fold_stability.json", fold_stability.to_dict("records"))
    distributions = pd.DataFrame(load("distributions"))
    quality, additions = pd.DataFrame(load("quality")), pd.DataFrame(load("new_readings"))
    stability = pd.DataFrame(load("selection_stability"))
    support = pd.DataFrame(load("fold_support"))
    main_names = [
        "majority",
        "context_u",
        "original360_u",
        "hr360_u",
        "activity360_u",
        "full60_u",
        "full120_u",
        "full360_u",
        "sparse360_u",
    ]
    main = metrics[metrics.model.isin(main_names)].copy()
    main["order"] = main.model.map({n: i for i, n in enumerate(main_names)})
    main = main.sort_values(["task", "order"]).drop(columns="order")
    principal = diff[(diff.model == "full360_u") & diff.baseline.isin(["full60_u", "context_u"])]
    questions = []
    for task in ["A", "B"]:
        row = metrics[(metrics.task == task) & (metrics.model == "full360_u")].iloc[0]
        lines = []
        for baseline, label in [("full60_u", "60 minutes"), ("context_u", "context")]:
            d = principal[(principal.task == task) & principal.baseline.eq(baseline)].iloc[0]
            interpretation = (
                "a positive conditional difference"
                if d.auroc_low > 0
                else "no clear positive difference"
                if d.auroc_high >= 0
                else "a negative conditional difference"
            )
            lines.append(
                f"versus {label}: ΔAUROC {d.auroc_difference:+.3f} (95% interval {d.auroc_low:+.3f} to {d.auroc_high:+.3f}), {interpretation}"
            )
        questions.append(
            dict(
                task=task,
                finding=f"Binary {task}: full six-hour AUROC {row.auroc:.3f} ({row.auroc_low:.3f}–{row.auroc_high:.3f}), AP {row.ap:.3f}, balanced accuracy {row.balanced_accuracy:.3f}; "
                + "; ".join(lines)
                + ".",
            )
        )
    write_json(public / "principal_findings.json", questions)

    def metric(task, model):
        return metrics[(metrics.task == task) & metrics.model.eq(model)].iloc[0]

    def contrast(task, model, baseline):
        d = diff[(diff.task == task) & diff.model.eq(model) & diff.baseline.eq(baseline)].iloc[0]
        return f"ΔAUROC {d.auroc_difference:+.3f} (95% interval {d.auroc_low:+.3f} to {d.auroc_high:+.3f})"

    selection_answers, weighting_answers, missing_answers = [], [], []
    for task in ["A", "B"]:
        selection_answers.append(
            f"Binary {task}: "
            + "; ".join(
                f"{label} versus full: {contrast(task, name + '360_u', 'full360_u')}"
                for name, label in [
                    ("sparse", "sparse"),
                    ("corr", "correlation filter"),
                    ("availability", "availability filter"),
                ]
            )
        )
        a, b = metric(task, "full360_u"), metric(task, "full360_w")
        weighting_answers.append(
            f"Binary {task}: balanced sensitivity {b.sensitivity:.3f} versus {a.sensitivity:.3f} unweighted; specificity {b.specificity:.3f} versus {a.specificity:.3f}; precision {b.precision:.3f} versus {a.precision:.3f}. "
            + contrast(task, "full360_w", "full360_u")
        )
        missing_answers.append(
            f"Binary {task}: missingness-only AUROC {metric(task, 'missing_only360_u').auroc:.3f}; quality-only {metric(task, 'quality360_u').auroc:.3f}; context {metric(task, 'context_u').auroc:.3f}. Removing flags versus full: "
            + contrast(task, "no_flags360_u", "full360_u")
        )
    answers = [
        dict(
            question="Does six hours outperform 60 minutes?",
            answer=" ".join(q["finding"] for q in questions),
        ),
        dict(
            question="Which cutoff discriminates better?",
            answer=f"At the prespecified unweighted full six-hour model, Binary A AUROC is {metric('A', 'full360_u').auroc:.3f} and Binary B is {metric('B', 'full360_u').auroc:.3f}. These are different outcomes, not a clinical cutoff selection. Interpret uncertainty, class support and prevalence alongside the point difference; raw AP is not comparable across prevalence.",
        ),
        dict(
            question="Do expanded features add beyond context?",
            answer=" ".join(
                f"Binary {t}: {contrast(t, 'full360_u', 'context_u')}." for t in ["A", "B"]
            )
            + " The full model includes recording-missingness flags. A context increment alone does not isolate physiological-value signal; interpret it with no-flags and missingness-only controls, not just F1 or accuracy.",
        ),
        dict(
            question="Does feature selection help?",
            answer=". ".join(selection_answers)
            + ". These are multiple exploratory comparisons; do not select a winning feature set from the outer folds.",
        ),
        dict(
            question="Does weighting improve minority detection?",
            answer=". ".join(weighting_answers)
            + ". Sensitivity here is for class 1. Binary A's minority is class 0, whose recall is specificity; Binary B's minority is class 1. A changed operating point does not necessarily improve AUROC.",
        ),
        dict(
            question="Are recording gaps and missingness affecting performance?",
            answer=". ".join(missing_answers)
            + ". These controls assess associations with the recording process and handling choices; they cannot establish that gaps caused weak prediction. See per-class missingness and complete-case composition.",
        ),
        dict(
            question="What should change next?",
            answer="Retain strict time-support allowlists, explicit recording indicators and matched-cohort evaluation. Resolve timestamp/batching, sensor quality and prompt/recall evidence. Prespecify confirmation of any promising effect; do not expand the search or open the final test automatically.",
        ),
    ]
    write_json(public / "research_answers.json", answers)
    figures = public / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    names = []

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        names.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for task, ax in zip(["A", "B"], axes):
        d = distributions[distributions.task == task].sort_values("class_id")
        bars = ax.bar([0, 1], d.responses, color=COLORS[:2])
        ax.bar_label(
            bars, labels=[f"{r.responses:,}\n{r.fraction:.1%}" for r in d.itertuples()], padding=4
        )
        ax.set(
            xticks=[0, 1],
            xticklabels=["3–4", "5–15"] if task == "A" else ["3–9", "10–15"],
            title=f"Binary {task} · composite score",
            ylabel="Development responses",
            ylim=(0, 3700),
        )
    fig.tight_layout()
    save(fig, "01_classes")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for task, ax in zip(["A", "B"], axes):
        for j, m in enumerate([60, 120, 360]):
            d = metrics.set_index(["task", "model"]).loc[
                [(task, f"{b}{m}_u") for b in ["original", "hr", "activity", "full", "sparse"]]
            ]
            ax.plot(
                np.arange(5) + (j - 1) * 0.07, d.auroc, "o-", color=COLORS[j], label=f"{m} minutes"
            )
        ctx = metrics[(metrics.task == task) & metrics.model.eq("context_u")].iloc[0]
        ax.axhline(ctx.auroc, color="#43576b", ls="--", label="Context")
        ax.axhline(0.5, color="grey", ls=":", label="Chance / majority")
        ax.set(
            xticks=range(5),
            xticklabels=["Original HR4", "Expanded HR", "HR + activity", "Full", "Sparse"],
            ylabel="Pooled out-of-fold AUROC",
            title=f"Binary {task}",
            ylim=(
                min(0.35, float(metrics.auroc.min()) - 0.03),
                max(0.65, float(metrics.auroc.max()) + 0.03),
            ),
        )
        ax.tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "02_windows")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, r in enumerate(principal.itertuples()):
        ax.errorbar(
            r.auroc_difference,
            i,
            xerr=[[r.auroc_difference - r.auroc_low], [r.auroc_high - r.auroc_difference]],
            fmt="o",
            color=COLORS[0 if r.task == "A" else 1],
            capsize=4,
        )
    ax.axvline(0, color="grey", ls="--")
    ax.set(
        yticks=range(len(principal)),
        yticklabels=[
            f"Binary {r.task}: 360m − {pretty(r.baseline)}" for r in principal.itertuples()
        ],
        xlabel="AUROC difference · positive favors six hours",
    )
    fig.tight_layout()
    save(fig, "03_paired")
    curves = load("curves")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, task in enumerate(["A", "B"]):
        prevalence = distributions[
            (distributions.task == task) & distributions.class_id.eq(1)
        ].fraction.iloc[0]
        for j, name in enumerate(["context_u", "full60_u", "full120_u", "full360_u"]):
            c = next(c for c in curves if c["task"] == task and c["model"] == name)
            axes[i, 0].plot(c["grid"], c["roc_tpr"], label=pretty(name), color=COLORS[j])
            axes[i, 1].plot(c["grid"], c["pr_precision"], label=pretty(name), color=COLORS[j])
        axes[i, 0].plot([0, 1], [0, 1], color="grey", ls=":")
        axes[i, 1].axhline(prevalence, color="grey", ls=":", label=f"Prevalence {prevalence:.3f}")
        for j in [0, 1]:
            axes[i, j].set(
                xlim=(0, 1),
                ylim=(0, 1),
                title=f"Binary {task} · {'ROC' if j == 0 else 'precision–recall'}",
                xlabel="False-positive rate" if j == 0 else "Recall",
                ylabel="Sensitivity" if j == 0 else "Precision",
            )
            axes[i, j].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "04_curves")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.7))
    for i, task in enumerate(["A", "B"]):
        for j, name in enumerate(["context_u", "full360_u", "full360_w"]):
            r = metrics[(metrics.task == task) & metrics.model.eq(name)].iloc[0]
            cm = np.array([[r.tn, r.fp], [r.fn, r.tp]], int)
            ax = axes[i, j]
            ax.imshow(cm / cm.sum(axis=1)[:, None], cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(
                        b,
                        a,
                        f"{cm[a, b]:,}\n{cm[a, b] / cm[a].sum():.1%}",
                        ha="center",
                        va="center",
                        color="white" if cm[a, b] / cm[a].sum() > 0.55 else "black",
                    )
            ax.set(
                xticks=[0, 1],
                yticks=[0, 1],
                xlabel="Predicted class",
                ylabel=f"Binary {task} · actual class",
                title=pretty(name),
            )
    fig.tight_layout()
    save(fig, "05_confusion")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    controls = [
        "context_u",
        "full360_u",
        "full360_w",
        "sparse360_u",
        "corr360_u",
        "availability360_u",
        "no_flags360_u",
        "missing_only360_u",
        "quality360_u",
        "multiscale_u",
    ]
    answer_html = "".join(
        f"<h3>{html.escape(a['question'])}</h3><p>{html.escape(a['answer'])}</p>" for a in answers
    )
    for task, ax in zip(["A", "B"], axes):
        d = metrics.set_index(["task", "model"]).loc[[(task, n) for n in controls]]
        ax.errorbar(
            d.auroc,
            range(len(d)),
            xerr=[d.auroc - d.auroc_low, d.auroc_high - d.auroc],
            fmt="o",
            capsize=3,
            color=COLORS[0],
        )
        ax.axvline(0.5, color="grey", ls=":")
        ax.set(
            yticks=range(len(d)),
            yticklabels=[pretty(n) for n in controls],
            xlabel="AUROC with participant interval",
            title=f"Binary {task}",
        )
    fig.tight_layout()
    save(fig, "06_controls")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, quality_metric, title, factor in zip(
        axes,
        ["accepted_readings", "occupied_fraction", "maximum_gap_seconds"],
        ["Accepted HR readings", "Occupied minute bins (%)", "Largest gap incl. edges (minutes)"],
        [1, 100, 1 / 60],
    ):
        d = quality[
            (quality.task == "A")
            & quality.survey.eq("All")
            & quality.class_id.eq("All")
            & quality.metric.eq(quality_metric)
        ].sort_values("window")
        ax.errorbar(
            range(3),
            d["median"] * factor,
            yerr=[(d["median"] - d.p25) * factor, (d.p75 - d["median"]) * factor],
            fmt="o",
            capsize=6,
            color=COLORS[0],
        )
        ax.set(
            xticks=range(3),
            xticklabels=["60m", "120m", "360m"],
            title=title,
            ylabel="Median and interquartile range",
        )
    fig.tight_layout()
    save(fig, "07_support")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    top = []
    for task, ax in zip(["A", "B"], axes):
        d = (
            stability[(stability.task == task) & stability.model.eq("full360_u")]
            .nlargest(8, "mean_absolute_coefficient")
            .sort_values("mean_absolute_coefficient")
        )
        top.append(d)
        labels = [
            r.feature.replace("__360m", "").replace("n_survey_category_", "survey:")
            + f" ({r.selected_folds}/5)"
            for r in d.itertuples()
        ]
        ax.barh(range(len(d)), d.mean_absolute_coefficient, color=COLORS[0])
        ax.set(
            yticks=range(len(d)),
            yticklabels=labels,
            xlabel="Mean |training coefficient|",
            title=f"Binary {task} · six-hour full model",
        )
    fig.tight_layout()
    save(fig, "08_coefficients")
    for task in ["A", "B"]:
        selected = stability[(stability.task == task) & stability.model.eq("sparse360_u")]
        top.append(
            selected[selected.mean_absolute_coefficient > 1e-8].nlargest(
                8, "mean_absolute_coefficient"
            )
        )

    def image(name, caption):
        blob = base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
        return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{blob}"><figcaption>{caption}</figcaption></figure>'

    def detail(title, body):
        return f"<details><summary>{title}</summary>{body}</details>"

    headline = " ".join(q["finding"] for q in questions)
    metrics_cols = [
        "task",
        "model",
        "responses",
        "participants",
        "auroc",
        "auroc_low",
        "auroc_high",
        "fold_auroc_mean",
        "ap",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
    ]
    q360 = quality[
        (quality.task == "A")
        & quality.window.eq(360)
        & quality.survey.eq("All")
        & quality.class_id.eq("All")
    ]
    add360 = additions[additions.window.eq(360) & additions.survey.eq("All")].iloc[0]
    cohort_support = support[support.model.isin(["context_u", "complete_context_u"])].sort_values(
        ["task", "cohort", "fold", "role"]
    )
    inventory = load("feature_inventory")
    sections = [
        (
            "Two cutoffs, two distinct questions",
            "<p>The three-item composite is unchanged: nervous + (6 − cope) + (6 − control), requiring all three valid 1–5 items. Binary A identifies scores ≥5; Binary B identifies scores ≥10. Neither defines a clinical diagnosis. All 3,706 EMAs remain accounted for; 3,689 valid-score responses from 125 development participants enter every primary comparator.</p>"
            + image(
                names[0],
                "Participant counts by class can overlap because a participant can contribute both classes.",
            )
            + table(distributions)
            + detail("Exact support in every training and validation fold", table(cohort_support)),
        ),
        (
            "A controlled comparison of 60, 120 and 360 minutes",
            "<p>Primary: six hours versus 60 minutes; 120 minutes is secondary. Physiological models all include the same time/survey context. Main training and validation memberships are identical across windows and features. The 47 full single-window candidates exclude earlier-history and adjacent-window columns, even when their names contain that lookback.</p><div class='callout'><b>Synthetic timeline:</b> minute-resolution actual start 09:22 → cutoff 09:21. The 60-minute window is [08:21,09:21); the six-hour window is [03:21,09:21). Empty minutes stay unobserved. Example item values 4,2,3 yield composite 11: positive in both tasks.</div>"
            + table(
                pd.DataFrame(
                    [
                        dict(
                            block=x["block"],
                            candidates=x["physiological_candidates"],
                            role=x["role"],
                        )
                        for x in inventory
                    ]
                )
            )
            + detail(
                "Exact predictor columns",
                table(
                    pd.DataFrame(
                        [dict(block=x["block"], feature=c) for x in inventory for c in x["columns"]]
                    )
                ),
            ),
        ),
        (
            "Nested logistic regression, optimized for ranking",
            "<p>Five unchanged outer participant folds, three grouped inner folds. Tune C={.001,.01,.1,1} and L2/L1/Elastic Net using mean inner AUROC. Refit imputation, scaling, category encoding and selection inside every training split. Balanced and unweighted training are separate experiments. Hard predictions use the fixed 0.5 probability threshold.</p><p>Training medians fill feature summaries only. Variable missingness flags are recording-process covariates. Separate controls test correlation filtering, ≥50% training availability filtering, sparse regularization, no flags, missingness-only and quality-only inputs. The 438-feature multiscale sensitivity is not a six-hour model.</p>"
            + detail("Chosen settings and convergence", table(load("settings")))
            + detail(
                "All prespecified trials, evaluated only inside training folds",
                table(load("trials")),
            ),
        ),
        (
            "Does six hours add discrimination?",
            "<div class='callout'>"
            + html.escape(headline)
            + "</div>"
            + image(
                names[1],
                "Point comparisons share the same held-out responses. No window is selected as optimal.",
            )
            + image(
                names[2],
                "Paired participant bootstrap: fixed OOF predictions, no retuning uncertainty or multiple-comparison correction.",
            )
            + table(main, metrics_cols)
            + "<p>Pooled AUROC compares probabilities from different fold models; differences in fold calibration can affect it. The mean fold AUROC is shown alongside it as a stability diagnostic, not a replacement endpoint selected after seeing results. For Binary A, context has mean fold AUROC 0.523 versus 0.498 for six-hour full features, despite the opposite pooled point ordering. Neither summary establishes a physiological gain.</p>"
            + detail(
                "Every model and paired comparison", table(metrics, metrics_cols) + table(diff)
            ),
        ),
        (
            "Ranking and minority-class detection are different",
            "<p>ROC uses continuous positive-class probability. PR curves must be interpreted beside prevalence; AP is noninterpolated average precision. Binary A's positive class includes both middle and high scores, so its sensitivity is not high-score-only recall. Weighting may change the sensitivity/specificity trade-off without improving ranking.</p>"
            + image(
                names[3],
                "Curves use a common plotting grid; numerical AUROC/AP are calculated from the original saved probabilities.",
            )
            + image(
                names[4],
                "Counts and row percentages, fixed threshold 0.5. Rows are actual classes.",
            )
            + detail("Errors by survey category", table(load("survey_metrics")))
            + detail("Both classes: support, precision, recall and F1", table(load("per_class")))
            + "<p>Precision/F1 with zero predicted support are reported as zero by convention. Undefined one-class ranking metrics remain missing. No decision threshold is tuned.</p>"
            + detail(
                "Fold performance and training-fit discrimination",
                table(load("fold_metrics")) + table(load("training_metrics")),
            ),
        ),
        (
            "Separate physiology, weighting and recording patterns",
            image(
                names[5],
                "All rows use the same 3,689 responses. Availability/quality controls do not establish a physiological mechanism.",
            )
            + detail(
                "Selected features and coefficient stability",
                table(
                    pd.concat(top),
                    [
                        "task",
                        "model",
                        "feature",
                        "selected_folds",
                        "mean_absolute_coefficient",
                        "min_coefficient",
                        "max_coefficient",
                    ],
                ),
            )
            + image(
                names[7],
                "Largest training coefficient magnitudes, not proven useful features. Parentheses show nonzero selection in five folds. Numeric inputs are standardized; missingness/category flags are not. Correlation and coding limit direct magnitude comparisons.",
            ),
        ),
        (
            "Six hours is not six hours of continuous monitoring",
            f"<p>Nested source-lineage checks confirm that {int(add360.new_readings):,} of {int(add360.responses):,} modeled responses gain accepted HR readings between 60 minutes and six hours; {int(add360.no_new_readings):,} gain none. The median number added is {add360['median']:.0f}. This is additional accepted logical records, not additional minutes of continuous monitoring.</p>"
            + "<p>Different EMAs can reuse the same sensor records in overlapping windows. These response counts are not independent recording segments, and we do not multiply window counts by six hours to estimate monitoring time.</p>"
            + image(
                names[6],
                "Occupied bins contain at least one accepted reading. Span is first-to-last reading, not continuous support. Gaps include window edges.",
            )
            + table(
                q360,
                [
                    "metric",
                    "responses",
                    "observed",
                    "zero_responses",
                    "median",
                    "p25",
                    "p75",
                    "p90",
                ],
            )
            + detail("Per-feature missingness by target class", table(load("missingness")))
            + detail(
                "Survey/class-specific support and new readings", table(quality) + table(additions)
            ),
        ),
        (
            "Check who complete-case filtering excludes",
            "<p>The common complete-case sensitivity requires the original four HR summaries at all three windows. All its comparators are refitted on identical eligible training and validation responses; other full-block features retain training-only imputation. Native full-feature completeness is reported for coverage, not used to silently discard the primary cohort.</p>"
            + table(load("composition"))
            + detail("Native coverage by feature block", table(load("coverage")))
            + detail(
                "Common-cohort refitted results",
                table(metrics[metrics.cohort.eq("complete")], metrics_cols),
            )
            + detail(
                "All-valid fits evaluated on common complete cases (different training population)",
                table(load("restricted_evaluation")),
            ),
        ),
        (
            "What this resolves—and what to change next",
            "<p>These are newly fitted binary, AUROC-tuned logistic experiments. Notebook 12 emphasized macro F1 and different cohorts; Notebook 13's expanded classification and Notebook 14's regression are prior exploratory evidence, not independent confirmation. Their headline metrics are not directly ranked against these results.</p><p>Use the prespecified paired full360-versus-full60 and full360-versus-context intervals to assess the primary question. An isolated F1 increase, a large coefficient, or a point maximum does not establish useful physiology. Cutoff comparisons involve different outcomes and prevalence; neither is a validated diagnosis.</p><p>Acquisition-versus-receipt timestamps, batching, sensor quality and prompt/recall alignment remain unresolved. Sparse/stale observations and missingness can influence predictions, but these analyses cannot identify their causal contribution. More within-person outcome variation does not guarantee wearable predictability.</p><div class='callout'><b>Next:</b> preserve a strict single-window feature allowlist; keep recording availability visibly separate; obtain synchronized timing, prompt and quality evidence before automatically expanding features or model searches. Any promising paired effect needs a separately specified confirmation. Keep the final test sealed.</div><p>Reproduction: config/binary_6hour.json; src/garmin_binary_6hour; specification.md; versioned private jobs and public aggregates. See verification.md for test, replay, privacy and integrity checks. No commit, push, provider message or raw reprocessing.</p>",
        ),
    ]
    style = """body{margin:0;background:#edf3f4;color:#203d49;font:17px/1.65 system-ui,sans-serif}header,main{max-width:1120px;margin:auto}header{padding:65px 25px 30px}header h1{font:600 clamp(38px,5vw,65px)/1.1 Georgia,serif;margin:12px 0}h2{font:600 32px/1.2 Georgia,serif}nav{display:flex;flex-wrap:wrap;gap:8px;margin:25px 0}nav a{padding:7px 12px;color:#12686d;border:1px solid #b8ced2;border-radius:20px;text-decoration:none;font-size:14px}section{background:white;border:1px solid #d2e0e4;border-radius:18px;padding:38px;margin:24px 20px}.eyebrow{color:#008489;letter-spacing:.17em;font-size:12px;font-weight:750;text-transform:uppercase}.callout{border-left:4px solid #008489;background:#eaf5f3;padding:18px 22px;margin:24px 0}figure{margin:26px 0}img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#56717d}.table{overflow:auto;max-height:620px;border:1px solid #d4e2e5;border-radius:8px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:12px}th{background:#e9f1f4;position:sticky;top:0;z-index:1;text-align:left}td,th{padding:9px 12px;border-bottom:1px solid #d4e2e5;white-space:nowrap}details{border:1px solid #d4e2e5;border-radius:8px;padding:14px;margin-top:16px}summary{cursor:pointer;font-weight:650}.sub{font-size:21px;max-width:900px}.tag{display:inline-block;background:#d9e9e9;border-radius:6px;padding:5px 10px;font-size:13px}footer{max-width:1080px;margin:30px auto;padding:20px;color:#56717d;font-size:13px}@media(max-width:600px){section{padding:22px;margin:15px 12px}header{padding:35px 22px}h2{font-size:28px}.sub{font-size:18px}}@media print{body{background:white}section{break-inside:avoid}nav{display:none}.table{max-height:none;overflow:visible}details{break-inside:avoid}}"""
    sections[-1] = (sections[-1][0], answer_html + sections[-1][1])
    nav = "".join(
        f'<a href="#s{i}">{i:02d} · {label}</a>'
        for i, label in enumerate(
            [
                "Targets",
                "Windows",
                "Methods",
                "Results",
                "Detection",
                "Controls",
                "Recording",
                "Selection",
                "Next",
            ],
            1,
        )
    )
    content = "".join(
        f'<section id="s{i}"><div class="eyebrow">{i:02d} / Methods & findings</div><h2>{title}</h2>{body}</section>'
        for i, (title, body) in enumerate(sections, 1)
    )
    document = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Six hours of Garmin data · Binary PSS experiment</title><style>{style}</style></head><body><header><div class="eyebrow">Garmin + EMA · Notebook 15 · Development only</div><h1>Does a longer physiological window help?</h1><p class="sub">A controlled binary logistic comparison of 60 minutes, two hours and six hours—judged by discrimination beyond time and survey context.</p><span class="tag">3,689 responses · 125 participants · frozen participant folds</span><nav>{nav}</nav></header><main>{content}</main><footer>Exploratory three-item composite thresholds; no clinical or real-time validation. Final test sealed. Run {run.name}.</footer></body></html>'
    lead = (
        "Six hours did not establish better discrimination than one hour or context."
        if ((principal.auroc_low <= 0) & (principal.auroc_high >= 0)).all()
        else "The paired comparisons show task-dependent differences that require cautious interpretation."
    )
    document = document.replace(
        "</header>",
        f"<div class='callout'><b>{lead}</b> Full six-hour AUROC: {metric('A', 'full360_u').auroc:.3f} for Binary A; {metric('B', 'full360_u').auroc:.3f} for Binary B. See paired intervals below.</div></header>",
    )
    (out / "index.html").write_text(document)
    concise = main[
        main.model.isin(["majority", "context_u", "full60_u", "full120_u", "full360_u"])
    ][
        [
            "task",
            "model",
            "auroc",
            "fold_auroc_mean",
            "ap",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
        ]
    ].copy()
    concise["model"] = concise.model.map(pretty)
    summary = (
        "# Binary PSS: six-hour logistic experiment for Will\n\n"
        + "**"
        + lead
        + "**\n\n"
        + "\n\n".join(q["finding"] for q in questions)
        + "\n\n"
        + markdown(concise)
        + "\n\n"
    )
    summary += "All main comparisons use identical 3,689 responses from 125 development participants; all 3,706 EMAs remain accounted for. Binary A is ≥5; Binary B is ≥10. Thresholds are exploratory, not clinical. Main six-hour models use 47 strictly within-window candidates plus context, excluding historical and adjacent-window columns. The complete 438-feature multiscale experiment is separate.\n\n"
    summary += "Pooled AUROC compares probabilities across different fold models and can reflect fold calibration differences. Mean fold AUROC is a stability diagnostic: Binary A context is 0.523 versus 0.498 for six-hour full features, despite their opposite pooled point ordering. It does not replace the specified pooled endpoint. Neither supports a reliable physiological gain.\n\n"
    summary += "\n\n".join(f"**{a['question']}** {a['answer']}" for a in answers[1:]) + "\n\n"
    summary += "Five frozen participant folds and three grouped inner folds tune L2/L1/Elastic Net by AUROC. Preprocessing and feature filtering are training-only. Balanced training, sparse selection, correlation/availability filters, no flags, missingness-only and quality-only controls are separately evaluated. AP is prevalence-sensitive; fixed-threshold recall is not interchangeable with ranking performance.\n\n"
    summary += f"Six hours adds accepted readings for {int(add360.new_readings):,}/{int(add360.responses):,} responses; {int(add360.no_new_readings):,} gain none. The median additional count is {add360['median']:.0f}. This does not establish continuous six-hour coverage. Complete-case sensitivity refits matched cohorts; native coverage and class-specific missingness are reported separately.\n\n"
    summary += "Interpret the paired intervals, not isolated point maxima or F1 gains. Intervals resample participants conditional on saved predictions and do not include retuning uncertainty or multiplicity correction. This follows earlier development exploration; no final-test evidence is used. Coefficients are training associations, not causal or validated feature importance.\n\nNext: retain strict single-window feature definitions and separate recording-process controls. Resolve exporter timing/batching, sensor quality and actual prompt/recall boundaries before further automatic searches. A promising difference requires a separately specified confirmation. See the standalone HTML for all results and verification.md for checks.\n"
    (out / "summary_for_will.md").write_text(summary)
    nb = nbformat.v4.new_notebook(
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            garmin_public_aggregate_notebook=True,
        )
    )
    cells = [
        nbformat.v4.new_markdown_cell(
            "# 15 · Binary PSS: does six hours help?\n\nDevelopment-only logistic regression. All tables and figures are aggregates or synthetic examples. The final test remains sealed. This notebook reads completed, immutable experiment aggregates; it does not rerun raw extraction or silently refit models."
        )
    ]
    cells.append(
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'config/binary_6hour.json').exists())\nreport = root / 'reports/binary_6hour'\nversion = '"
            + run.name
            + "'\npublic = report / version\ndef load(name): return json.loads((public / (name + '.json')).read_text())\ndef table(name, columns=None):\n    frame = pd.DataFrame(load(name))\n    display(frame if columns is None else frame[columns])\ndef figure(name): display(Image(filename=str(public / 'figures' / (name + '.png'))))"
        )
    )
    explanations = [
        (
            "Targets and exact class support",
            "Binary A: 3–4 versus 5–15. Binary B: 3–9 versus 10–15. Participant class counts overlap. Require all three valid composite items; no target imputation.",
            "figure('01_classes')\ntable('distributions')\nsupport = pd.DataFrame(load('fold_support'))\ndisplay(support[support.model.isin(['context_u','complete_context_u'])])",
        ),
        (
            "Single-window inputs and a synthetic timeline",
            "Minute start 09:22 → cutoff 09:21. Six hours is [03:21,09:21); one hour is [08:21,09:21). Earlier-reference and adjacent-window features extend outside these intervals and are excluded from the primary experiment. Synthetic items 4,2,3 give 11: positive in both tasks.",
            "inventory = pd.DataFrame(load('feature_inventory'))\ndisplay(inventory[['block','physiological_candidates','role']])",
        ),
        (
            "Train-only nested optimization",
            "Five original participant folds, three grouped inner folds. Tune C and L1/L2/Elastic Net by AUROC. Balanced/unweighted are separate; threshold stays 0.5. Median imputation fills features, never raw readings or outcomes. All transforms and selectors are fitted inside training splits.",
            "settings = pd.DataFrame(load('settings'))\ndisplay(settings[settings.model.isin(['context_u','full60_u','full360_u','full360_w'])])",
        ),
        (
            "Matched primary comparisons",
            "Use paired intervals to evaluate the six-hour increment over 60 minutes and context. Conditional intervals omit retuning uncertainty and multiplicity correction.",
            "for item in load('principal_findings'): display(Markdown(item['finding']))\nfigure('02_windows')\nfigure('03_paired')\nmetrics = pd.DataFrame(load('metrics'))\ndisplay(metrics[metrics.model.isin(['majority','context_u','full60_u','full120_u','full360_u'])])",
        ),
        (
            "Ranking versus fixed-threshold detection",
            "AP is noninterpolated average precision and depends on prevalence. AUROC uses continuous probabilities. Binary A positive recall does not isolate the high class. Confusion matrices display counts and row percentages.",
            "figure('04_curves')\nfigure('05_confusion')",
        ),
        (
            "Regularization, weighting and missingness",
            "Missingness/quality controls test recording patterns, not physiology. Coefficients are training associations; magnitude does not prove usefulness. Correlated predictors can substitute.",
            "figure('06_controls')\nfigure('08_coefficients')\nd = pd.DataFrame(load('paired_differences'))\ndisplay(d[(d.task == 'B') & d.model.isin(['full360_w','sparse360_u','availability360_u','no_flags360_u'])])",
        ),
        (
            "What six-hour recording support means",
            "Occupied minute bins require at least one accepted reading, not complete recording. First-to-last span does not fill internal gaps. Nested source slices justify accepted-count differences without new extraction.",
            "figure('07_support')\ntable('new_readings')\ntable('lineage_checks')\nq = pd.DataFrame(load('quality'))\ndisplay(q[(q.task == 'A') & (q.window == 360) & (q.survey == 'All') & (q.class_id == 'All')])",
        ),
        (
            "Selection and incomplete coverage",
            "Report who complete-case filtering excludes. Common-cohort refits and restricted evaluation of larger-training-cohort fits answer different questions. Do not compare unequal-cohort headline metrics.",
            "table('composition')\ntable('coverage')\ntable('restricted_evaluation')",
        ),
        (
            "Conclusions, limitations and verification",
            "These are new AUROC-tuned binary experiments, not a relabeling of older macro-F1 results. Preserve the final test and resolve timing/recall/quality uncertainty before expanding searches.",
            "display(Markdown((report / 'summary_for_will.md').read_text()))\nif (report / 'verification.md').exists(): display(Markdown((report / 'verification.md').read_text()))",
        ),
    ]
    for i, (title, prose, code) in enumerate(explanations, 1):
        cells += [
            nbformat.v4.new_markdown_cell(f"## {i:02d} · {title}\n\n{prose}"),
            nbformat.v4.new_code_cell(code),
        ]
    nb.cells = cells
    nbformat.write(nb, root / "notebooks/15_binary_pss_6hour_experiments.ipynb")
    return out


if __name__ == "__main__":
    print(render(Path.cwd()))
