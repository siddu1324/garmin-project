"""Aggregate-only notebook and standalone presentation driven by actual results."""

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
from garmin_comprehensive_regression.evaluate import finite
from garmin_preprocessing.common import verify_bundle, write_json

COLORS = ["#167f86", "#ce873c", "#6960a9", "#8c5267", "#607c99"]


def table(data, columns=None):
    f = pd.DataFrame(data).copy()
    if columns:
        f = f[columns]
    for c in f:
        if pd.api.types.is_float_dtype(f[c]):
            f[c] = f[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    return '<div class="table">' + f.to_html(index=False, escape=True, border=0) + "</div>"


def render(root):
    root = Path(root)
    run = root / json.loads((root / "private/binary_improvement/current.json").read_text())["run"]
    report = root / "reports/binary_improvement"
    public = report / run.name
    if (report / "completion.json").exists():
        seal = json.loads((report / "completion.json").read_text())
        assert all(sha256(root / p) == h for p, h in seal["presentation_sha256"].items())
        verify_bundle(public)
        return report

    def load(name):
        return json.loads((public / f"{name}.json").read_text())

    metrics = pd.DataFrame(load("metrics"))
    paired = pd.DataFrame(load("paired"))
    thresholds = pd.DataFrame(load("thresholds"))
    quality = pd.DataFrame(json.loads((report / "audit/quality.json").read_text()))
    classes = pd.DataFrame(json.loads((report / "audit/classes.json").read_text()))
    figs = public / "figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.facecolor": "white",
        }
    )
    figures = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figs / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        figures.append(name)

    def row(task, model):
        return metrics[metrics.task.eq(task) & metrics.model.eq(model)].iloc[0]

    def comparison(task, model, reference):
        return paired[
            paired.task.eq(task) & paired.model.eq(model) & paired.reference.eq(reference)
        ].iloc[0]

    def difference(task, model, reference):
        r = comparison(task, model, reference)
        return f"ΔAUROC {r.auroc:+.3f} (95% interval {r.auroc_low:+.3f} to {r.auroc_high:+.3f})"

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, t in zip(axes, ["A", "B"]):
        f = classes[classes.task.eq(t)]
        ax.bar(["Class 0", "Class 1"], f.responses, color=COLORS[:2])
        for i, r in enumerate(f.itertuples()):
            ax.text(
                i, r.responses + 50, f"{r.responses:,}\n{r.participants} participants", ha="center"
            )
        ax.set(
            title=f"Binary {t}: positive ≥{5 if t == 'A' else 10}",
            ylabel="Development responses",
            ylim=(0, 3900),
        )
    save(fig, "01_classes")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, t in zip(axes, ["A", "B"]):
        for color, b, label in zip(
            COLORS,
            ["original", "hr", "activity", "full", "sparse"],
            ["HR4", "Expanded HR", "HR + activity", "Full47", "Sparse47"],
        ):
            ax.plot(
                [1, 2, 6, 12, 24],
                [row(t, f"{b}{m}").auroc for m in [60, 120, 360, 720, 1440]],
                "o-",
                label=label,
                color=color,
            )
        ax.axhline(row(t, "context").auroc, ls="--", color="#333", label="Context")
        ax.axhline(0.5, ls=":", color="#999")
        ax.set(
            title=f"Binary {t}",
            xlabel="Lookback hours",
            ylabel="Pooled OOF AUROC",
            ylim=(0.35, 0.65),
            xticks=[1, 2, 6, 12, 24],
        )
        ax.legend(fontsize=8, ncol=2)
    save(fig, "02_windows")
    primary = [
        ("full720", "full360"),
        ("full1440", "full360"),
        ("compact_smote", "compact_none"),
        ("context_compact_conditional", "context_compact_none"),
        ("minute360", "original360"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    labels = [
        "12h − 6h full",
        "24h − 6h full",
        "SMOTE − unbalanced (numeric)",
        "Conditional − unbalanced (context)",
        "Minute − reading HR4 (6h)",
    ]
    for ax, t in zip(axes, ["A", "B"]):
        for i, (a, b) in enumerate(primary):
            r = comparison(t, a, b)
            ax.errorbar(
                r.auroc,
                i,
                xerr=[[r.auroc - r.auroc_low], [r.auroc_high - r.auroc]],
                fmt="o",
                color=COLORS[0],
                capsize=3,
            )
        ax.axvline(0, color="#999", ls="--")
        ax.set(
            yticks=range(5),
            yticklabels=labels,
            title=f"Binary {t}",
            xlabel="Paired ΔAUROC · 95% conditional interval",
        )
        ax.invert_yaxis()
    save(fig, "03_primary")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    arms = ["none", "weighted", "over", "under", "smote"]
    for i, t in enumerate(["A", "B"]):
        for j, key in enumerate(["auroc", "balanced_accuracy"]):
            ax = axes[i, j]
            for prefix, methods, color, label in [
                ("", arms, COLORS[0], "Numeric compact"),
                ("context_", arms[:-1] + ["conditional"], COLORS[1], "Context + compact"),
            ]:
                vals = [row(t, f"{prefix}compact_{a}")[key] for a in methods]
                ax.plot(range(5), vals, "o-", color=color, label=label)
            ax.axhline(0.5, ls=":", color="#999")
            ax.set(
                title=f"Binary {t}: {key.replace('_', ' ')}",
                ylim=(0.3, 0.7),
                xticks=range(5),
                xticklabels=[
                    "Unbalanced",
                    "Weighted",
                    "Random over",
                    "Random under",
                    "SMOTE /\nconditional",
                ],
            )
            ax.tick_params(axis="x", labelsize=8)
            ax.legend(fontsize=8)
    save(fig, "04_rebalance")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for i, t in enumerate(["A", "B"]):
        for j, (model, mode, label) in enumerate(
            [
                ("context", "default", "Context · 0.5"),
                ("compact_smote", "default", "Numeric SMOTE · 0.5"),
                ("compact_smote", "inner_BA_threshold", "SMOTE · inner threshold"),
            ]
        ):
            r = thresholds[
                thresholds.task.eq(t) & thresholds.model.eq(model) & thresholds["mode"].eq(mode)
            ].iloc[0]
            cm = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=float)
            fractions = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[i, j]
            ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(
                        b,
                        a,
                        f"{cm[a, b]:,.0f}\n{fractions[a, b]:.1%}",
                        ha="center",
                        va="center",
                        color="white" if fractions[a, b] > 0.6 else "#203e48",
                    )
            ax.set(
                title=label,
                xlabel="Predicted class",
                ylabel=f"Binary {t}: actual class",
                xticks=[0, 1],
                yticks=[0, 1],
            )
    save(fig, "05_confusion")
    curves = {(r["task"], r["model"]): r for r in load("curves")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, t in enumerate(["A", "B"]):
        for color, model, label in zip(
            COLORS,
            ["context", "full360", "full1440", "compact_none", "compact_smote"],
            ["Context", "Full 6h", "Full 24h", "Compact", "SMOTE"],
        ):
            r = curves[(t, model)]
            axes[i, 0].plot(r["grid"], r["tpr"], label=label, color=color)
            axes[i, 1].plot(r["grid"], r["precision"], label=label, color=color)
        axes[i, 0].plot([0, 1], [0, 1], ":", color="#999")
        axes[i, 1].axhline(
            row(t, "context").prevalence,
            ls=":",
            color="#999",
            label=f"Prevalence {row(t, 'context').prevalence:.3f}",
        )
        for ax in axes[i]:
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.legend(fontsize=8)
        axes[i, 0].set(title=f"Binary {t}: ROC", xlabel="False positive rate", ylabel="Sensitivity")
        axes[i, 1].set(title=f"Binary {t}: precision–recall", xlabel="Recall", ylabel="Precision")
    save(fig, "06_curves")
    cal = pd.DataFrame(load("calibration"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, t in zip(axes, ["A", "B"]):
        for model, mode, color, label in [
            ("context", "raw", COLORS[0], "Context"),
            ("compact_none", "raw", COLORS[1], "Numeric unbalanced"),
            ("compact_smote", "raw", COLORS[2], "SMOTE raw"),
            ("compact_smote", "prior_corrected", COLORS[3], "SMOTE prior correction"),
        ]:
            c = cal[
                cal.task.eq(t) & cal.model.eq(model) & cal["mode"].eq(mode) & cal["count"].ge(5)
            ]
            ax.plot(c.mean_probability, c.observed_fraction, "o-", color=color, label=label)
        ax.plot([0, 1], [0, 1], ":", color="#999")
        ax.set(
            title=f"Binary {t}: calibration",
            xlabel="Mean predicted probability",
            ylabel="Observed class1 fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=8)
    save(fig, "07_calibration")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    q = quality[quality.task.eq("A") & quality.survey.eq("All")]
    for ax, key, label, factor in [
        (axes[0], "occupied_minutes", "Median occupied minutes", 1),
        (axes[1], "maximum_gap_seconds", "Median maximum gap (minutes)", 1 / 60),
    ]:
        z = q[q.metric.eq(key)].sort_values("window")
        ax.bar(range(5), z["median"] * factor, color=COLORS[0])
        ax.set(xticks=range(5), xticklabels=["1h", "2h", "6h", "12h", "24h"], ylabel=label)
        for i, r in enumerate(z.itertuples()):
            ax.text(i, r.median * factor, f"{r.median * factor:.0f}", ha="center", va="bottom")
    save(fig, "08_support")
    choices = [
        "original360",
        "minute360",
        "temporal360",
        "history360",
        "adjacent360",
        "interaction360",
        "full360",
        "full_quality360",
        "missing360",
        "quality360",
        "multiscale",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True)
    for ax, t in zip(axes, ["A", "B"]):
        for i, model in enumerate(choices):
            r = comparison(t, model, "context")
            ax.errorbar(
                r.auroc,
                i,
                xerr=[[r.auroc - r.auroc_low], [r.auroc_high - r.auroc]],
                fmt="o",
                capsize=2,
                color=COLORS[0],
            )
        ax.axvline(0, color="#999", ls="--")
        ax.set(
            yticks=range(len(choices)),
            yticklabels=choices,
            title=f"Binary {t}",
            xlabel="ΔAUROC versus matched context",
        )
        ax.invert_yaxis()
    save(fig, "09_representations")
    answers = []
    answers.append(
        dict(
            question="Did 12 or 24 hours outperform six hours?",
            answer=" ".join(
                f"Binary {t}: 12h {difference(t, 'full720', 'full360')}; 24h {difference(t, 'full1440', 'full360')}."
                for t in ["A", "B"]
            ),
        )
    )
    answers.append(
        dict(
            question="Did SMOTE improve ranking or just detection?",
            answer=" ".join(
                f"Binary {t}: {difference(t, 'compact_smote', 'compact_none')}; SMOTE AUROC {row(t, 'compact_smote').auroc:.3f}, sensitivity {row(t, 'compact_smote').sensitivity:.1%}, precision {row(t, 'compact_smote').precision:.1%}."
                for t in ["A", "B"]
            )
            + " These are numeric-only compact comparisons; contextual conditional SMOTE is a separate procedure. "
            + f"For Binary B, raw SMOTE Brier score worsens to {row('B', 'compact_smote').brier:.3f} from {row('B', 'compact_none').brier:.3f}; lower is better. Prior correction largely restores prevalence calibration but does not establish discrimination.",
        )
    )
    answers.append(
        dict(
            question="Which cutoff discriminates better?",
            answer=" ".join(
                f"Binary {t}: full 24h AUROC {row(t, 'full1440').auroc:.3f} ({row(t, 'full1440').auroc_low:.3f}–{row(t, 'full1440').auroc_high:.3f}), positive prevalence {row(t, 'full1440').prevalence:.1%}."
                for t in ["A", "B"]
            )
            + " The tasks differ; raw AP does not rank cutoffs fairly across prevalence.",
        )
    )
    answers.append(
        dict(
            question="Did representation changes help?",
            answer=" ".join(
                f"Binary {t}: observed-minute HR4 versus reading HR4 {difference(t, 'minute360', 'original360')}."
                for t in ["A", "B"]
            )
            + " History, disjoint contrasts, observed-block variation and supported activity strata have separate matched results below. No feature definition is chosen using outer performance.",
        )
    )
    best = []
    descriptive_names = {
        "context_compact_under": "context + compact physiology with random undersampling",
        "original720": "context + original four HR summaries over 12 hours",
        "context_control_over": "context only with random oversampling (added matched control)",
        "context_control_conditional": "context only with donor-context copying (added matched control)",
    }
    for t in ["A", "B"]:
        allowed = metrics[
            metrics.task.eq(t)
            & metrics.cohort.eq("all")
            & ~metrics.model.str.startswith(("permuted_", "all_"))
        ]
        r = allowed.sort_values("auroc", ascending=False).iloc[0]
        best.append(
            f"Binary {t}: {descriptive_names.get(r.model, r.model)}, AUROC {r.auroc:.3f} ({r.auroc_low:.3f}–{r.auroc_high:.3f}); fold range {r.fold_auroc_min:.3f}–{r.fold_auroc_max:.3f}."
            + f" Versus matching context: {difference(t, r.model, 'context')}."
        )
    exploratory = comparison("B", "original720", "original360")
    best.append(
        f"A secondary Binary B HR4 comparison, 12h versus 6h, has ΔAUROC {exploratory.auroc:+.4f} "
        f"({exploratory.auroc_low:+.4f} to {exploratory.auroc_high:+.4f}). "
        "This narrow positive conditional interval is one of many exploratory comparisons; its increment over context remains uncertain."
    )
    answers.append(
        dict(
            question="What is the largest logistic AUROC?",
            answer=" ".join(best)
            + " These are post-inspection point maxima on the main all-valid cohort, not independently validated winners; quality-only controls must not be relabeled physiology.",
        )
    )
    answers.append(
        dict(
            question="Does physiology beat context?",
            answer=" ".join(
                f"Binary {t}: full 24h versus context {difference(t, 'full1440', 'context')}."
                for t in ["A", "B"]
            )
            + " The full model includes missingness indicators; no-flags and missingness-only controls distinguish value and recording information.",
        )
    )
    matched_controls = paired[paired.reference.str.startswith("context_control_")].copy()
    write_json(
        public / "matched_context_controls.json", finite(matched_controls.to_dict("records"))
    )
    answers[-1]["answer"] += (
        " A disclosed post-result amendment added 40 fits with matched context-only rebalancing. "
        "All eight physiological increments have intervals including zero. "
        f"For A undersampling: {difference('A', 'context_compact_under', 'context_control_under')}. "
        "Its apparent advantage over unweighted context is therefore not evidence of added physiology. "
        "The largest A AUROC is now a context-only control; it must not be relabeled physiological signal."
    )
    answers.append(
        dict(
            question="Which features consistently contributed?",
            answer="In the secondary Binary B 12-hour HR4 model, HR SD was negative and latest HR/slope positive in all five fits. Missing-HR-mean and missing-HR-SD flags were also positive in all five fits. The leading Binary A undersampling model did not consistently retain most physiological coefficients. Correlated features can substitute; coefficient magnitudes across scaled readings and unscaled indicators are not directly comparable. These are training associations, not validated contributions. A matched 12-hour no-flags/missingness-only confirmation remains a future experiment, not an outcome-driven expansion here.",
        )
    )
    answers.append(
        dict(
            question="What remains blocked?",
            answer="Across 15 prespecified descriptive feature checks, median absolute context-residualized class differences were small: 0.037 SD (A) and 0.063 SD (B), on feature-specific observed cohorts. Six-hour full features contained 17 pairs with |r|>0.95, but none were wholly constant among observed values. Median occupied minutes were 58/360, 113/720 and 330/1440. The stricter support cohort retained 608 responses/87 people; its matched full-versus-context intervals still included zero. Retrospective wake alignment likewise did not establish an improvement. These checks do not identify one causal explanation: acquisition/receipt timing, batching, quality, counter semantics and prior-prompt logs remain unresolved, and repeated development exploration limits inference.",
        )
    )
    answers.append(
        dict(
            question="What should we discuss with Will?",
            answer="Present all paired intervals, class-specific errors, donor plausibility and calibration costs alongside AUROC. Request collection/export clock code, prompt logs and sensor-quality metadata. Keep any apparent improvement as a hypothesis for a separately specified confirmation; do not expand searches or unseal the final test automatically.",
        )
    )
    write_json(public / "answers.json", answers)
    primary_rows = pd.concat(
        [paired[paired.model.eq(a) & paired.reference.eq(b)] for a, b in primary]
    )
    clear = primary_rows[primary_rows.auroc_low.gt(0)]
    headline = (
        "The primary comparisons do not establish a clear AUROC improvement."
        if clear.empty
        else "Some exploratory comparisons have positive conditional intervals; independent confirmation is still required."
    )
    common = metrics[
        metrics.model.isin(
            [
                "all_majority",
                "all_constant",
                "context",
                "full360",
                "full720",
                "full1440",
                "compact_none",
                "compact_smote",
                "context_compact_conditional",
            ]
        )
    ]
    columns = [
        "task",
        "model",
        "responses",
        "participants",
        "auroc",
        "auroc_low",
        "auroc_high",
        "fold_auroc_mean",
        "ap",
        "prevalence",
        "balanced_accuracy",
        "sensitivity",
        "precision",
        "brier",
        "probability_sd",
    ]

    def image(name, caption):
        blob = base64.b64encode((figs / f"{name}.png").read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{blob}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'

    def details(title, body):
        return f"<details><summary>{title}</summary>{body}</details>"

    selected = pd.DataFrame(load("selection_stability"))
    stability = (
        selected[selected.model.isin(["original720", "context_compact_under", "minute360"])]
        .sort_values(["nonzero_folds", "mean_absolute_coefficient"], ascending=False)
        .head(40)
    )
    sections = [
        (
            "The question and the actual sample",
            "<p>Can longer observed histories, less sampling-sensitive representations or safer rebalancing improve participant-held-out ranking? Logistic regression is primary. Both cutoffs are exploratory thresholds of the three-item study composite, not diagnoses.</p>"
            + image(
                "01_classes",
                "Class counts share the same 3,689 development responses; participant class counts overlap.",
            )
            + details(
                "Exact class support in every training and validation fold",
                table(load("fold_support")),
            ),
        ),
        (
            "Audit before optimization",
            '<p>The audit replayed 3,706 scores and clock conversions and independently checked 74,120 saved boundaries. Original actual starts, precision buffers and corrected half-open windows remain unchanged. No implementation bug was identified in these checks. Exporter acquisition timing remains unknown.</p><div class="timeline">Synthetic: local 09:22, UTC offset−240min → UTC 13:22 → cutoff 13:21 → six-hour window [07:21,13:21). The right endpoint is excluded.</div>'
            + details(
                "Executed audit checks",
                table([json.loads((report / "audit/checks.json").read_text())]),
            )
            + table(
                [
                    {
                        "area": "Scoring and identity",
                        "confirmed": "All scores replayed; stable EMA keys and frozen folds match",
                        "unresolved": "Study composite is not a clinical diagnosis",
                    },
                    {
                        "area": "Clocks and windows",
                        "confirmed": "Actual-local minus row offset; precision buffer; half-open corrected windows",
                        "unresolved": "Acquisition versus receipt/batching and live availability",
                    },
                    {
                        "area": "Sensor acceptance",
                        "confirmed": "HR bounds/conflicts and counter resets/boundaries preserved",
                        "unresolved": "Provider quality flags and counter semantics",
                    },
                    {
                        "area": "Sampling and fitting",
                        "confirmed": "Grouped folds; training-only preprocessing/resampling; private donor traces",
                        "unresolved": "Physiological realism of synthetic feature combinations",
                    },
                    {
                        "area": "Recall",
                        "confirmed": "Morning since waking; Midday/Endday since prior prompt",
                        "unresolved": "Wake is retrospective; full prior-prompt logs absent",
                    },
                ]
            )
            + "<p>See audit.md for the confirmed/problem/unknown table and exact local source citations. No new raw extraction or HRV was performed.</p>",
        ),
        (
            "Longer windows on identical responses",
            image(
                "02_windows",
                "Single-window inputs only: shared 3,689 responses / 125 participants. Context is common across lookbacks.",
            )
            + image(
                "03_primary",
                "Prespecified paired differences. Intervals resample participants conditional on fixed predictions; no multiplicity or retuning correction.",
            )
            + table(common, columns)
            + details(
                "All model metrics and paired contrasts", table(metrics, columns) + table(paired)
            )
            + "<p>Pooled AUROC compares probabilities from different fold models; calibration differences can reverse their pooled ordering. Fold means and ranges are provided, without replacing the specified primary metric.</p>",
        ),
        (
            "SMOTE is an experiment, not an assumption",
            image(
                "04_rebalance",
                "Numeric-only standard SMOTE uses at most six train-selected features. The contextual conditional variant preserves the donor context and restricts mixing within participant/category/missingness pattern.",
            )
            + "<p>Imputation, selection, scaling and neighbor search see training rows only. Validation rows are never augmented. No eligible conditional neighbor means counted duplication, not an invented physiological neighbor. Cross-person interpolation stays a plausibility concern even when coordinate values remain bounded.</p>"
            + details("Aggregate donor plausibility audit", table(load("resampling_summary")))
            + "<p>Across the five outer training fits, 94–95% of standard SMOTE interpolations cross participants, and 40–50% use a donor with an imputed selected coordinate. These are repeated training-fit views, not unique new EMAs. The conditional variant removes cross-person/category mixing but falls back to duplicates when no eligible neighbor exists. Neither is established physiologically realistic.</p>"
            + details(
                "Original/resampled counts and donor audit for every inner/outer fit",
                table(load("resampling")),
            )
            + details("Selected logistic settings and inner thresholds", table(load("selected")))
            + "<h3>Matched context controls: disclosed comparison correction</h3>"
            + "<p>After reviewing the 600 main fits, we added exactly 40 context-only fits using the same four rebalancing procedures, grid and participant folds. This separates the effect of resampling from adding physiology. The original specification and initial aggregates are preserved; context_control_amendment.md records the timing of this amendment.</p>"
            + table(
                matched_controls[
                    ["task", "model", "reference", "auroc", "auroc_low", "auroc_high"]
                ].rename(
                    columns={
                        "auroc": "ΔAUROC",
                        "auroc_low": "95% lower",
                        "auroc_high": "95% upper",
                    }
                ),
            )
            + "<p>Every interval includes zero. For Binary A, undersampled context + physiology has AUROC 0.536 versus 0.536 for matched context alone. Context oversampling and context donor-copying are numerically identical, as expected: neither interpolates physiological coordinates.</p>",
        ),
        (
            "Ranking, detection and calibration",
            image(
                "05_confusion",
                "Actual-class rows; cell counts and percentages. A training-selected threshold can change recall without changing ranking.",
            )
            + image(
                "06_curves",
                "Numerical AP is noninterpolated average precision. Display curves use a plotting grid; metrics use original probabilities.",
            )
            + image(
                "07_calibration",
                "Fixed probability bins; cells with fewer than five responses omitted from the calibration plot only. Analytic prior correction is a sensitivity, not a guarantee of calibration.",
            )
            + details(
                "Threshold comparisons, Brier scores and per-class results",
                table(thresholds) + table(load("per_class")),
            ),
        ),
        (
            "Test the information, not just more columns",
            image(
                "09_representations",
                "Feature and recording-process controls share identical responses. History and multiscale models are explicitly separate from single-window models.",
            )
            + details("Reading versus observed-minute means", table(load("weighting_differences")))
            + details("Correlation and near-constant inputs", table(load("feature_correlations")))
            + details("Training coefficient/selection stability", table(stability))
            + "<p>Coefficients are training associations. Nonzero does not mean useful or causal. Mean/SD of observed-minute means do not interpolate empty minutes. Step-stratum HR is not resting HR or causal activity adjustment.</p>",
        ),
        (
            "Coverage is still a measurement constraint",
            image(
                "08_support",
                "Observed bins and gaps are distinct from nominal lookback duration. Overlapping EMA windows are not independent sensor records.",
            )
            + details("All lookback/survey recording diagnostics", table(quality))
            + details("Additional accepted readings beyond six hours", table(load("new_readings")))
            + details(
                "Feature and class-specific missingness",
                table(json.loads((report / "audit/missingness.json").read_text())),
            )
            + details(
                "Retained/excluded cohort composition",
                table(json.loads((report / "audit/composition.json").read_text())),
            )
            + details(
                "Restricting evaluation versus retraining on support",
                table(load("restricted_evaluation"))
                + table(metrics[metrics.cohort.eq("supported")], columns),
            ),
        ),
        (
            "Recall alignment and diagnostic controls",
            "<p>Morning items say since waking; Midday and Endday say since the last prompt. Current-survey wake answers support retrospective comparisons only. No prior completed response is used as a prompt timestamp. Both reported-minute scenarios are kept separate.</p>"
            + "<p>The nominal Morning intersection contains 227 responses from 86 participants. Wake minus fixed-hour ΔAUROC is −0.047 [−0.137,+0.051] for A and +0.0004 [−0.023,+0.026] for B; the later-minute sensitivity is also inconclusive. These small retrospective cohorts do not validate recall-aligned prediction.</p>"
            + details(
                "Retrospective Morning models: identical cohorts within each scenario",
                table(metrics[metrics.cohort.str.startswith("retro_")], columns),
            )
            + details("Survey-specific performance", table(load("survey_metrics")))
            + details(
                "Training-label permutation controls",
                table(metrics[metrics.model.str.startswith("permuted_")], columns),
            )
            + "<p>Three fixed training-label permutations are leakage diagnostics, not repeated-measures permutation p-values. Validation labels remain intact.</p>"
            + "<p>The controls show no suspiciously high ranking: pooled AUROC ranges 0.455–0.473 for A and 0.447–0.516 for B. This supports the mechanical checks but is not proof that every possible leakage mechanism is absent.</p>"
            + details("Participant influence on AUROC", table(load("influence")))
            + details(
                "Observed/context-adjusted descriptive class contrasts",
                table(load("class_associations")),
            )
            + "<p>Full-record residualization and within-person means appear only in these retrospective diagnostics. They never enter predictive features. Leave-one-person ranges do not include model retraining.</p>",
        ),
        (
            "What the results support",
            "".join(
                f"<h3>{html.escape(a['question'])}</h3><p>{html.escape(a['answer'])}</p>"
                for a in answers
            ),
        ),
        (
            "Reproducibility and limits",
            "<p>Five original development folds, three grouped inner folds; bounded C and L1/L2/Elastic Net grid. All raw data, earlier artifacts and final-test sealing are preserved. No model is selected using outer performance. Further boosting was deliberately not added: prior checkpoints tested it, while this checkpoint investigates representations and imbalance.</p>"
            + details("Verification", table([load("verification")]))
            + "<p>Full configurations, fold support, metrics, selections and calibration tables are in the versioned aggregate directory. Individual outcomes, memberships, donors, models and predictions remain ignored/private. See verification.md and completion.json for execution and preservation checks.</p>",
        ),
    ]
    css = """*{box-sizing:border-box}body{margin:0;background:#f0f5f6;color:#203e48;font:17px/1.65 system-ui,sans-serif}header,main{max-width:1200px;margin:auto;padding:38px}header{padding-top:65px}h1,h2{font-family:Georgia,serif;line-height:1.12}h1{font-size:clamp(38px,5vw,70px);max-width:1050px;margin:16px 0}h2{font-size:34px}h3{font-size:20px}p{max-width:1050px}.eyebrow{color:#087d84;letter-spacing:.18em;font-size:12px;font-weight:800}.tag,.callout,.timeline{background:#dcecee;padding:16px 22px;border-left:4px solid #16858a}.tag{display:inline-block;padding:4px 10px;border:0;font-size:14px}nav{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0}nav a{border:1px solid #b8cdd1;border-radius:22px;padding:6px 12px;color:#17646b;text-decoration:none;font-size:14px}section{background:white;border:1px solid #d8e5e8;border-radius:18px;padding:34px;margin:0 0 28px}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:13px;color:#637f8b}.table{overflow:auto;max-height:560px;margin:18px 0;border:1px solid #d8e5e8;border-radius:9px}table{border-collapse:collapse;white-space:nowrap;font-size:12px;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #dce7e9}th{background:#e8f1f3;position:sticky;top:0}details{border:1px solid #d8e5e8;border-radius:9px;padding:16px;margin:18px 0}summary{cursor:pointer;font-weight:650}footer{padding:20px 38px;color:#627e89;font-size:13px}@media(max-width:650px){header,main{padding:16px}header{padding-top:32px}section{padding:20px}h2{font-size:28px}body{font-size:15px}}@media print{body{background:white}header,main{padding:10px}section{break-inside:avoid}.table{max-height:none}nav{display:none}}"""
    nav = "".join(
        f'<a href="#s{i}">{i:02d} · {label}</a>'
        for i, label in enumerate(
            [
                "Sample",
                "Audit",
                "Windows",
                "SMOTE",
                "Calibration",
                "Features",
                "Coverage",
                "Recall",
                "Findings",
                "Checks",
            ],
            1,
        )
    )
    document = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Garmin + EMA · Binary improvement</title><style>{css}</style></head><body><header><div class="eyebrow">GARMIN + EMA · NOTEBOOK 16 · DEVELOPMENT ONLY</div><h1>Can better inputs and rebalancing reveal a stress signal?</h1><p>A controlled logistic experiment: longer windows, observed physiology, class imbalance and participant-held-out discrimination.</p><span class="tag">3,689 responses · 125 participants · final test sealed</span><nav>{nav}</nav><div class="callout"><b>{headline}</b> AUROC is primary; improved recall alone is not evidence of useful ranking.</div></header><main>'
    document += "".join(
        f'<section id="s{i}"><div class="eyebrow">{i:02d} / METHODS & FINDINGS</div><h2>{title}</h2>{body}</section>'
        for i, (title, body) in enumerate(sections, 1)
    )
    document += f"</main><footer>Version{run.name} · Exploratory research, not clinical classification · No final-test evaluation</footer></body></html>"
    (report / "index.html").write_text(document)
    summary = "# Binary stress improvement — summary for Will\n\n**" + headline + "**\n\n"
    summary += "All main comparisons retain identical 3,689 responses from 125 development participants. All 3,706 EMAs remain accounted for. Previous six-hour full AUROC was 0.508 (A) and 0.510 (B); exact refit replays are verified.\n\n"
    summary += "\n\n".join(
        f"**{i}. {a['question']}** {a['answer']}" for i, a in enumerate(answers, 1)
    )
    summary += "\n\nThe full audit, exact train/validation and resampled class counts, fold stability, Brier scores, threshold sensitivities, support restrictions and retrospective Morning comparisons are in index.html and the versioned JSON tables. Sampling intervals and activity semantics remain provisional. No clinical or causal interpretation is warranted.\n\nSee verification.md for test/replay results. No commit, push, provider message or final-test evaluation.\n"
    (report / "summary_for_will.md").write_text(summary)
    nb = nbformat.v4.new_notebook(
        metadata=dict(
            kernelspec=dict(display_name="Python3", language="python", name="python3"),
            garmin_public_aggregate_notebook=True,
        )
    )
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# 16 · Can physiological representations and rebalancing improve binary PSS discrimination?\n\nExploratory development-only logistic experiment. This teaching notebook displays completed immutable aggregate results, not participant rows. The final test remains sealed."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nroot=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/binary_improvement.json').exists())\nreport=root/'reports/binary_improvement'\npublic=report/'"
            + run.name
            + "'\ndef table(name):\n    value=json.loads((public/(name+'.json')).read_text())\n    display(pd.DataFrame([value] if isinstance(value,dict) else value))\ndef figure(name): display(Image(filename=str(public/'figures'/(name+'.png'))))"
        ),
    ]
    teaching = [
        (
            "Targets and support",
            "Items4,2,3 give11:4+(6−2)+(6−3). Binary A is≥5; Binary B is≥10. All three valid items are required.",
            'figure("01_classes")\nsupport=pd.DataFrame(json.loads((public/"fold_support.json").read_text()))\ndisplay(support[support.model.eq("context")])',
        ),
        (
            "Audit and synthetic timing",
            "Synthetic minute start09:22 → cutoff09:21. Six hours=[03:21,09:21). Original fields and offsets are preserved. An export timestamp before the cutoff does not prove live acquisition availability.",
            'display(Markdown((report/"audit.md").read_text()))',
        ),
        (
            "Matched longer-window comparisons",
            "Single-window blocks exclude reference histories and adjacent windows. Nested tuning sees only training participants.",
            'figure("02_windows")\nfigure("03_primary")\ntable("metrics")',
        ),
        (
            "SMOTE and simpler balancing",
            "Synthetic numeric vector [60,5] and minority neighbor [80,9], interpolation fraction 0.25, produce [65,6]. These are augmented feature summaries, never new sensor readings. Context categories and missingness flags are not interpolated.",
            'figure("04_rebalance")\ntable("resampling")\ndisplay(Markdown((report/"context_control_amendment.md").read_text()))\ntable("matched_context_controls")',
        ),
        (
            "Ranking versus decisions and calibration",
            "Inner-fold thresholds change decisions, not AUROC. A prior shift can damage probability calibration. Prior correction is a limited sensitivity.",
            'figure("05_confusion")\nfigure("06_curves")\nfigure("07_calibration")\ntable("thresholds")',
        ),
        (
            "Physiological information versus recording information",
            "Reading values60,60,60 in one minute and120 in another give mean75 but observed-minute mean90. Empty minutes remain missing.",
            'figure("09_representations")\ntable("weighting_differences")\ntable("feature_correlations")',
        ),
        (
            "Coverage, gaps and representation",
            "A 24-hour lookback is not 24 hours of recording. Quality predictors and support-filtered cohorts are explicitly separate.",
            'figure("08_support")\ntable("new_readings")\ntable("restricted_evaluation")',
        ),
        (
            "Recall and diagnostic limits",
            "Morning wake is known from the current response: retrospective-only. Prior prompt logs are unavailable. Full-record context residuals/person means are descriptive only and never predictors.",
            'table("survey_metrics")\ntable("influence")\ntable("class_associations")',
        ),
        (
            "Findings and verification",
            "Interpret matched differences and conditional intervals, not the largest point AUROC or improved F1 alone.",
            'display(Markdown((report/"summary_for_will.md").read_text()))\ntable("verification")\nif (report/"verification.md").exists(): display(Markdown((report/"verification.md").read_text()))',
        ),
    ]
    for title, prose, code in teaching:
        nb.cells.extend(
            [
                nbformat.v4.new_markdown_cell("## " + title + "\n\n" + prose),
                nbformat.v4.new_code_cell(code),
            ]
        )
    nbformat.write(nb, root / "notebooks/16_binary_stress_improvement.ipynb")
    return report


if __name__ == "__main__":
    print(render(Path.cwd()))
