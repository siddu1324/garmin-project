"""Standalone presentation, figures, dictionary and aggregate-only teaching notebook."""

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from garmin_preprocessing.common import write_json, verify_bundle
from garmin_inventory.privacy import sha256
from .inventory import original_inventory

LABELS = {
    "context": "Context only",
    "original60": "Original HR4 + steps · 60m",
    "hr60": "Expanded HR · 60m",
    "hr_all": "Expanded HR · 8 windows",
    "hr_activity60": "HR + activity · 60m",
    "full": "Full supported · 8 windows",
    "full_corr": "Full + correlation filter",
    "full_l1": "Full · L1 selection",
    "full_elastic": "Full · Elastic Net",
    "group_select": "Inner-selected feature group",
    "pulseox_full": "Full + provisional PulseOx",
    "dummy_majority": "Majority baseline",
    "full_balanced": "Full · balanced weights",
    "missingness": "Context + missingness only",
    "quality": "Context + quality only",
    "full_no_flags": "Full values without flags",
}
VIEWS = [
    "dummy_majority",
    "context",
    "original60",
    "hr60",
    "hr_all",
    "hr_activity60",
    "full",
    "full_corr",
    "full_l1",
    "full_elastic",
    "group_select",
]
TASK_NAMES = {
    "three_class": "Three classes: 3–4 / 5–9 / 10–15",
    "at_least_5": "Binary A: 3–4 vs 5–15",
    "at_least_10": "Binary B: 3–9 vs 10–15",
}


def markdown_table(frame):
    def cell(value):
        text = f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    rows = [list(frame.columns), ["---"] * len(frame.columns), *frame.to_numpy().tolist()]
    return "\n".join("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)


def render(root):
    root = Path(root)
    pointers = json.loads((root / "private/comprehensive_features/current.json").read_text())
    fp = root / pointers["features"]
    run = root / pointers["models"]
    out = root / "reports/comprehensive_features"
    public = out / run.name
    completion = out / "completion.json"
    if completion.exists():
        sealed = json.loads(completion.read_text())
        if sealed["model_version"] == run.name:
            verify_bundle(public)
            for name, expected in sealed["presentation_sha256"].items():
                if sha256(root / name) != expected:
                    raise RuntimeError("Sealed presentation changed; create a new version.")
            return out
        raise RuntimeError("Preserve the sealed presentation; publish a new report directory.")

    def load(n):
        return json.loads((public / (n + ".json")).read_text())

    summary = json.loads((fp / "summary.json").read_text())
    dictionary = json.loads((fp / "dictionary.json").read_text())
    coverage = pd.read_json(fp / "coverage.json")
    metrics = pd.DataFrame(load("metrics"))
    contrast = pd.DataFrame(load("paired_differences"))
    folds = pd.DataFrame(load("fold_metrics"))
    settings = pd.DataFrame(load("settings"))
    stability = pd.DataFrame(load("selection_stability"))
    inventory = original_inventory(root)
    for name, value in [
        ("feature_dictionary", dictionary),
        ("original_feature_inventory", inventory),
        ("feature_summary", summary),
    ]:
        write_json(public / (name + ".json"), value)
    coverage.to_json(public / "feature_coverage.json", orient="records", indent=2)
    matching = [
        p.parent
        for p in (root / "private/comprehensive_features/diagnostics").glob("*/manifest.json")
        if not p.parent.name.startswith(".")
        and json.loads(p.read_text()).get("inputs", {}).get("features") == sha256(fp / "manifest.json")
    ]
    if len(matching) != 1:
        raise RuntimeError("Expected one verified diagnostic bundle matching current features.")
    diag = matching[0]
    verify_bundle(diag)
    for name in ["recording_support", "retrospective_associations"]:
        write_json(public / (name + ".json"), json.loads((diag / (name + ".json")).read_text()))
    verification_path = root / "private/comprehensive_features/verification.json"
    if verification_path.exists():
        write_json(public / "verification.json", json.loads(verification_path.read_text()))
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    figs = public / "figures"
    figs.mkdir(exist_ok=True)

    def save(name):
        plt.savefig(figs / name, dpi=150, bbox_inches="tight")
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    counts = summary["count_by_modality"]
    colors = ["#286b85" if k != "pulseox" else "#c58a34" for k in counts]
    ax.bar(list(counts), list(counts.values()), color=colors)
    ax.set_ylabel("Candidate columns")
    ax.set_title("678 candidates: 438 supported summaries + 240 provisional PulseOx")
    ax.tick_params(axis="x", rotation=20)
    save("01_feature_counts.png")
    c = coverage[coverage.survey.eq("All")].pivot_table(
        index="modality", columns="lookback_minutes", values="missing_fraction", aggfunc="median"
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(1 - c.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(c.columns)), c.columns)
    ax.set_yticks(range(len(c.index)), c.index)
    ax.set_xlabel("Lookback minutes")
    ax.set_title("Median feature availability within each family")
    fig.colorbar(im, ax=ax, label="Fraction of all 3,706 responses")
    save("02_availability.png")
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), sharey=True)
    for ax, task in zip(axes, TASK_NAMES):
        f = metrics[metrics.task.eq(task)].set_index("view").loc[VIEWS]
        y = np.arange(len(f))
        ax.plot(f.macro_f1, y, "o-", label="Macro F1", color="#007b78")
        ax.plot(f.roc_auc, y, "s-", label="Macro AUROC", color="#325bb2")
        ax.plot(f.balanced_accuracy, y, "^-", label="Balanced accuracy", color="#b37821")
        ax.set_yticks(y, [LABELS[x] for x in f.index])
        ax.set_title(TASK_NAMES[task])
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    axes[-1].legend(loc="lower right")
    fig.suptitle("Same responses, same participant folds: expanded logistic models")
    fig.tight_layout()
    save("03_matched_models.png")
    cm = load("confusion_matrices")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, view in zip(axes, ["context", "original60", "full"]):
        matrix = np.array(
            next(x["matrix"] for x in cm if x["task"] == "three_class" and x["view"] == view)
        )
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(LABELS[view])
        ax.set_xticks([0, 1, 2], ["Low", "Middle", "High"])
        ax.set_yticks([0, 1, 2], ["Low", "Middle", "High"])
        ax.set_xlabel("Predicted")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                )
    axes[0].set_ylabel("Observed class")
    fig.tight_layout()
    save("04_confusion.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, color in [
        ("roc_auc", "#325bb2"),
        ("balanced_accuracy", "#b37821"),
        ("macro_f1", "#007b78"),
    ]:
        vals = []
        for m in [5, 15, 30, 60, 120, 360, 720, 1440]:
            view = "hr60" if m == 60 else f"hr_{m}m"
            vals.append(
                metrics[metrics.task.eq("three_class") & metrics.view.eq(view)][metric].iloc[0]
            )
        ax.plot(range(8), vals, "o-", label={"roc_auc": "Macro AUROC", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1"}[metric], color=color)
    ax.set_xticks(range(8), [5, 15, 30, 60, 120, 360, 720, 1440])
    ax.set_xlabel("Lookback minutes (categorical spacing)")
    ax.set_ylabel("Pooled OOF score")
    ax.legend()
    ax.set_title("Expanded HR at fixed lookbacks: exploratory comparisons")
    save("05_lookbacks.png")
    fig, ax = plt.subplots(figsize=(11, 5))
    order = [
        "context",
        "original60",
        "full",
        "full_no_flags",
        "missingness",
        "quality",
        "full_balanced",
        "pulseox_full",
    ]
    f = metrics[metrics.task.eq("three_class")].set_index("view").loc[order]
    x = np.arange(len(f))
    ax.bar(x - 0.18, f.elevated_recall, 0.36, label="High-class recall", color="#007b78")
    ax.bar(x + 0.18, f.elevated_precision, 0.36, label="High-class precision", color="#bc853a")
    ax.set_xticks(x, [LABELS.get(n, n) for n in f.index], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("High-stress class: recall gains can trade off precision")
    save("06_high_class.png")
    s = (
        stability[
            stability.task.eq("three_class")
            & stability.view.eq("full_l1")
            & stability.selected_folds.gt(0)
        ]
        .sort_values(["selected_folds", "mean_absolute_standardized_coefficient"], ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(np.arange(len(s)), s.selected_folds, color="#286b85")
    ax.set_yticks(np.arange(len(s)), s.feature)
    ax.invert_yaxis()
    ax.set_xlim(0, 5)
    ax.set_xlabel("Outer folds with nonzero coefficient (of 5)")
    ax.set_title("L1 selection stability; correlated predictors and flags can substitute")
    save("07_selection.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cal = pd.DataFrame(load("calibration"))
    for view in ["context", "original60", "full", "full_balanced"]:
        f = cal[cal.task.eq("three_class") & cal.view.eq(view) & cal.target_class.eq(2)]
        axes[0].plot(f.mean_probability, f.observed_fraction, "o-", label=LABELS[view])
        g = folds[folds.task.eq("three_class") & folds.view.eq(view)]
        axes[1].scatter(g.fold, g.roc_auc, label=LABELS[view])
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(
        xlabel="Mean predicted high-class probability",
        ylabel="Observed high-class fraction",
        title="Descriptive reliability bins",
    )
    axes[1].set(xlabel="Frozen outer fold", ylabel="Macro OVR AUROC", title="Fold stability")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save("08_calibration_folds.png")
    key = metrics[metrics.task.eq("three_class")].set_index("view")
    principal = contrast[
        contrast.task.eq("three_class")
        & contrast.model.eq("full")
        & contrast.baseline.eq("context")
    ].iloc[0]
    original = contrast[
        contrast.task.eq("three_class")
        & contrast.model.eq("full")
        & contrast.baseline.eq("original60")
    ].iloc[0]

    def ci(row, metric):
        return f"{row[metric + '_difference']:+.3f} (95% interval {row[metric + '_low']:+.3f} to {row[metric + '_high']:+.3f})"

    conclusion = (
        "The full supported feature set shows a positive conditional macro-F1 difference beyond context."
        if principal.macro_f1_low > 0
        else "The full supported feature set does not establish an improvement beyond context: its conditional macro-F1 interval includes zero."
        if principal.macro_f1_high >= 0
        else "The full supported feature set performs worse than context on macro F1 in this comparison."
    )
    conclusion += (
        f" Its three-class macro AUROC is {key.loc['full', 'roc_auc']:.3f} and balanced "
        f"accuracy is {key.loc['full', 'balanced_accuracy']:.3f}; neither matched increment "
        "over context excludes zero. Improved macro F1 alone does not establish useful physiological discrimination."
    )
    result_quality = (
        f"Full-model high-class recall is {key.loc['full', 'elevated_recall']:.1%}, precision "
        f"{key.loc['full', 'elevated_precision']:.1%}. High-class prevalence is 515/3,689 (14.0%). "
        f"Full-model log loss is {key.loc['full', 'log_loss']:.3f} versus "
        f"{key.loc['context', 'log_loss']:.3f} for context, and accuracy is "
        f"{key.loc['full', 'accuracy']:.1%} versus {key.loc['context', 'accuracy']:.1%}. "
        "The expanded model predicts minority classes more often, but probability quality worsens."
    )
    result_text = f"Full versus context: macro F1 {ci(principal, 'macro_f1')}; AUROC {ci(principal, 'roc_auc')}; balanced accuracy {ci(principal, 'balanced_accuracy')}. Full versus original HR4+steps: macro F1 {ci(original, 'macro_f1')}. {result_quality}"
    best = []
    for task in TASK_NAMES:
        f = metrics[metrics.task.eq(task) & metrics.view.isin(VIEWS[1:])]
        r = f.loc[f.macro_f1.idxmax()]
        best.append(
            dict(
                task=task,
                view=r["view"],
                macro_f1=r.macro_f1,
                roc_auc=r.roc_auc,
                balanced_accuracy=r.balanced_accuracy,
                elevated_precision=r.elevated_precision,
                elevated_recall=r.elevated_recall,
            )
        )
    write_json(public / "descriptive_point_maxima.json", best)

    def table(frame, cols=None):
        f = frame[cols] if cols else frame
        return (
            '<div class="table">'
            + f.to_html(index=False, border=0, float_format=lambda v: f"{v:.3f}", escape=True)
            + "</div>"
        )

    def image(name, caption):
        encoded = base64.b64encode((figs / name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'

    def section(number, title, body):
        return f'<section id="s{number}"><p class="eyebrow">{number:02d} / METHODS AND FINDINGS</p><h2>{title}</h2>{body}</section>'

    def details(title, body):
        return f"<details><summary>{title}</summary>{body}</details>"

    primary = metrics[metrics.task.eq("three_class")].set_index("view").loc[VIEWS].reset_index()
    primary["view"] = primary.view.map(LABELS)
    support = pd.DataFrame(load("cohort_composition"))
    support = support[
        support.cohort.eq("all_valid") & support.included & support.survey.eq("All")
    ].iloc[0]
    body = []
    body.append(
        section(
            1,
            "What was planned—and what the data can support.",
            '<p>The original <b>TULIP_Features_Simple.xlsx</b> contains <b>54 feature-family rows</b>. It is predominantly an IBI/HRV plan, not a list of 200 ready-to-use predictors. The windowed HRV part alone expands to 33 measures × 46 summaries = 1,518 conditional columns; unresolved durations and percentile definitions prevent an exact total.</p><div class="callout"><b>Before:</b> 91 supported wearable columns existed; Notebook12 modeled 43 distinct wearable columns across its experiments, with five in its primary 60-minute model.<br><b>Now:</b> 678 candidates are extracted: <b>438 supported observed summaries</b> and <b>240 provisional PulseOx summaries</b>. These counts exclude context, quality and model-generated missingness indicators.</div>'
            + image(
                "01_feature_counts.png",
                "Feature count measures breadth, not independent information or physiological validity.",
            )
            + details(
                "Every original-sheet family, implementation and blocker",
                table(
                    pd.DataFrame(inventory["rows"]),
                    [
                        "sheet_row",
                        "original_feature_family",
                        "previous_implementation",
                        "previously_modeled",
                        "current_status",
                        "current_implementation",
                        "evidence_or_blocker",
                    ],
                ),
            ),
        )
    )
    body.append(
        section(
            2,
            "One EMA response becomes one traceable feature row.",
            '<div class="flow"><b>Immutable development streams</b><span>→ accepted groups / supported counter pairs</span><span>→ eight preceding windows</span><span>→ 678 feature summaries + separate quality</span><span>→ stable EMA-key outcome join</span><b>→ held-out participant prediction</b></div><p>All <b>3,706 responses from 125 development participants</b> remain in the matrix. There are <b>3,689 valid outcome/context rows</b> for modeling. Seventeen incomplete scores remain missing. The 32 final-test participants are sealed.</p><p>Garmin Unix timestamps receive no further timezone shift. EMA UTC is actual local start minus its row’s local-minus-UTC offset. Minute-resolution starts subtract 60 seconds; second-resolution starts do not. Windows are [cutoff−L, cutoff): no future readings or boundary-crossing counter increments.</p><div class="synthetic"><b>Synthetic example:</b> local start 09:22 → cutoff 09:21 → five-minute window [09:16,09:21). Accepted HR 60,90,120 at 09:16,09:17,09:18 gives mean90, sample SD30, median90, range60 and slope30 bpm/min. Latest120 is three minutes old. Three occupied minutes do not establish complete five-minute monitoring. An example score nervous4+(6−cope2)+(6−control3)=11 is class2; those answers never enter predictors.</div>'
            + details(
                "Executed source and provenance",
                "<p>garmin_comprehensive/{features,extract,augment,models,run,evaluate,diagnostics,verify}.py; config/comprehensive_features.json. Frozen original HR/counter values matched in326,128 checks; existing60-minute history matched in11,118 checks. Source hashes and row slices remain private. No source measurements were modified.</p>",
            ),
        )
    )
    body.append(
        section(
            3,
            "Feature families describe observations—not continuous physiology.",
            '<ul><li><b>HR:</b>30 summaries per window: distribution, shape, endpoints, OLS trend, half-window change, adjacent-reading differences, observed-minute and rolling summaries.</li><li><b>Steps/floors:</b>7 supported-pair summaries each. Rates use supported seconds; they are not whole-window activity totals.</li><li><b>Co-observation:</b>HR during intervals with zero/positive step increments; not causal activity adjustment.</li><li><b>Multiscale:</b>mean/SD differences between adjacent non-overlapping intervals.</li><li><b>Earlier reference:</b>48 reference/deviation columns. At each L, history [c−(L+1440)m,c−Lm) excludes the current window; ≥30 observed UTC minutes in ≥3 UTC hours. Standardized deviations require nonzero reference SD.</li></ul><p>HR acceptance remains30–240bpm with existing clock/conflict rules. Distribution summaries need≥2 readings/span≥1s; slope≥3/span≥60s. Shape estimates need≥8/span≥60s and nonconstant values. Adjacent changes require two original valid pairs with gaps≤60s; removing a rejected reading never creates adjacency. Rolling five-minute blocks require≥3 occupied minute bins; empty bins remain missing.</p><div class="warning"><b>Separate or blocked:</b>PulseOx70–100 is a provisional percent-like branch. No new real HRV is calculated: beat ordering, timestamp anchoring and quality remain unverified. Respiration exists only in current-survey embedded fields, not a preceding stream. Calories/intensity semantics are unresolved. Proprietary Garmin stress and VLF remain excluded.</div>'
            + details(
                "Complete dictionary: units, formulas, sources and support",
                table(
                    pd.DataFrame(dictionary),
                    ["name", "units", "formula", "requirements", "source", "status", "limitation"],
                ),
            ),
        )
    )
    body.append(
        section(
            4,
            "Coverage and missingness constrain the experiment.",
            image(
                "02_availability.png",
                "Each cell summarizes feature availability, not complete sensor coverage. Differences within a family remain in the full coverage table.",
            )
            + "<p>Only <b>one</b> response has all supported features observed. Requiring complete cases across the full matrix would make the experiment unusable. Every primary comparator therefore uses the same3,689 valid-score/context responses with training-only feature medians. All-missing and constant training columns are removed; missingness flags are trained separately. No target or raw sensor reading is filled.</p><p>The original five-feature60-minute complete cohort is also refitted and compared with imputed predictions on identical validation responses; the different training populations are labeled. Quality-only, missingness-only and values-without-flags controls test whether availability explains apparent signal.</p>"
            + details(
                "Per-feature availability and missingness",
                table(
                    coverage[coverage.survey.eq("All")],
                    [
                        "feature",
                        "modality",
                        "lookback_minutes",
                        "available",
                        "missing",
                        "available_participants",
                        "missing_fraction",
                    ],
                ),
            ),
        )
    )
    body.append(
        section(
            5,
            "Logistic regression first, with selection inside training.",
            '<p>Targets stay fixed: three classes3–4/5–9/10–15; Binary A≥5; Binary B≥10. These are exploratory categories, not clinical diagnoses. Context is actual-local cyclic time and training-encoded survey type.</p><p>Five original participant folds define held-out predictions. Three grouped inner folds choose regularization and feature groups. Medians, scaling, constant removal, correlation filters and encoders are fitted within every training split.</p><p>L2 tests C = {0.01, 0.1, 1, 10}. L1 and Elastic Net (mixing ratio 0.5) test C = {0.01, 0.1, 1}. Correlation filtering removes later features with |r|&gt;.95 using training values only. Group selection chooses HR-all, HR/activity-all or full using inner macro F1. Class balancing is a separate sensitivity. No boosting model was trained.</p><div class="note">375 unique outer fits cover25 comparisons × 3 tasks × 5 folds. An initial scheduler listed the identical quality control twice; the extra15 entries reused the same jobs. The schedule and completeness counter were corrected, with a regression test. No predictions or fit decisions changed.</div>'
            + details(
                "Fold-specific selected settings and feature counts",
                table(
                    settings,
                    [
                        "task",
                        "view",
                        "fold",
                        "block",
                        "C",
                        "l1_ratio",
                        "transformed_features",
                        "selected_nonzero",
                        "n_iter",
                    ],
                ),
            ),
        )
    )
    body.append(
        section(
            6,
            "Do expanded inputs improve matched predictions?",
            f"<p><b>{html.escape(conclusion)}</b></p><p>{html.escape(result_text)}</p>"
            + table(
                primary,
                [
                    "view",
                    "macro_f1",
                    "roc_auc",
                    "balanced_accuracy",
                    "elevated_precision",
                    "elevated_recall",
                ],
            )
            + image(
                "03_matched_models.png",
                "All imputed comparisons share training/validation membership. Differences across model families here concern feature sets and logistic penalties, not new participant splits.",
            )
            + "<p>Paired95% intervals resample participants1,000 times while keeping saved predictions fixed. They do not incorporate retuning uncertainty, correct for many comparisons, or turn repeated development exploration into confirmation.</p>"
            + details(
                "All tasks and matched uncertainty",
                table(
                    contrast,
                    [
                        "task",
                        "model",
                        "baseline",
                        "cohort",
                        "macro_f1_difference",
                        "macro_f1_low",
                        "macro_f1_high",
                        "roc_auc_difference",
                        "roc_auc_low",
                        "roc_auc_high",
                        "balanced_accuracy_difference",
                        "balanced_accuracy_low",
                        "balanced_accuracy_high",
                    ],
                ),
            ),
        )
    )
    body.append(
        section(
            7,
            "High-stress detection and probability quality.",
            image(
                "04_confusion.png",
                "Three-class confusion counts on the same3,689 held-out development responses. Rows are observed classes; columns are predictions.",
            )
            + image(
                "06_high_class.png",
                "Weighting can increase recall while reducing precision. PulseOx results remain provisional.",
            )
            + image(
                "08_calibration_folds.png",
                "Calibration bins are descriptive and can be sparse. Weighted probabilities are not assumed calibrated. Fold variation limits claims from small pooled differences.",
            )
            + "<p>Binary A’s positive class is PSS5–15, not the10–15 high-stress category. High-class performance means class2 in the three-class task or the positive class in Binary B. Macro AUROC evaluates probability ranking; balanced accuracy evaluates the default class decision. They answer different questions.</p>"
            + details(
                "Per-class metrics and exact fold support",
                table(
                    pd.DataFrame(load("per_class")),
                    ["task", "view", "target_class", "precision", "recall", "f1", "support"],
                )
                + table(
                    folds,
                    [
                        "task",
                        "view",
                        "fold",
                        "training_responses",
                        "validation_responses",
                        "training_support",
                        "validation_support",
                    ],
                ),
            ),
        )
    )
    body.append(
        section(
            8,
            "More columns do not guarantee more information.",
            image(
                "05_lookbacks.png",
                "These are prespecified lookbacks, not a search yielding an optimal clinical window.",
            )
            + image(
                "07_selection.png",
                "Selection frequency is descriptive. A coefficient can stand in for a correlated feature or missingness pattern; it is not a causal effect.",
            )
            + "<p>Many new columns summarize the same readings. SD/variance, quantiles, overlapping windows and historical deviations can be redundant. Retrospective person-centered associations are diagnostic only; their full-record means never enter the models. The earlier53.5% within-person score variation does not establish that wearable measurements explain it.</p><p>Morning asks since waking; Midday and Endday ask since the last prompt. Fixed lookbacks need not match those recall intervals. Sparse or stale readings can satisfy minimum arithmetic support. Acquisition-versus-receipt timestamps, batching, source quality, activity confounding and selection remain unresolved.</p>"
            + details(
                "Recording support and retrospective associations",
                table(pd.DataFrame(load("recording_support")))
                + table(pd.DataFrame(load("retrospective_associations"))),
            ),
        )
    )
    body.append(
        section(
            9,
            "What to take to Will.",
            f"<p>{html.escape(conclusion)} The original plan is now reconciled row by row; unavailable HRV and undocumented modalities were not fabricated to inflate the feature count.</p><p><b>Next:</b> resolve exporter timestamp/batching and quality metadata, and obtain prompt/wake boundary evidence. If a feature-group increment has a consistently positive paired interval, specify a narrow confirmation using the same cohort controls before any new model search. If intervals include zero or favor context, do not automatically expand complexity. Keep the final test sealed until the analysis and measurement contract are frozen.</p><p>Ask the provider for the collection/exporter version, timestamp assignment code, original ordered IBI payloads and rejected-beat flags, counter semantics, and actual prompt delivery logs. This report drafts the questions only.</p>"
            + details(
                "Descriptive point maxima—not selected winners",
                table(pd.DataFrame(best))
                + "<p>These are the largest observed macro-F1 point estimates among the prespecified primary logistic feature comparisons. They are not an unbiased estimate of a selected winner, and no final-test evaluation occurred.</p>",
            )
            + details(
                "Verification and reproducibility",
                "<p>Existing tests, boundary/adjacency/history checks, train-only leakage tests, independent synthetic positive control, saved-estimator replays, fresh-fit checks, privacy scans and raw-input integrity checks are recorded in verification.json and verification.md. All participant-level matrices, predictions, coefficients attached to estimators and memberships remain ignored/private. No commit, push or provider message.</p>",
            ),
        )
    )
    css = """*{box-sizing:border-box}body{margin:0;background:#f3f6f7;color:#19333e;font:18px/1.6 system-ui,sans-serif}header,main,footer{max-width:1160px;margin:auto;padding:35px}header{padding-top:70px}h1,h2{font-family:Georgia,serif;font-weight:500;line-height:1.15}h1{font-size:55px}h2{font-size:36px;margin-top:0}section{background:white;border:1px solid #d9e2e6;border-radius:16px;padding:38px;margin-bottom:28px}.eyebrow{font-size:12px;color:#007e7a;letter-spacing:.14em;font-weight:750}nav{display:flex;gap:14px;flex-wrap:wrap}a{color:#2455a0}figure{margin:25px 0}img{display:block;width:100%;height:auto}figcaption{font-size:15px;color:#526872;margin-top:10px}.table{overflow:auto;max-height:650px;margin:20px 0;border:1px solid #dce5e8;border-radius:8px}table{border-collapse:collapse;font-size:14px;min-width:100%;text-align:left}td,th{padding:10px;border-bottom:1px solid #dce5e8;vertical-align:top}th{background:#ecf3f5;position:sticky;top:0}details{border:1px solid #d8e3e7;border-radius:8px;padding:15px;margin:20px 0}summary{cursor:pointer;font-weight:650;color:#245767}.callout,.note,.synthetic,.warning{border-left:4px solid #00817b;background:#edf7f5;padding:22px;margin:22px 0}.warning{border-color:#ba852e;background:#fff6e8}.synthetic{border-color:#3265ac;background:#f0f5fc}.flow{display:grid;gap:10px;grid-template-columns:repeat(3,1fr);padding:25px;background:#eef5f7;border-radius:10px}.flow>*{padding:10px}.lead{font-size:22px;color:#4c6570}code{overflow-wrap:anywhere}button{padding:10px 15px;background:white;border:1px solid #b6ced5;border-radius:6px;cursor:pointer}footer{font-size:15px;color:#506775}@media(max-width:600px){header,main,footer{padding:20px}h1{font-size:38px}h2{font-size:28px}section{padding:22px}.flow{grid-template-columns:1fr}.table{max-height:500px}}@media print{body{font-size:11pt;background:white}nav,button{display:none}header,main{padding:0}section{break-before:page;border:0;padding:0}h1{font-size:32pt}h2{font-size:24pt}.table{max-height:none;overflow:visible}table{font-size:8pt}figure{break-inside:avoid}details{break-inside:auto}}"""
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Comprehensive physiological features · Garmin + EMA</title><style>'
        + css
        + '</style></head><body><header><p class="eyebrow">GARMIN × EMA / EXPLORATORY DEVELOPMENT RESEARCH / 17 OCTOBER 2026</p><h1>More physiological features.<br>A fair test of what they add.</h1><p class="lead">From the original feature sheet to678 transparent candidate columns, participant-grouped logistic evaluation and measurement limits.</p><nav>'
        + "".join(f'<a href="#s{i}">{i:02d}</a>' for i in range(1, 10))
        + '</nav><p><button onclick="window.print()">Print / PDF</button> <button onclick="document.querySelectorAll(\'details\').forEach(d=>d.open=!d.open)">Toggle technical details</button></p></header><main>'
        + "".join(body)
        + "</main><footer>All displayed examples are synthetic or aggregate. Timing and physiological validity remain provisional; this is not validated real-time prediction or a clinical diagnostic tool.</footer></body></html>"
    )
    # Fix prose spacing without touching identifiers or numerical values.
    import re

    def prose_spacing(text):
        text = re.sub(
            r"\b(all|All|the|The|the|same|of|in|to|with|from|and|only|Only|across|versus|before|after|has|contains|than|per|rows|folds|responses|Notebook|remains|Paired|paired|its|classes|class|start|HR|SD|sampleSD|columns|Now|mean|median|range|slope|Latest|nervous|cope|control|feature|cover|extra|earlier|PSS)(?=\d)",
            r"\1 ",
            text,
        )
        text = re.sub(r"(?<=[A-Za-z])(?=[≥≤])", " ", text)
        return re.sub(r"(?<=\d)(bpm|min|s)\b", r" \1", text)

    page = prose_spacing(page)
    (out / "index.html").write_text(page)
    dictionary_md = (
        "# Comprehensive feature dictionary\n\n678 candidate columns;438 supported observed summaries and240 provisional PulseOx columns.\n\n"
        + markdown_table(pd.DataFrame(dictionary))
        + "\n"
    )
    (out / "feature_dictionary.md").write_text(prose_spacing(dictionary_md))
    (out / "original_feature_inventory.md").write_text(
        "# Original feature-sheet reconciliation\n\n"
        + inventory["counting_note"]
        + "\n\n"
        + markdown_table(pd.DataFrame(inventory["rows"]))
        + "\n"
    )
    summary_md = (
        "# Comprehensive physiological features — summary for Will\n\n"
        + conclusion
        + "\n\n"
        + result_text
        + "\n\n"
        + markdown_table(
            primary[
                [
                    "view",
                    "macro_f1",
                    "roc_auc",
                    "balanced_accuracy",
                    "elevated_precision",
                    "elevated_recall",
                ]
            ]
        )
        + "\n\n## All three tasks: descriptive point maxima\n\n"
        + markdown_table(pd.DataFrame(best))
        + "\n\n## Feature counts\n\n"
        + markdown_table(pd.DataFrame(list(summary["count_by_modality"].items()), columns=["Family", "Columns"]))
        + "\n\nEach lookback has 85 candidates except 24 hours (83): adjacent-window contrasts are omitted there. The supported full model offers 438 physiological columns before training-only removal. Its three-class folds use 880–881 transformed inputs including encoded context and missingness flags; without flags, 442–443. The original five physiological features use 15 transformed inputs with context and flags. Correlation filtering retains 765–782 transformed inputs; L1 retains 788–815 nonzero coefficients. These larger numbers are model columns, not additional physiological measurements.\n\nAll supported physiological modalities at 60 minutes are HR and activity; that requested comparator is the HR + activity 60-minute row, not a duplicate experiment. PulseOx remains a separate sensitivity.\n"
        + "\n\nThese maxima are exploratory, not selected or validated winners. Read paired_differences.json for uncertainty.\n\nOriginal workbook:54 family rows, with33 windowed IBI measure variants and1,518 conditional IBI summaries; no exact overall scalar total is specified. Prior supported wearable columns91; prior modeled distinct wearable columns43. Now678 candidates (438 supported-summary branch,240 provisional PulseOx), all3,706 development rows retained;3,689 scored/context-valid rows modeled. Only one response is complete across all supported columns.\n\nNo new real HRV, final-test evaluation, raw changes, commit, push or provider message. Missingness controls are distinct from physiology. Existing timing/recall and source-quality uncertainties limit interpretation.\n\nNext: resolve measurement provenance and recall boundaries; do not automatically expand model complexity.\n"
    )
    (out / "summary_for_will.md").write_text(prose_spacing(summary_md))
    # Aggregate-only notebook; no private participant table is ever loaded by a cell.
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
        garmin_public_aggregate_notebook=True,
    )
    cells = []

    def md(s):
        cells.append(nbf.v4.new_markdown_cell(prose_spacing(s)))

    def code(s):
        cells.append(nbf.v4.new_code_cell(s))

    md(
        "# 13 · Comprehensive physiological features\n\nAn exploratory development experiment. We test whether a broader, documented feature set adds information beyond time/survey context. The32 final-test participants remain sealed.\n\nAll tables below are aggregates; examples are synthetic."
    )
    code(
        "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nROOT = Path.cwd() if (Path.cwd() / 'reports').exists() else Path.cwd().parent\nREPORT = ROOT / 'reports/comprehensive_features/"
        + run.name
        + "'\ndef load(name):\n    return json.loads((REPORT / (name + '.json')).read_text())\ndef table(name, columns=None, rows=20):\n    f = pd.DataFrame(load(name))\n    display((f[columns] if columns else f).head(rows))\ndef figure(name):\n    display(Image(filename=str(REPORT / 'figures' / name)))"
    )
    md(
        "## 1 · Available modalities and the original plan\n\nThe source sheet has54 family rows. HRV depends on unresolved beat timing/quality; respiration has no preceding stream. PulseOx is separate and provisional. The complete row-by-row inventory is in the report."
    )
    code(
        "display(pd.Series(load('feature_summary')['count_by_modality'], name='candidate_columns').to_frame())\nfigure('01_feature_counts.png')"
    )
    md(
        "## 2 · Feature inventory and support\n\nThe full dictionary records each column, formula, units and support. New HR summaries are observed-reading features, not HRV. Steps/floors are supported counter increments. Earlier histories never overlap their current feature window."
    )
    code(
        "table('feature_dictionary',['name','units','requirements','status'],15)\ninventory = load('original_feature_inventory')\ndisplay(Markdown(inventory['counting_note']))\ndisplay(pd.DataFrame(inventory['rows'])[['sheet_row','original_feature_family','current_status']])"
    )
    md(
        "## 3 · Synthetic calculation and time boundary\n\nMinute start09:22→cutoff09:21→[09:16,09:21). For accepted HR60,90,120 at one-minute spacing, mean=90, sampleSD=30, range=60, median=90 and slope=30bpm/min. Current-survey answers form the target only:4+(6−2)+(6−3)=11→class2. No real participant row is displayed."
    )
    code(
        "import numpy as np\nx=np.array([60.,90.,120.])\ndisplay(pd.DataFrame([{'synthetic_mean_bpm':x.mean(),'synthetic_sample_sd_bpm':x.std(ddof=1),'synthetic_median_bpm':np.median(x),'synthetic_range_bpm':np.ptp(x)}]))"
    )
    md(
        "## 4 · Availability and train-only missing-data handling\n\nAll3,706 rows remain in the feature matrix;3,689 valid-score/context rows enter every primary comparator. Only one response is complete across all supported features. Impute feature summaries from each training split, never outcomes or raw readings. Counts and gaps are separate quality diagnostics."
    )
    code(
        "figure('02_availability.png')\ntable('cohort_composition',['cohort','included','survey','responses','participants','scores_available','class_support'],20)"
    )
    md(
        "## 5 · Matched logistic comparisons\n\nFrozen five outer participant folds; three grouped inner folds. MacroF1 selects C and feature groups inside training. L2, L1, ElasticNet and correlation filtering are bounded prespecified comparisons. Every preprocessing step is fitted within the corresponding training split."
    )
    code(
        "figure('03_matched_models.png')\nf=pd.DataFrame(load('metrics'))\ndisplay(f[f.task.eq('three_class')][['view','responses','participants','macro_f1','roc_auc','balanced_accuracy','elevated_precision','elevated_recall']])"
    )
    md(
        "## 6 · Expanded lookbacks and controls\n\nQuality/missingness controls do not represent physiology. A provisional PulseOx addition does not establish validated oxygenation measurements. There is no optimal-window selection from the held-out results."
    )
    code("figure('05_lookbacks.png')\nfigure('06_high_class.png')")
    md(
        "## 7 · Feature selection and stability\n\nNo selection uses outer validation labels. Coefficients and selection frequencies are descriptive; correlated summaries and missingness flags can substitute for one another."
    )
    code(
        "figure('07_selection.png')\ntable('settings',['task','view','fold','block','C','l1_ratio','transformed_features','selected_nonzero'],15)"
    )
    md(
        "## 8 · All targets, class support and uncertainty\n\nBinaryA≥5 differs from high-stress BinaryB≥10. Paired participant bootstrap intervals condition on these fixed development predictions; they do not account for retuning or multiple exploratory comparisons."
    )
    code(
        "table('descriptive_point_maxima',rows=3)\ntable('paired_differences',['task','model','baseline','macro_f1_difference','macro_f1_low','macro_f1_high','roc_auc_difference','roc_auc_low','roc_auc_high'],20)\nfigure('04_confusion.png')"
    )
    md(
        "## 9 · Interpretation diagnostics\n\nIrregular sampling and stale readings can pass arithmetic support. Retrospective person-centered correlations never enter predictors. Probability calibration and fold stability help distinguish rank signal from the default decision threshold."
    )
    code("figure('08_calibration_folds.png')\ntable('retrospective_associations',rows=6)")
    md(
        "## 10 · Conclusions and next justified step\n\n"
        + conclusion
        + "\n\n"
        + result_text
        + "\n\nResolve exporter clocks/batching, signal quality and recall-period boundaries. No automatic boosting expansion. Full source references, counts and original-plan reconciliation are in the standalone HTML and feature dictionary."
    )
    code(
        "display(Markdown((ROOT / 'reports/comprehensive_features/summary_for_will.md').read_text()))\nif (REPORT / 'verification.json').exists():\n    display(pd.Series(load('verification'),name='verification').to_frame())"
    )
    nb.cells = cells
    nbf.write(nb, root / "notebooks/13_comprehensive_physio_features.ipynb")
    print(out / "index.html")
    return out


if __name__ == "__main__":
    render(Path.cwd())
