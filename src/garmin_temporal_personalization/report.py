"""Self-contained aggregate presentation and executed-notebook source."""

import base64
import html
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat
from garmin_preprocessing.common import write_json, verify_bundle
from garmin_inventory.privacy import sha256


def table(value, columns=None):
    f = pd.DataFrame(
        [{"measure": k, "value": v} for k, v in value.items()] if isinstance(value, dict) else value
    )
    if columns is not None:
        f = f.reindex(columns=columns)
    return (
        '<div class="table">'
        + f.to_html(
            index=False, border=0, na_rep="—", float_format=lambda x: f"{x:.3f}", escape=True
        )
        + "</div>"
    )


def report(root):
    root = Path(root)
    folder = root / "reports/temporal_personalization"
    pointer = json.loads((root / "private/temporal_personalization/current.json").read_text())
    run = root / pointer["run"]
    public = folder / run.name
    if (folder / "completion.json").exists():
        seal = json.loads((folder / "completion.json").read_text())
        assert seal["complete"] and all(
            sha256(root / p) == h for p, h in seal["presentation_sha256"].items()
        )
        verify_bundle(public)
        return folder

    def load(n):
        return json.loads((public / f"{n}.json").read_text())

    def audit(n):
        return json.loads((folder / "audit" / f"{n}.json").read_text())

    metrics = pd.DataFrame(load("metrics"))
    paired = pd.DataFrame(load("paired"))
    long = audit("longitudinal")
    consistency = audit("consistency")
    pool = pd.DataFrame(audit("prior_pooling"))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    figures = public / "figures"
    figures.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    def image(name, caption):
        encoded = base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
        return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>'

    def row(t, m):
        return metrics[metrics.task.eq(t) & metrics.model.eq(m)].iloc[0]

    def delta(t, m, r):
        z = paired[paired.task.eq(t) & paired.model.eq(m) & paired.reference.eq(r)].iloc[0]
        return f"{z.auroc:+.3f} [{z.auroc_low:+.3f}, {z.auroc_high:+.3f}]"

    dist = pd.DataFrame(audit("item_distribution"))
    fig, axs = plt.subplots(2, 2, figsize=(11, 6))
    for ax, (name, title) in zip(
        axs.flat,
        [
            ("composite", "Composite: 3–15"),
            ("item_nervous", "Nervousness: stored 1–5"),
            ("item_cope", "Coping: reversed, 6 − stored"),
            ("item_control", "Control: reversed, 6 − stored"),
        ],
    ):
        z = dist[dist.measure.eq(name)]
        ax.bar(z.value, z.fraction * 100, color="#177e89")
        ax.set(title=title, xlabel="Score", ylabel="Responses (%)")
    save(fig, "01_labels")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set(
        xlim=(6.8, 9.7),
        ylim=(-0.5, 3.5),
        yticks=[0, 1, 2, 3],
        yticklabels=[
            "HR observation window",
            "Prior answer: verified",
            "Prior answer: delayed",
            "Prior answer: unknown",
        ],
        xlabel="Synthetic local time",
    )
    ax.broken_barh([(8.35, 1)], (-0.25, 0.5), facecolors="#8ac5c0")
    ax.scatter([8.5, 9, 9 + 20 / 60], [0, 0, 0], color="#177e89", s=28, zorder=3)
    ax.scatter([9.35, 9.5], [0, 0], color="#b45645", marker="x", s=35, zorder=3)
    ax.text(8.4, -0.4, "Dots: included readings; crosses: at/after cutoff", fontsize=8)
    ax.plot([7, 7 + 5 / 60], [1, 1], color="#177e89", marker="o", lw=3)
    ax.text(7.2, 1.12, "Start 07:00 → receipt 07:05: usable", fontsize=9)
    ax.plot([9 + 10 / 60, 9 + 25 / 60], [2, 2], color="#b45645", marker="o", lw=3)
    ax.text(6.95, 2.12, "Receipt 09:25: unavailable at cutoff", fontsize=9)
    ax.scatter([7.5], [3], color="#80878c", s=50)
    ax.text(7.65, 3.08, "No receipt clock: blocked, even if start is earlier", fontsize=9)
    ax.axvline(9.35, color="#183e4a", ls="--")
    ax.text(9.38, 0.55, "09:21 cutoff", rotation=90, fontsize=9)
    ax.set_xticks([7, 7.5, 8, 8.5, 9, 9.5], ["07:00", "07:30", "08:00", "08:30", "09:00", "09:30"])
    save(fig, "02_timeline")
    examples = [("A", "context_control_over"), ("A", "context_compact_under"), ("B", "original720")]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(examples))
    for j, (col, label, color) in enumerate(
        [
            ("pooled_auroc", "Pooled", "#177e89"),
            ("within_fold_pair_weighted_auroc", "Within fold, pair weighted", "#d29548"),
            ("cross_fold_pair_auroc", "Cross-fold pairs", "#6c719b"),
        ]
    ):
        vals = [pool[pool.task.eq(t) & pool.model.eq(m)].iloc[0][col] for t, m in examples]
        ax.bar(x + (j - 1) * 0.23, vals, width=0.22, label=label, color=color)
    ax.axhline(0.5, color="#777", ls=":")
    ax.set(
        ylim=(0.35, 0.65),
        ylabel="AUROC",
        xticks=x,
        xticklabels=["A · context oversampling", "A · physiology undersampling", "B · 12h HR4"],
    )
    ax.legend(ncol=3, fontsize=9)
    save(fig, "03_pooling")
    names = [
        "context",
        "physiology",
        "raw",
        "reference",
        "personal",
        "extended",
        "hgb_context",
        "hgb_personal",
    ]
    labels = [
        "Context",
        "Physiology only",
        "Context + raw",
        "Context + reference",
        "Context + personal",
        "Personal + clock/activity",
        "HGB context",
        "HGB personal",
    ]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, t in zip(axs, ["A", "B"]):
        z = pd.DataFrame([row(t, m) for m in names])
        ax.errorbar(
            z.auroc,
            np.arange(len(z)),
            xerr=[z.auroc - z.auroc_low, z.auroc_high - z.auroc],
            fmt="o",
            color="#177e89",
            capsize=3,
        )
        ax.axvline(0.5, ls=":", color="#777")
        ax.set(
            yticks=np.arange(len(z)),
            yticklabels=labels,
            title=f"Binary {t}",
            xlabel="Pooled AUROC · 95% conditional interval",
            xlim=(0.32, 0.64),
        )
        ax.invert_yaxis()
    save(fig, "04_models")
    contrasts = [
        ("raw", "context"),
        ("personal", "raw"),
        ("personal", "reference"),
        ("extended", "personal"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, t in zip(axs, ["A", "B"]):
        z = pd.DataFrame(
            [
                paired[paired.task.eq(t) & paired.model.eq(m) & paired.reference.eq(r)].iloc[0]
                for m, r in contrasts
            ]
        )
        ax.errorbar(
            z.auroc,
            np.arange(4),
            xerr=[z.auroc - z.auroc_low, z.auroc_high - z.auroc],
            fmt="o",
            color="#177e89",
            capsize=3,
        )
        ax.axvline(0, ls=":", color="#777")
        ax.set(
            yticks=np.arange(4),
            yticklabels=[
                "Raw − context",
                "Personal − raw",
                "Personal − reference",
                "Extended − personal",
            ],
            xlabel="Paired ΔAUROC",
            title=f"Binary {t}",
        )
        ax.invert_yaxis()
    save(fig, "05_increment")
    availability = pd.DataFrame(load("availability"))
    cols = [
        "hr__mean__60m",
        "reference_mean_bpm",
        "clock_reference_mean",
        "clock_mean_deviation",
        "step_block_rate_change",
    ]
    z = availability[availability.cohort.eq("all")].set_index("feature").loc[cols]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(np.arange(5), z.available, color="#177e89")
    ax.bar_label(bars, padding=4)
    ax.set(
        yticks=np.arange(5),
        yticklabels=[
            "Current HR mean",
            "Earlier 24h HR reference",
            "Clock-matched HR reference",
            "Current − clock reference",
            "24h-block supported step change",
        ],
        xlim=(0, 3689),
        xlabel="Available responses / 3,689",
    )
    ax.invert_yaxis()
    save(fig, "06_availability")
    cal = pd.DataFrame(load("calibration"))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, t in zip(axs, ["A", "B"]):
        for m in ["context", "personal", "hgb_personal"]:
            z = cal[cal.task.eq(t) & cal.model.eq(m) & cal.responses.ge(5)]
            ax.plot(z.mean_probability, z.prevalence, "o-", label=m)
        ax.plot([0, 1], [0, 1], ":", color="#777")
        ax.set(
            xlabel="Mean predicted probability",
            ylabel="Observed positive fraction",
            title=f"Binary {t}",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=9)
    save(fig, "07_calibration")
    # Make the requested table explicit about unavailable history metrics.
    comparison = pd.DataFrame(load("comparison"))
    primary = paired[[(m, r) in contrasts for m, r in zip(paired.model, paired.reference)]]
    summary = f"""# Temporal alignment and personalization — summary for Will

**Earlier wearable references did not establish a physiological prediction gain. EMA-history prediction remains untested because answer availability is not timestamped.**

All 3,706 development responses remain accounted for; 3,689 valid-score/context responses from 125 people form the main matched cohort. The five original participant folds and sealed final test are unchanged.

## What we learned

- **Labels:** pooled internal-consistency alpha {consistency["alpha"]:.3f}; person-centered alpha {consistency["person_centered_alpha"]:.3f}. These are descriptive covariance measures, not validated reliability or proof of clinical meaning. Morning refers to since waking; Midday/Endday to the last prompt.
- **Temporal EMA structure:** {long["retrospective_pairs"]:,} consecutive valid-response pairs have retrospective lag-one correlation {long["lag1_correlation"]:.3f}; after full-record person centering it is {long["retrospective_person_centered_lag1"]:.3f}. These are associations, not available-at-cutoff predictions. Centering never enters a model.
- **Online-history feasibility:** {long["potential_later_responses"]:,} later responses across {long["potential_calibratable_people"]} people follow five valid starts, but **zero** have verified historical-answer availability. Earlier starts do not prove submission/receipt. Previous-EMA, persistence and personalized-label predictions are therefore blocked, not scored as zero performance.
- **Raw physiology versus context:** ΔAUROC A {delta("A", "raw", "context")}; B {delta("B", "raw", "context")}.
- **Personalized wearable representations versus raw:** A {delta("A", "personal", "raw")}; B {delta("B", "personal", "raw")}. Versus reference alone: A {delta("A", "personal", "reference")}; B {delta("B", "personal", "reference")}.
- **Clock-matched/activity extension:** versus personal block A {delta("A", "extended", "personal")}; B {delta("B", "extended", "personal")}. No primary interval establishes a gain.
- **Secondary boosting:** context+personal pooled AUROC A {row("A", "hgb_personal").auroc:.3f}, B {row("B", "hgb_personal").auroc:.3f}; the matching context increments are A {delta("A", "hgb_personal", "hgb_context")}, B {delta("B", "hgb_personal", "hgb_context")}.
- **Recording constraints:** raw/reference-observed cohort 1,312 responses/112 people; strict 60-minute support 1,115/104. Sensor coverage, complete observed rows and all-row imputed comparisons are not interchangeable. Empty periods remain unobserved.
- **Pooling:** a pooled OOF AUROC mixes probabilities from different fitted models. Notebook 16 A context oversampling has within-fold ranking near chance despite pooled AUROC≈.542. We decomposed within-/cross-fold pairs without normalizing predictions using validation labels.

## What is not resolved

This does not show that physiology is fundamentally uninformative. It shows no established gain from these fixed representations under the recorded-clock pipeline. Its increment **beyond EMA history cannot yet be tested**. Associations after adjusting for survey/time/activity remain small with participant-bootstrap intervals spanning zero. Missingness controls and feature-level imputation do not establish clean separation of physiological and recording information.

No real personalized chronological benchmark or change-state predictor is claimed. The ≥2-point increase threshold is exploratory, not a validated meaningful-change threshold. Sleep/HRV/respiration/PulseOx are not introduced as validated features. Garmin acquisition/receipt, source quality and prompt semantics remain unresolved.

## Recommended next experiment

Obtain response-version submission/receipt clocks and complete prompt logs. Then run the specified matched previous-EMA versus previous-EMA+physiology comparison, followed by prequential personalization after five verified available answers, with permitted population training labels also available before fitting. No guessed completion delays, automatic new model search or final-test unsealing.

The history constructor and chronological baseline are implemented and synthetically tested. Raw files, previous checkpoints and frozen assignments are preserved; no messages, commit or push.
"""
    (folder / "summary_for_will.md").write_text(summary)
    common_cols = [
        "task",
        "model",
        "responses",
        "participants",
        "auroc",
        "auroc_low",
        "auroc_high",
        "ap",
        "prevalence",
        "balanced_accuracy",
        "brier",
        "equal_fold_auroc",
    ]

    def detail(title, body):
        return f"<details><summary>{html.escape(title)}</summary>{body}</details>"

    sections = [
        (
            "The problem is not solved by more columns",
            "<p>Notebook 16 tested longer windows, compact rebalancing, missingness controls and richer representations. No reliable physiological increment emerged. Notebook 17 tests the interpretation of temporal information and a small fixed set of earlier-recording references. It cannot prove why all previous models were weak.</p>"
            + image(
                "03_pooling",
                "Read-only Notebook 16 recomputation. Within-fold and cross-fold pairs are distinct; no held-out calibration or probability normalization.",
            )
            + detail("All prior-model ranking decompositions", table(pool)),
        ),
        (
            "Labels, recall and measurement validity",
            image(
                "01_labels",
                "Observed development distributions. Coping/control shown after reverse coding; no target imputation.",
            )
            + "<p>Stored items are 1–5; score=nervous+(6−cope)+(6−control). Morning recalls since waking; Midday/Endday since the last prompt. Alpha and within-person variance describe these responses, not verified wearable predictability.</p>"
            + table(consistency)
            + detail("Within-/between-person variation", table(audit("variation")))
            + '<p>See <a href="data_quality_audit.md">data_quality_audit.md</a> for exact local questionnaire/protocol references and unverified acquisition/receipt semantics.</p>',
        ),
        (
            "Which records were available?",
            image(
                "02_timeline",
                "Synthetic timeline only. Minute start 09:22 → cutoff 09:21; current wearable interval [08:21,09:21). Known availability must precede the cutoff.",
            )
            + '<div class="callout">The real export has starts, but no submission/receipt clock. We did not substitute an arbitrary delay or assume an earlier start means an earlier available answer.</div>'
            + table(long)
            + detail(
                "Retrospective score transitions — not predictions",
                table(audit("transitions_retrospective")),
            ),
        ),
        (
            "Personalized wearable inputs and recording quality",
            image(
                "06_availability",
                "Availability counts describe observed support, not complete daily recordings.",
            )
            + "<p>The original earlier-recording reference excludes the current hour. The new UTC-clock reference uses matching clock phase over the prior seven days, requiring 30 observed minutes on 3 dates. It is not a validated circadian/resting baseline. Rolling 24-hour step-rate changes require supported intervals in both blocks; no complete-day totals are invented.</p>"
            + detail("Exact feature status", table(audit("feature_status")))
            + detail("Coverage, gaps and freshness", table(load("coverage")))
            + detail(
                "Context/activity-adjusted associations (retrospective)",
                table(load("associations")),
            )
            + detail(
                "Successive pre-EMA window changes — descriptive only",
                table(load("successive_window_changes")),
            )
            + "<p>The successive-window coverage audit was added after the fixed model fits. It compares recorded fixed-hour summaries, not complete interresponse or prompt-recall intervals; it was not used to select predictors or models.</p>",
        ),
        (
            "Two evaluation tasks, one evidence boundary",
            "<p><strong>Task A:</strong> original unseen-participant folds, with earlier unlabeled wearables allowed. <strong>Task A online history:</strong> distinct setting, blocked until prior answers are known to be available. <strong>Task B:</strong> at least five verified available calibration answers, then chronological prediction; population parameters must also use only information available before fitting.</p><p>The available schema gives potential sequence counts but no verified Task B eligibility. The two tasks are never pooled. A two-point increase is a specified exploratory change target, not a clinical threshold.</p>"
            + detail("Retained/excluded composition", table(load("composition")))
            + detail("Exact training/validation support", table(load("fold_support"))),
        ),
        (
            "Actual held-out results",
            image(
                "04_models",
                "Identical main evaluation responses: 3,689 EMAs  / 125 participants. Conditional participant-bootstrap intervals; repeated exploration limits interpretation.",
            )
            + table(metrics[metrics.model.isin(names)], common_cols)
            + detail("Every model and cohort", table(metrics, common_cols))
            + detail(
                "Train versus validation; selected settings",
                table(load("training_metrics")) + table(load("selected")),
            ),
        ),
        (
            "Does physiology add information?",
            image(
                "05_increment",
                "Primary paired contrasts on identical responses. Positive means higher AUROC; every primary interval includes zero.",
            )
            + table(
                primary[["task", "model", "reference", "auroc", "auroc_low", "auroc_high"]].rename(
                    columns={"auroc": "ΔAUROC"}
                )
            )
            + "<p>No established physiological increment in the executed comparisons. Its increment beyond previous EMA answers remains <strong>untested</strong>, not negative: those models need availability evidence.</p>"
            + detail(
                "Requested final comparison, including blocked rows",
                table(
                    comparison,
                    [
                        "task",
                        "label",
                        "status",
                        "responses",
                        "participants",
                        "auroc",
                        "auroc_low",
                        "auroc_high",
                        "ap",
                        "balanced_accuracy",
                    ],
                ),
            )
            + detail("All matched contrasts", table(paired)),
        ),
        (
            "Calibration, strata and scientific controls",
            image(
                "07_calibration",
                "Bins with fewer than five responses omitted from this plot only. No held-out calibration fitted; Brier scores accompany AUROC.",
            )
            + detail("Fold ranking and probability shifts", table(load("folds")))
            + detail("Survey/support strata", table(load("strata")))
            + detail(
                "Participant-level performance: aggregate estimability",
                table(load("participant_summary")),
            )
            + "<p>Individual participant metrics stay private. Two fixed training-label-permutation controls are computational checks, not a formal temporal permutation test. Missingness-only and quality models are reported separately; median imputation can still reveal recording patterns implicitly.</p>",
        ),
        (
            "What to do next",
            '<div class="callout">Resolve answer availability before claiming history-based prediction. Resolve acquisition and prompt timing before claiming live physiological prediction.</div><p>Request exact EMA version/submission/receipt clocks, complete prompt logs, collector/SDK timestamp code and device quality metadata. The focused request is drafted in data_quality_audit.md; nothing was sent. Then test whether physiology adds to verified EMA history on identical responses.</p>'
            + detail("Verification", table(load("verification")))
            + "<p>Reproduce with README.md. All models and participant outputs are ignored/private; public files contain aggregates and synthetic examples only. No final-test evaluation, raw modification, commit or push.</p>",
        ),
    ]
    css = """*{box-sizing:border-box}body{margin:0;background:#eef3f5;color:#173e4c;font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header,main,nav{max-width:1160px;margin:auto}header{padding:65px 30px 30px}h1{font-family:Georgia,serif;font-size:50px;line-height:1.1;margin:15px 0}h2{font-family:Georgia,serif;font-size:32px;line-height:1.25}h3{font-size:20px}.eyebrow{color:#177e89;letter-spacing:.15em;font-size:12px;font-weight:800;text-transform:uppercase}.sub{font-size:22px;max-width:850px}.chips{display:flex;gap:12px;flex-wrap:wrap}.chips span{background:#fff;padding:10px 16px;border:1px solid #d5e1e5;border-radius:8px}nav{display:flex;flex-wrap:wrap;gap:12px;padding:20px 30px}a{color:#177e89}nav a{font-size:14px;text-decoration:none}main{padding:0 20px 60px}section{background:white;border:1px solid #d6e3e8;border-radius:18px;padding:35px;margin:20px 0}figure{margin:28px 0}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#597987}table{border-collapse:collapse;min-width:600px;width:100%;font-size:12px}th{text-align:left;background:#e8f1f3}td,th{padding:10px;border-bottom:1px solid #d5e1e5;vertical-align:top}.table{overflow-x:auto;margin:18px 0}details{border:1px solid #d5e1e5;padding:14px;border-radius:8px;margin:12px 0}summary{cursor:pointer;font-weight:700}.callout{padding:20px;background:#e5f2f0;border-left:5px solid #177e89;font-weight:600}footer{padding:25px;font-size:13px}@media(max-width:600px){h1{font-size:35px}h2{font-size:26px}header{padding:30px 20px}.sub{font-size:19px}section{padding:20px}main{padding:0 10px}body{font-size:16px}}@media print{nav{display:none}section{break-inside:avoid}details{display:block}}"""
    body = "".join(
        f'<section id="s{i}"><div class="eyebrow">{i:02d} / Evidence & findings</div><h2>{html.escape(title)}</h2>{text}</section>'
        for i, (title, text) in enumerate(sections, 1)
    )
    nav = "".join(
        f'<a href="#s{i}">{i}. {html.escape(title)}</a>' for i, (title, _) in enumerate(sections, 1)
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Garmin + EMA · Temporal personalization</title><style>{css}</style></head><body><header><div class="eyebrow">Notebook 17 · Exploratory development research</div><h1>Can temporal history recover the missing signal?</h1><p class="sub">Earlier wearable references did not establish a prediction gain. EMA history remains a promising question—not a verified predictive input.</p><div class="chips"><span>3,706 responses accounted for</span><span>125 development participants</span><span>32 final-test participants sealed</span><span>0 verified online-history rows</span></div></header><nav>{nav}</nav><main>{body}</main><footer>Immutable run {run.name}. Recorded-clock analysis, not validated real-time prediction. Public aggregates only.</footer></body></html>"""
    (folder / "index.html").write_text(page)
    nb = nbformat.v4.new_notebook(
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            garmin_public_aggregate_notebook=True,
        )
    )
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "#17 · Temporal alignment and personalized stress\n\nDevelopment-only, exploratory. Earlier wearable references are distinct from prior EMA-answer history. This notebook reads verified aggregate outputs; it never displays participant rows or real clocks. Final test remains sealed."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nroot=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/temporal_personalization.json').exists())\nreport=root/'reports/temporal_personalization'\npublic=report/'"
            + run.name
            + "'\ndef table(name,audit=False):\n    data=json.loads(((report/'audit' if audit else public)/(name+'.json')).read_text())\n    display(pd.DataFrame([{'measure':k,'value':v} for k,v in data.items()] if isinstance(data,dict) else data))\ndef figure(name): display(Image(filename=str(public/'figures'/(name+'.png'))))"
        ),
    ]
    teaching = [
        (
            "Outcome and measurement evidence",
            "Synthetic items 4,2,3 give 11; missing any item gives no outcome. Alpha is not a reliability certificate.",
            'figure("01_labels")\ntable("consistency",True)\ntable("variation",True)',
        ),
        (
            "Past availability is a separate clock",
            "Synthetic prior start07:00/receipt07:05 can precede09:21. Receipt09:25 cannot. Unknown receipt stays unknown.",
            'figure("02_timeline")\ntable("longitudinal",True)\ndisplay(Markdown((report/"data_quality_audit.md").read_text()))',
        ),
        (
            "Why pooled AUROC can mislead",
            "Cross-fold pairs compare scores produced by different fitted models; within-fold comparisons do not. No validation normalization is used.",
            'figure("03_pooling")\ntable("prior_pooling",True)',
        ),
        (
            "Wearable references and quality",
            "[cutoff−25h,cutoff−1h) is the earlier reference, not resting HR. New UTC phase matching is not verified local circadian alignment. Empty periods remain missing.",
            'figure("06_availability")\ntable("availability")\ntable("coverage")',
        ),
        (
            "Distinct tasks and cohorts",
            "Task A uses unseen participants. Online-history and chronological Task B require verified answer availability; the real-data gate admits zero. Synthetic tests check chronology without inventing real receipt times.",
            'table("comparison")\ntable("fold_support")',
        ),
        (
            "Nested benchmark results",
            "Fixed logistic and small HGB grids. Training-only transforms; original outer folds; no optimal-window claim.",
            'figure("04_models")\nfigure("05_increment")\ntable("paired")',
        ),
        (
            "Calibration and diagnostic controls",
            "Threshold .5 is fixed. Higher recall is not higher AUROC. Participant metrics are summarized only where both classes are observed.",
            'figure("07_calibration")\ntable("participant_summary")\ntable("strata")\ntable("associations")',
        ),
        (
            "Findings and next evidence",
            "Physiology beyond EMA history is untested, not disproven. Obtain exact submission/receipt/version records before that comparison.",
            'display(Markdown((report/"summary_for_will.md").read_text()))\ntable("verification")',
        ),
    ]
    for title, prose, code in teaching:
        nb.cells.extend(
            [
                nbformat.v4.new_markdown_cell("## " + title + "\n\n" + prose),
                nbformat.v4.new_code_cell(code),
            ]
        )
    nbformat.write(nb, root / "notebooks/17_temporal_personalized_stress.ipynb")
    write_json(public / "presentation.json", dict(figures=7, sections=9, code_cells=9))
    return folder


if __name__ == "__main__":
    print(report(Path.cwd()))
