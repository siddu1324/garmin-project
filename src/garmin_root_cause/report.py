"""Aggregate-only teaching notebook and self-contained presentation."""

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
from .preservation import completed

LABELS = {
    "target_nervous": "Nervousness ≥3",
    "target_cope_difficulty": "Coping difficulty ≥3",
    "target_control_difficulty": "Control difficulty ≥3",
    "target_composite5": "Composite ≥5",
    "target_composite10": "Composite ≥10",
    "target": "Counter-defined activity",
}


def polish(text):
    replacements = {
        "all3,706": "all 3,706",
        "All3,706": "All 3,706",
        "all2,772": "all 2,772",
        "All2,772": "All 2,772",
        "same1,355": "same 1,355",
        "all60": "all 60",
        "mean70": "mean 70",
        "SD10": "SD 10",
        "latest80": "latest 80",
        "slope10": "slope 10",
        "all1,355": "all 1,355",
        "per10bpm": "per 10 bpm",
        "per10": "per 10",
        "only5": "only 5",
        "Only5": "Only 5",
        "versus591": "versus 591",
        "Composite unchanged": "Composite unchanged",
        "Notebooks00": "Notebooks 00",
        "Notebook08": "Notebook 08",
        "Notebook18": "Notebook 18",
        "Notebook07": "Notebook 07",
        "matched60": "matched 60",
        "fixed60": "fixed 60",
        "-versus60": "-versus-60",
        "fixed.5": "fixed .5",
        "AUROC.5": "AUROC .5",
        "stored1": "stored 1",
        "Stored1": "Stored 1",
        "displayed0": "displayed 0",
        "five1": "five 1",
        "all32": "all 32",
        "and32": "and 32",
        "code≥3": "code ≥3",
        "same60": "same 60",
        "about0": "about 0",
        "declared20": "declared 20",
        "limited20": "limited 20",
        "retain36.6": "retain 36.6",
        "cohort:51.2": "cohort: 51.2",
        "replicates0": "replicates 0",
        "reproduce0": "reproduce 0",
        "is0.1354": "is 0.1354",
        "versus0.1352": "versus 0.1352",
        "and0.1348": "and 0.1348",
        "has335": "has 335",
        "versus1,020": "versus 1,020",
        "Nominal836": "Nominal 836",
        "later-minute826": "later-minute 826",
        "remain.5": "remain .5",
        "No new": "No new",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def table(rows, columns=None):
    f = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if columns is not None:
        f = f[columns]
    return (
        '<div class="table">'
        + f.to_html(
            index=False, border=0, float_format=lambda x: f"{x:,.3f}", na_rep="—", escape=True
        )
        + "</div>"
    )


def build(root):
    root = Path(root)
    if completed(root):
        return root / "reports/root_cause_analysis"
    ptr = json.loads((root / "private/root_cause_analysis/current.json").read_text())
    run = root / ptr["run"]
    data = root / ptr["data"]
    base = root / "reports/root_cause_analysis"
    out = base / run.name

    def read(n):
        return json.loads((out / f"{n}.json").read_text())

    metrics = pd.DataFrame(read("metrics"))
    paired = pd.DataFrame(read("paired"))
    assoc = pd.DataFrame(read("associations"))
    dist = pd.DataFrame(read("distributions"))
    variance = pd.DataFrame(read("variation"))
    folds = pd.DataFrame(read("folds"))
    probe = read("probe_overview")
    physical = read("physical_consistency")
    lag = read("lag")
    integrity = json.loads((data / "integrity_audit.json").read_text())
    verify = read("verification")
    (out / "integrity_audit.json").write_text(json.dumps(integrity, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f7f8fa",
            "axes.facecolor": "#f7f8fa",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#26394b",
            "text.color": "#173047",
        }
    )
    colors = ["#0b7285", "#d97732", "#4c5b85", "#8e6c88"]
    figures = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        figures.append(name)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    for y, start, end, label in [
        (2, 0, 60, "Fixed 60-minute sensor interval"),
        (1, -30, 60, "Verified prior prompt → cutoff (currently blocked)"),
        (0, 25, 60, "Current-survey waking → cutoff (retrospective only)"),
    ]:
        ax.plot([start, end], [y, y], lw=10, color=colors[int(y)], solid_capstyle="butt")
        ax.text(-35, y + 0.25, label, fontsize=10)
    ax.axvline(60, color="#b33d40", ls="--")
    ax.axvline(61, color="#8994a2", ls=":")
    ax.set(
        ylim=(-0.4, 2.8),
        yticks=[],
        xlim=(-40, 85),
        xticks=[-30, 0, 25, 60],
        xticklabels=["07:51", "08:21", "08:46", "09:21\ncutoff"],
        title="Synthetic timeline • event meaning matters as much as clock arithmetic",
    )
    ax.annotate(
        "09:22 start",
        xy=(61, 2.4),
        xytext=(67, 2.55),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#8994a2"},
    )
    save(fig, "synthetic_timeline")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, name, title in zip(
        axes,
        ["ordinal_nervous", "ordinal_cope_difficulty", "ordinal_control_difficulty"],
        ["Nervousness", "Reversed coping", "Reversed control"],
    ):
        z = dist[(dist.measure == name) & (dist.cohort == "all_development")]
        ax.bar(z.value, z.responses, color=colors[0])
        ax.set(title=title, xlabel="Stored / oriented item code", xticks=range(1, 6))
        ax.set_ylabel("Development responses")
    save(fig, "item_distributions")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].bar(
        ["Candidate bins", "Support + class eligible"],
        [probe["candidate_unique_intervals"], probe["eligible_intervals"]],
        color=colors[:2],
    )
    axes[0].set(title="Fixed activity-probe attrition", ylabel="Unique five-minute bins")
    pm = metrics[(metrics.task == "target") & (metrics.model == "hr")].iloc[0]
    axes[1].bar(
        ["Zero-change / inactive", "≥20 steps/min / active"],
        [pm.negative_support, pm.positive_support],
        color=colors[:2],
    )
    for i, v in enumerate([pm.negative_support, pm.positive_support]):
        axes[1].text(i, v + 8, str(int(v)), ha="center")
    axes[1].set(title="Only five inactive intervals", ylabel="Eligible bins")
    axes[1].set_ylim(0, 660)
    save(fig, "activity_probe")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    z = paired[
        (paired.left == "context_hr")
        & (paired.right == "context")
        & paired.task.str.startswith("target_")
    ].copy()
    y = np.arange(len(z))
    point = z.difference_auroc.to_numpy()
    lo = z.difference_auroc_low.to_numpy()
    hi = z.difference_auroc_high.to_numpy()
    ax.errorbar(point, y, xerr=[point - lo, hi - point], fmt="o", color=colors[0], capsize=4)
    ax.axvline(0, color="#8f9aa6", ls="--")
    ax.set(
        yticks=y,
        yticklabels=[LABELS[x] for x in z.task],
        xlabel="Δ pooled AUROC: context + HR minus context",
        title="No established HR increment on identical responses",
    )
    save(fig, "physiological_increment")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    z = assoc[assoc.association == "within_person_adjusted_rank"]
    p = z.correlation.to_numpy()
    lo = z.lower.to_numpy()
    hi = z.upper.to_numpy()
    y = np.arange(len(z))
    ax.errorbar(p, y, xerr=[p - lo, hi - p], fmt="o", capsize=4, color=colors[1])
    ax.axvline(0, color="#8f9aa6", ls="--")
    ax.set(
        yticks=y,
        yticklabels=["Composite", "Nervousness", "Coping difficulty", "Control difficulty"],
        xlabel="Adjusted within-person rank association (descriptive)",
        title="A small nervousness association does not establish prediction",
    )
    save(fig, "item_associations")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    z = variance[(variance.measure == "pss_combo")]
    axes[0].bar(
        ["All valid", "Matched observed"],
        z.within_fraction * 100,
        label="Within person",
        color=colors[0],
    )
    axes[0].bar(
        ["All valid", "Matched observed"],
        z.between_fraction * 100,
        bottom=z.within_fraction * 100,
        label="Between people",
        color=colors[2],
    )
    axes[0].legend(frameon=False)
    axes[0].set(
        ylabel="Share of score sums of squares (%)", title="Variation is not explained variation"
    )
    z = folds[(folds.task == "target_nervous") & (folds.model.isin(["context", "context_hr"]))]
    for model, col in zip(["context", "context_hr"], colors):
        s = z[z.model == model]
        axes[1].plot(s.fold, s.auroc, "o-", label=model, color=col)
    axes[1].axhline(0.5, color="#8f9aa6", ls="--")
    axes[1].set(
        xlabel="Frozen outer fold",
        ylabel="AUROC",
        title="Nervousness: fold stability",
        xticks=range(5),
    )
    axes[1].legend(frameon=False)
    save(fig, "variation_folds")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    z = metrics[(metrics.task == "ordinal_nervous")]
    axes[0].bar(z.model, z.ranked_probability_score, color=colors[0])
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(
        title="Ordinal nervousness • lower is better",
        ylabel="Ranked probability score",
        ylim=(0, 0.16),
    )
    calibration = pd.DataFrame(read("calibration"))
    z = calibration[(calibration.task == "target_nervous") & (calibration.model == "context_hr")]
    axes[1].plot([0, 1], [0, 1], ls="--", color="#8f9aa6")
    axes[1].scatter(
        z.mean_probability, z.positive_fraction, s=20 + z.responses / 3, color=colors[1]
    )
    axes[1].set(
        xlabel="Mean predicted probability",
        ylabel="Observed elevated fraction",
        title="Nervousness calibration • fixed deciles",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    save(fig, "ordinal_calibration")

    def image(name, alt):
        encoded = base64.b64encode((out / f"{name}.png").read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'

    nerv = metrics[(metrics.task == "target_nervous") & (metrics.model == "context_hr")].iloc[0]
    inc = paired[
        (paired.task == "target_nervous")
        & (paired.left == "context_hr")
        & (paired.right == "context")
    ].iloc[0]
    main = metrics[
        (metrics.kind == "binary")
        & (metrics.task != "target")
        & (metrics.model.isin(["context", "context_hr", "context_hr_steps"]))
    ].copy()
    main["task"] = main.task.map(LABELS)
    take = [
        "task",
        "model",
        "auroc",
        "ap",
        "prevalence",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "brier",
    ]
    sections = []

    def section(title, body):
        sections.append((title, polish(body)))

    section(
        "01 · What did the audit resolve?",
        f"""<p class="lead">No new confirmed implementation defect was found. The evidence does not identify a single root cause of weak prediction.</p>
<div class="cards"><article><b>3,706</b><span>Development responses accounted for</span></article><article><b>1,355 / 113</b><span>Matched responses / participants</span></article><article><b>32 sealed</b><span>Final-test participants untouched</span></article></div>
<p>The original export → recorded clocks → identity → acceptance → windows → summaries → model inputs → saved predictions chain reproduces in bounded checks. That verifies implementation under the recorded-clock interpretation, not device acquisition semantics.</p>
{table([dict(check=k, result=v) for k, v in verify.items() if isinstance(v, (int, bool))])}
<details><summary>Original-source evidence and rule classification</summary>{table([dict(check=k, result=str(v)) for k, v in integrity.items()])}<p>See evidence_classification.md for confirmed historical defects, verified current behavior, unresolved assumptions and plausible scientific explanations. The old 12/24-hour overflow remains superseded; current bundles use corrected boundaries.</p></details>""",
    )
    section(
        "02 · The same timestamp can mean different things",
        image(
            "synthetic_timeline",
            "Synthetic clocks only. The pre-EMA cutoff is not a verified acquisition clock or a recall boundary.",
        )
        + """<div class="callout">Minute start 09:22 → cutoff 09:21. The immediate five-minute window is [09:16,09:21). Second start 09:22:35 → [09:17:35,09:22:35). No reading at cutoff enters.</div>
<p>Garmin Unix ts is preserved without an extra timezone shift. Missing collector/API/version and payload evidence leave acquisition, callback, receipt and batching unresolved. Repeated integer HR values can be real stability or stale reports; source quality flags are absent.</p>
<p><b>Synthetic feature calculation:</b> readings 60, 70, 80 bpm at elapsed 0, 60, 120 seconds give mean70, sample SD10, latest80 and slope10 bpm/min. These summarize three observed readings, not every moment. A counter pair crossing the left boundary is excluded; a reset breaks adjacency rather than becoming negative activity.</p>""",
    )
    section(
        "03 · Can HR recognize counter-defined activity?",
        image(
            "activity_probe",
            "The frozen support rules leave an unbalanced physical-consistency probe.",
        )
        + f"""<p>{probe["candidate_unique_intervals"]:,} nonoverlapping candidate bins → {probe["eligible_intervals"]} eligible bins from {probe["eligible_participants"]} people. HR occupies ≥4/5 minutes, largest HR gap≤90s; wholly contained accepted counter pairs support≥240s with unsupported gap≤60s. Active≥20 steps per supported minute; inactive=zero additions. No activity input enters the HR-only model.</p>
<div class="callout">HR-only AUROC {pm.auroc:.3f} [{pm.auroc_low:.3f}, {pm.auroc_high:.3f}]; balanced accuracy {pm.balanced_accuracy:.3f}. All 596 intervals are predicted active at .5. Only four people exhibit both labels. This is inconclusive, not sensor validation.</div>
<p>Equal-person active-minus-inactive HR difference {physical["mean_within_person_active_minus_inactive_bpm"]:.1f} bpm [{physical["lower"]:.1f}, {physical["upper"]:.1f}] rests on only four people. Large AUPRC ({pm.ap:.3f}) mainly reflects {pm.prevalence:.1%} active prevalence. Within-fold pair-weighted AUROC {pm.within_fold_pair_weighted_auroc:.3f} excludes single-class folds; its limited denominator cannot resolve the uncertainty. No threshold was relaxed.</p>
<details><summary>Support, exclusions, physiological values and fold denominators</summary>{table(read("probe_exclusions"))}{table(read("probe_distributions"))}{table(folds[folds.task == "target"], ["model", "fold", "responses", "participants", "positive_support", "negative_support", "auroc"])}</details>""",
    )
    section(
        "04 · Does the composite hide different items?",
        image(
            "item_distributions",
            "Stored item codes 1–5; coping/control are reverse-oriented only for like-direction analysis.",
        )
        + """<div class="formula">PSS = nervous + (6 − cope) + (6 − control) ∈ [3,15]</div>
<p>Synthetic nervous4, cope2, control3 → 4+4+3=11. Missing/invalid items leave the composite missing. Individual items are also described in their original direction. The binary midpoint contrasts (oriented code≥3) are fixed exploratory thresholds, not clinical diagnoses.</p>
<p>Primary new question: elevated nervousness, context+HR versus context. Secondary: coping, control and unchanged composite≥5/≥10. Separate proportional-odds models preserve five ordered item levels; fixed Ridge preserves the continuous composite. No winning target is selected.</p>
<details><summary>All label distributions, participant class mix, survey differences and exclusions</summary>"""
        + table(read("class_mix"))
        + table(read("composition"))
        + table(read("survey"))
        + """</details>""",
    )
    section(
        "05 · Association is not unseen-person prediction",
        image(
            "item_associations",
            "Participant bootstrap, 500 draws; multiple descriptive analyses remain exploratory.",
        )
        + f"""<p>The adjusted within-person rank association is about +0.071 for nervousness (interval +0.003 to +0.135), versus −0.006 for the composite. This small isolated interval does not establish a new predictor or prove the composite is defective. Coping/control concern different experiences; opposite directions can coexist.</p>
<p>Adjustment includes survey/time, observed steps, occupancy, freshness and gaps. Retrospective person means never enter predictive features. A composite random-intercept model gives a within-person HR effect of {read("mixed_effects")["coefficients"][1]["estimate"]:.3f} composite points per10bpm (model SE {read("mixed_effects")["coefficients"][1]["model_se"]:.3f}); Gaussian assumptions and model-based SE limit interpretation.</p>
{image("variation_folds", "53.5% within-person score variation does not identify wearable-explainable variation.")}
<p>All valid development scores: 53.5% within-person /46.5% between-person sums of squares; matched cohort:51.2%/48.8%. Variation may reflect unmeasured events, recalled experience or measurement error as well as physiology. No information-theoretic ceiling is identified. Retrospective lag-one correlation reproduces {lag["pooled"]:.3f}, person-centered {lag["person_centered"]:.3f}, from {lag["pairs"]:,} pairs. Zero verified online-history rows.</p>
<details><summary>All associations and random-intercept model</summary>{table(assoc)}{table(read("mixed_effects")["coefficients"])}</details>""",
    )
    section(
        "06 · Controlled held-out comparisons",
        f"""<p class="lead">Nervousness: context+HR AUROC {nerv.auroc:.3f}; HR increment {inc.difference_auroc:+.3f} [{inc.difference_auroc_low:+.3f}, {inc.difference_auroc_high:+.3f}]. No established increment.</p>
<p>Every stress comparator uses the same1,355 observed responses /113 participants, unchanged five participant folds. C∈{{0.1,1}} is selected only in three grouped inner folds. No imputation, weighting, threshold search or quality predictors. Constant probabilities are training prevalences. Final-test evaluation remains sealed.</p>
{image("physiological_increment", "Paired participant-bootstrap intervals from 2,000 fixed-prediction draws; not independent confirmation.")}
{table(main, take)}
<p>Neither ordinal nervousness nor the unchanged continuous composite establishes an HR gain. Regularization/model family is bounded here; no further search follows these results. Zero sensitivity at .5 for some low-prevalence targets is reported, not repaired using validation outcomes.</p>
{image("ordinal_calibration", "Ordinal comparisons retain ordering; calibration plot is descriptive and never used to refit predictions.")}
<details><summary>Full metrics, train/validation support and paired comparisons</summary>{table(metrics)}{table(read("training"))}{table(read("supports"))}{table(paired)}</details>""",
    )
    section(
        "07 · Recall alignment is currently an evidence gate",
        """<p>All three Morning items say <b>since waking up</b>; Midday and Endday say <b>since the last prompt</b>. Endday is not automatically the whole day. A fixed60-minute interval can include time outside the recalled period or omit part of it.</p>
<p>Morning:934 responses;836 nominal retrospective wake candidates,826 with the later-minute boundary. These use a current answer and provisional same-date/offset assumptions. Midday+Endday:2,772 responses lack the required complete delivered-prompt history. All3,706 lack a verified acquisition contract and exact answer-version availability clock.</p>
<p>Notebook08 already compared retrospective wake and fixed windows on matched responses. Those results remain explicitly retrospective; Notebook18 neither relabels them prospective nor repeats a performance search. No newly verified boundary exists, so the new aligned extraction gate opens for zero real responses.</p>
<div class="callout">Synthetic valid case: delivered prompt07:00, receipt07:00:02, verified event-clock contract, cutoff09:21 → [07:00,09:21). Unknown delivery, current-answer waking or availability at/after cutoff → blocked.</div>
<details><summary>Exact affected counts and gate reasons</summary>"""
        + table(read("missing_evidence_counts"))
        + table(read("recall_availability"))
        + """</details>""",
    )
    section(
        "08 · What remains computable?",
        table(
            [
                dict(
                    feature="Recorded HR summaries",
                    status="Computable; timing/quality provisional",
                    next="Keep observed support separate",
                ),
                dict(
                    feature="Supported counter additions",
                    status="Computable under provisional counter interpretation",
                    next="Request unchanged/reporting/reset contract",
                ),
                dict(
                    feature="Earlier wearable references",
                    status="Computable, no established prior gain",
                    next="Do not call them resting HR",
                ),
                dict(
                    feature="Retrospective wake windows",
                    status="Descriptive/retrospective only",
                    next="Preserve Notebook08 evidence",
                ),
                dict(
                    feature="Prospective recall-aligned features",
                    status="Blocked",
                    next="Verified boundary, availability and acquisition clocks",
                ),
                dict(
                    feature="EMA-history features",
                    status="Blocked",
                    next="Submission/receipt/version clocks",
                ),
                dict(
                    feature="IBI HRV / validated PulseOx",
                    status="Blocked / provisional",
                    next="Quality and event semantics; no new calculation",
                ),
                dict(
                    feature="Garmin stress / VLF",
                    status="Excluded / deferred",
                    next="No initial predictor use",
                ),
            ]
        )
        + """<p>Previously tested HR/activity prediction and the limited20-run-hypothesis IBI variability pilot answer different questions. Numerical IBI package agreement does not establish normal-beat status or HRV predictability.</p>
<p>Complete-case selection retains36.6% of all responses; feature support and survey representation differ among excluded responses. Dense sampled periods coexist with absent windows. Conditional bootstrap intervals do not include upstream measurement uncertainty, nested-selection variation or the cost of repeated development exploration.</p>
<details><summary>Recorded values, gaps, availability and score selection</summary>"""
        + table([dict(measure=k, value=str(v)) for k, v in read("sensor_summary").items()])
        + table(read("sensor_availability_scores"))
        + table(read("quality_summary"))
        + """</details>""",
    )
    section(
        "09 · Decision for Will and Orson",
        """<p class="lead">Prioritize measurement evidence and an independently anchored validation protocol before another prediction experiment.</p>
<ol><li>Request collector timestamp/batch code and one original payload-to-row trace.</li><li>Clarify counter emission behavior and obtain enough independently supported inactive and active periods.</li><li>Obtain complete prompt logs and answer-version receipt/submission clocks; separately observed waking if prospective Morning alignment is required.</li><li>Only after those gates pass, freeze a small matched aligned-versus-fixed comparison and an independently measured activity positive control.</li></ol>
<p>No automatic model expansion, altered target threshold, imputed raw data, HRV, final-test evaluation, commit, push or provider message. The evidence supports a small descriptive nervousness lead, not a claim that changing targets solves stress prediction.</p>
<p>Files: specification.md; evidence_classification.md; missing_evidence_request.md; summary_for_will.md; versioned aggregate tables and figures; fully executed Notebook18. Reproduce with the commands in README.md. Individual outputs remain ignored/private.</p>""",
    )
    css = """*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef2f5;color:#1d3348;font:17px/1.6 system-ui,sans-serif}header,main,nav{max-width:1140px;margin:auto}header{padding:70px 36px 40px}h1{font-size:clamp(32px,5vw,58px);line-height:1.12;letter-spacing:-.04em;max-width:950px}h2{font-size:28px;line-height:1.25}header .tag{color:#0b7285;font-weight:700;letter-spacing:.13em;font-size:13px}nav{display:flex;flex-wrap:wrap;gap:8px;padding:0 36px 30px}nav a{color:#0b7285;background:white;border-radius:20px;padding:5px 14px;text-decoration:none;font-size:14px}main{padding:0 20px 55px}section{background:white;padding:32px;margin:0 0 24px;border-radius:18px;box-shadow:0 3px 18px #17304708;scroll-margin-top:16px}p{max-width:100ch}.lead{font-size:22px;line-height:1.5;font-weight:600}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.cards article{background:#eaf5f5;padding:20px;border-radius:12px}.cards b{display:block;font-size:30px;color:#0b7285}.cards span{font-size:14px}.callout,.formula{background:#edf6f7;padding:18px 22px;border-left:4px solid #0b7285;border-radius:6px;margin:22px 0}.formula{font-size:24px;text-align:center}figure{margin:24px 0}img{max-width:100%;height:auto;border-radius:10px}figcaption{font-size:13px;color:#5c6f80}details{margin:20px 0;border:1px solid #dce4eb;border-radius:8px;padding:12px}summary{cursor:pointer;font-weight:600}.table{overflow-x:auto;max-width:100%;margin:16px 0}table{border-collapse:collapse;font-size:13px;line-height:1.45;min-width:100%;width:max-content}td,th{padding:10px 12px;text-align:left;border-bottom:1px solid #e5eaef;vertical-align:top;min-width:110px;max-width:420px;overflow-wrap:break-word}th{background:#eef4f6;white-space:nowrap}a{color:#0b7285}footer{padding:20px;text-align:center;color:#627285;font-size:13px}@media(max-width:600px){header{padding:35px 22px 20px}section{padding:20px}main{padding:0 10px 25px}.cards{grid-template-columns:1fr}nav{padding:0 20px 20px}.lead{font-size:19px}h2{font-size:24px}.formula{font-size:18px}}@media print{body{background:white}section{box-shadow:none;break-inside:avoid}nav,details{display:none}header{padding-top:20px}}"""
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Garmin + EMA · Root-cause diagnostics</title><style>'
        + css
        + '</style></head><body><header><p class="tag">NOTEBOOK 18 · DEVELOPMENT ONLY · EXPLORATORY</p><h1>Weak prediction: what is—and is not—explained?</h1><p class="lead">An independent audit, a constrained activity probe, and item-level comparisons for Will and Orson.</p><p>Recorded-data analysis. No validated real-time or clinical claims.</p></header><nav>'
        + "".join(
            f'<a href="#s{i}">{i:02} · {html.escape(t.split("·")[1].strip())}</a>'
            for i, (t, b) in enumerate(sections, 1)
        )
        + "</nav><main>"
        + "".join(
            f'<section id="s{i}"><h2>{html.escape(t)}</h2>{b}</section>'
            for i, (t, b) in enumerate(sections, 1)
        )
        + "</main><footer>All figures and results are aggregate. Synthetic clocks only. Original folds and prior artifacts preserved.</footer></body></html>"
    )
    (base / "index.html").write_text(page)
    summary = f"""# Notebook 18 — root-cause diagnostics for Will and Orson

**No new confirmed processing defect or reliable stress-prediction gain was established.**
The audit narrows implementation explanations; missing measurement evidence still prevents a single-root-cause claim.

- **Integrity:** all3,706 development scores/cutoffs checked;10-person original-sensor trace;320 additional eight-lookback HR values and80 counter-rate values;83,927 counter-pair masks;185 train+validation estimator replays and three fresh fits. No current defect required a correction. Prior overflow remains explicitly superseded.
- **Physical consistency:**41,024 unique five-minute candidate bins leave596 eligible bins /90 people:591 active,5 inactive. HR-only AUROC{pm.auroc:.3f} [{pm.auroc_low:.3f},{pm.auroc_high:.3f}], balanced accuracy.500; every bin predicted active. Only four people have both labels. Inconclusive—not a sensor-validation result. Do not relax support or infer zero activity from absent readings.
- **Targets:** all stress comparisons use the same1,355 responses /113 people. Nervousness≥3: context+HR AUROC{nerv.auroc:.3f}; increment over context{inc.difference_auroc:+.3f} [{inc.difference_auroc_low:+.3f},{inc.difference_auroc_high:+.3f}]. No binary target establishes an HR increment. Ordinal nervousness ranked probability score is0.1354 versus0.1352 for context and0.1348 for training-distribution constant (lower is better). Continuous composite remains unchanged.
- **Descriptive lead:** adjusted within-person rank association for nervousness is+0.071 [+.003,+.135], versus−.006 for composite. This small isolated exploratory association does not establish unseen-person predictability, an optimal target, or clinical meaning.
- **Variation:**53.5% within-person score sums of squares overall,51.2% on the matched cohort. Neither is variance explained by wearables. Retrospective lag correlations reproduce0.584 and0.221 after person centering; person means never enter prediction.
- **Recall mismatch:** Morning since waking; Midday/Endday since last prompt. All2,772 Midday/Endday rows lack complete delivered-prompt boundaries;934 Morning rows have only current-answer waking. Nominal836/later-minute826 candidates remain retrospective. All3,706 lack a verified prospective recall boundary and acquisition contract. Tested extraction gates block real aligned features; no guessed logs or current answers used as predictors.

The primary cohort has335 elevated-nervousness versus1,020 lower responses. Prediction sensitivity at the fixed.5 threshold remains poor. Fold-specific train/validation supports, per-target prevalence, AUROC/AUPRC, balanced accuracy, sensitivity/specificity, Brier scores, ordinal/continuous errors and participant-bootstrap paired differences are in the versioned aggregate tables. Pooled OOF ranking differs from within-fold ranking because fits have different calibration; a training-prevalence constant has within-fold AUROC.5 even when pooled AUROC differs.

**Decision:** prioritize timestamp/export provenance, delivered-prompt/answer-version logs and a validated activity positive control with adequate inactive support. If verified metadata arrives, freeze a matched recall-aligned-versus60-minute comparison before fitting. No additional model search is justified solely by these results.

This is a prospectively specified exploratory follow-up after Notebooks00–17, not independent confirmation. The small C grid uses only participant-grouped training splits. All model blocks have identical complete observed responses; no feature/target imputation. Conditional bootstrap intervals omit acquisition uncertainty and repeated-search/model-selection variation. No clinical, causal, real-time or final-test claim.

Outputs: `notebooks/18_root_cause_diagnostics.ipynb`; `reports/root_cause_analysis/index.html`; aggregate tables/figures `{out.relative_to(root)}/`; ignored development audit data `{data.relative_to(root)}/`; ignored predictions/models `{run.relative_to(root)}/`. All prior artifacts, raw files, assignments and32 final-test participants are preserved. No commit, push or provider message.
"""
    (base / "summary_for_will.md").write_text(polish(summary))
    # Notebook loads only public aggregate artifacts. It never displays private tables or clocks.
    nb = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "garmin_public_aggregate_notebook": True,
        }
    )
    cells = [
        nbformat.v4.new_markdown_cell(
            "# 18 · Root-cause diagnostics\n\nDevelopment-only, exploratory. No new confirmed implementation defect or established HR prediction gain. Measurement uncertainty remains unresolved. All participant-level outputs stay private; the final test stays sealed."
        ),
        nbformat.v4.new_code_cell(
            f"from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nROOT = Path.cwd() if (Path.cwd() / 'reports').exists() else Path.cwd().parent\nOUT = ROOT / {str(out.relative_to(root))!r}\ndef aggregate(name):\n    return json.loads((OUT / (name + '.json')).read_text())\ndef show(name, columns=None):\n    frame = pd.DataFrame(aggregate(name))\n    display(frame if columns is None else frame[columns])\ndef figure(name):\n    display(Image(filename=str(OUT / (name + '.png'))))\ndisplay(Markdown('**Pinned aggregate version:** `{run.name}`. 3,706 development responses; 32 final-test participants sealed.'))"
        ),
    ]
    content = [
        (
            "Independent integrity checks",
            "All-row clock/score checks and bounded original-source traces verify recorded-data arithmetic. Acquisition semantics are a separate question.",
            "display(pd.DataFrame([aggregate('verification')]))\ndisplay(pd.DataFrame([aggregate('integrity_audit')['original_sensor_trace']]))",
        ),
        (
            "Synthetic clocks and feature calculation",
            "Minute start09:22 → cutoff09:21 → [09:16,09:21). Second start09:22:35 → [09:17:35,09:22:35). Synthetic HR60/70/80 at0/60/120s: mean70, sampleSD10, latest80, slope10bpm/min. Counter boundary crossings remain excluded.",
            "figure('synthetic_timeline')",
        ),
        (
            "Conditional physical-consistency probe",
            "Only5 inactive versus591 active intervals remain. No activity input enters HR-only logistic. This is inconclusive, not device validation; do not relax the frozen support rules.",
            "figure('activity_probe')\ndisplay(pd.DataFrame([aggregate('probe_overview')]))\nshow('probe_exclusions')\ndisplay(pd.DataFrame(aggregate('metrics')).query(\"task == 'target'\")[[ 'model','responses','participants','positive_support','negative_support','auroc','auroc_low','auroc_high','ap','balanced_accuracy','sensitivity','specificity']])",
        ),
        (
            "Outcome construction and participant variation",
            "Composite unchanged: nervous+(6−cope)+(6−control). Stored1–5, displayed0–4; coping/control are reversed for like-direction contrasts only. Thresholds are exploratory, not diagnoses. Within-person variation is not wearable-explained variation.",
            "figure('item_distributions')\nfigure('variation_folds')\nshow('class_mix')\nshow('composition')\ndisplay(pd.DataFrame([aggregate('lag')]))",
        ),
        (
            "Descriptive item associations",
            "Full-record person means are retrospective diagnostics only. Adjusted nervousness association is small; multiple exploratory intervals do not establish prediction. Random-intercept analysis applies to the continuous composite, not Gaussian item claims.",
            "figure('item_associations')\nshow('associations')\ndisplay(pd.DataFrame(aggregate('mixed_effects')['coefficients']))",
        ),
        (
            "Matched held-out comparisons",
            "Same1,355 observed responses /113 people; five frozen participant folds, small training-only C grid. No imputation or quality predictors. Ordinal models retain five ordered levels; continuous composite uses fixed Ridge. No winning target is selected.",
            "figure('physiological_increment')\nfigure('ordinal_calibration')\nshow('metrics')\nshow('paired')\nshow('supports')",
        ),
        (
            "Recall evidence gate and next decision",
            "Morning since waking; Midday/Endday since last prompt. Current-answer waking remains retrospective. No new verified prospective boundaries exist. Code gates require event meaning, UTC conversion, prompt completeness and availability before cutoff; unknown clocks never become guessed delays.",
            "show('missing_evidence_counts')\nshow('recall_availability')\ndisplay(Markdown((ROOT / 'reports/root_cause_analysis/summary_for_will.md').read_text()))",
        ),
    ]
    for title, body, code in content:
        cells.extend(
            [
                nbformat.v4.new_markdown_cell("## " + title + "\n\n" + polish(body)),
                nbformat.v4.new_code_cell(code),
            ]
        )
    nb.cells = cells
    nbformat.write(nb, root / "notebooks/18_root_cause_diagnostics.ipynb")
    return base


if __name__ == "__main__":
    print(build(Path.cwd()))
