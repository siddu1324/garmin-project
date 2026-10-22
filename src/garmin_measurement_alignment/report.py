"""Aggregate-only teaching notebook and self-contained meeting presentation."""

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


def build(root):
    root = Path(root)
    base = root / "reports/measurement_alignment"
    if (base / "completion.json").exists():
        raise ValueError("Sealed presentation exists; verify it, do not overwrite")
    ptr = json.loads((root / "private/measurement_alignment/current.json").read_text())
    out = root / ptr["public"]

    def read(name):
        return json.loads((out / (name + ".json")).read_text())

    timing = pd.DataFrame(read("timing"))
    cadence = pd.DataFrame(read("cadence"))
    assoc = pd.DataFrame(read("availability_associations"))
    cov = pd.read_csv(out / "coverage.csv")
    sens = pd.read_csv(out / "timing_sensitivity.csv")
    counters = read("counters")
    overview = cov[cov.grouping.eq("sensor/lookback_minutes")]
    hr = overview[overview.sensor.eq("hr")].sort_values("lookback_minutes")
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )

    def save(name):
        plt.savefig(figdir / (name + ".png"), facecolor="white")
        plt.close()

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(timing.survey, timing.delay_median_minutes, color=["#347c80", "#375c91", "#bb7626"])
    ax.set(
        ylabel="Median minutes after scheduled start",
        title="Actual start is observed; notification delivery is not",
    )
    for i, r in timing.iterrows():
        ax.text(i, r.delay_median_minutes + 0.3, f"{r.delay_median_minutes:.2f} min", ha="center")
    ax.set_ylim(0, 12)
    save("start_delays")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(8), hr.any_accepted / 3706 * 100, "o-", color="#347c80")
    axes[0].set(ylabel="Responses with ≥1 accepted HR reading (%)", ylim=(0, 100))
    axes[1].plot(range(8), hr.occupied_fraction_median * 100, "o-", color="#bb7626")
    axes[1].set(ylabel="Median occupied-minute fraction (%)", ylim=(0, 100))
    for ax in axes:
        ax.set_xticks(range(8), [5, 15, 30, 60, 120, 360, 720, 1440], rotation=45)
        ax.set_xlabel("Lookback (minutes)")
    fig.suptitle("Finding some HR within 24 hours is not complete 24-hour coverage")
    fig.tight_layout()
    save("coverage")
    strata = cov[
        cov.grouping.eq("sensor/lookback_minutes/n_survey_category") & cov.sensor.eq("hr")
    ].copy()
    strata["percent"] = 100 * strata.any_accepted / strata.responses
    matrix = strata.pivot(index="n_survey_category", columns="lookback_minutes", values="percent")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu")
    ax.set_xticks(range(8), matrix.columns.astype(int))
    ax.set_yticks(range(3), matrix.index)
    ax.set_xlabel("Lookback (minutes)")
    ax.set_title("HR availability by survey — any accepted reading, not full support")
    for (i, j), v in np.ndenumerate(matrix.to_numpy()):
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center", color="white" if v > 65 else "#163344")
    fig.colorbar(im, ax=ax, label="% responses")
    save("survey_coverage")
    selected = assoc[assoc.sensor.isin(["hr", "steps"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = selected.survey_time_standardized_difference * 100
    lo = selected.survey_time_standardized_difference_low * 100
    hi = selected.survey_time_standardized_difference_high * 100
    ax.errorbar(x, np.arange(len(x)), xerr=[x - lo, hi - x], fmt="o", capsize=4, color="#347c80")
    ax.axvline(0, color="#777", ls="--")
    ax.set_yticks(
        range(len(x)),
        [f"{r.sensor.upper()} · composite ≥{r.threshold}" for r in selected.itertuples()],
    )
    ax.set_xlabel("Availability difference: positive − negative (percentage points)")
    ax.set_title("Selection diagnostic, standardized by survey and local time")
    save("availability")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(4)
    for j, r in cadence.iterrows():
        ax.bar(
            x + (j - 0.5) * 0.35,
            [100 * r[f"gap_gt_{v}_seconds"] / r.adjacent_pairs for v in [2, 60, 300, 3600]],
            width=0.35,
            label=r.sensor.upper(),
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [">2 s", ">60 s", ">5 min", ">1 hour"])
    ax.set_ylabel("% consecutive sorted timestamp pairs (log scale)")
    ax.set_title("Recording gaps are observed; their cause is unverified")
    ax.legend()
    save("cadence")
    fig, ax = plt.subplots(figsize=(9, 4))
    for shift, g in sens.groupby("shift_seconds"):
        ax.plot(
            range(8),
            g.sort_values("lookback_minutes").responses_any_accepted,
            "o-",
            label=f"{shift:+d} seconds",
        )
    ax.set_xticks(range(8), [5, 15, 30, 60, 120, 360, 720, 1440])
    ax.set(
        ylabel="Responses with any accepted HR",
        xlabel="Lookback (minutes)",
        title="Clock sensitivity: support only, no performance-based shift",
    )
    ax.legend()
    save("sensitivity")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hlines(
        [2, 1, 0], [0, 35, 5], [141, 141, 141], colors=["#347c80", "#375c91", "#aaa"], linewidth=8
    )
    ax.scatter(
        [0, 35, 100, 141, 142],
        [2, 1, 0, 1, 1],
        c=["#347c80", "#375c91", "#bb7626", "#b83c40", "#b83c40"],
        s=55,
        zorder=3,
    )
    ax.annotate("Verified wake", (0, 2), xytext=(0, 2.28))
    ax.annotate("Verified delivered prompt", (35, 1), xytext=(30, 1.3))
    ax.annotate("Previous completed EMA ≠ last prompt", (100, 0), xytext=(35, -0.45))
    ax.annotate(
        "Cutoff 09:21; start 09:22\nminute precision → 60 s buffer",
        (141, 1),
        xytext=(95, 2.5),
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_xlim(-5, 155)
    ax.set_ylim(-0.8, 3)
    ax.set_xticks([0, 35, 100], ["07:00", "07:35", "08:40"])
    ax.set_yticks([0, 1, 2], ["Invalid substitution", "Midday / Endday", "Morning"])
    ax.set_title("Synthetic teaching timeline — no participant timestamps")
    save("synthetic_timeline")

    def table(df):
        return (
            '<div class="table">'
            + df.to_html(index=False, border=0, float_format=lambda v: f"{v:,.3f}", na_rep="—")
            + "</div>"
        )

    def image(name, caption):
        return (
            '<figure><img alt="'
            + html.escape(caption)
            + '" src="data:image/png;base64,'
            + base64.b64encode((figdir / (name + ".png")).read_bytes()).decode()
            + '"><figcaption>'
            + caption
            + "</figcaption></figure>"
        )

    sixty = hr[hr.lookback_minutes.eq(60)].iloc[0]
    day = hr[hr.lookback_minutes.eq(1440)].iloc[0]
    sections = []
    sections.append(
        (
            "Available clocks",
            f"""<p>All <b>3,706 development responses / 125 participants</b> have actual and scheduled starts. Morning + Endday: <b>1,868 minute-resolution starts</b>; Midday: <b>1,838 second-resolution starts</b>. Neither field is a verified delivered-prompt or submission clock.</p>{table(timing[["survey", "responses", "minute_resolution", "second_resolution", "delay_median_minutes", "delay_gt_60_minutes"]])}{image("start_delays", "Delay = actual − scheduled start. No missing starts among exported responses; this cannot count unanswered prompts.")}<details><summary>Field audit and source provenance</summary>{table(pd.DataFrame(read("field_audit")).fillna(""))}<p>Exact original filenames are retained privately in exact_source_catalog.json; they are not published because names may identify participants. Normalized columns retain source hash, row, schema and mapping.</p></details>""",
        )
    )
    sections.append(
        (
            "Recall is not a fixed 24 hours",
            f"""<p>All three composite items ask <b>since waking</b> in Morning and <b>since the last prompt</b> in Midday/Endday. Endday does not establish an all-day outcome. Current-survey waking remains retrospective. A previous completed EMA is not a prompt log.</p>{image("synthetic_timeline", "Synthetic times only. Minute start 09:22 → cutoff 09:21 → five-minute [09:16,09:21). Second start 09:22:35 → [09:17:35,09:22:35).")}<p>Local actual start minus its row’s local−UTC offset gives UTC. Garmin Unix ts receives no additional shift. Preserve the buffer and [left, cutoff) boundaries.</p><p><a href="../recall_alignment/824551e40b8e301d/questionnaire_evidence.md">Exact questionnaire PDF/page evidence</a></p>""",
        )
    )
    sections.append(
        (
            "What Garmin recordings establish",
            f"""<p>HR logical records: <b>{int(cadence.loc[cadence.sensor.eq("hr"), "records"].iloc[0]):,}</b>; prior validated accepted timestamp groups: <b>15,613,695</b>. Median participant cadence is approximately 1.05 s for HR and step records. Fractional Unix timestamps are storage detail, not proof of acquisition accuracy.</p>{image("cadence", "Diagnostic gap thresholds are descriptions, not new eligibility rules. Consecutive records across gaps do not create HRV adjacency.")}{table(cadence[["sensor", "records", "timestamp_repeat_pairs", "source_order_reversals", "repeated_value_adjacent", "gap_gt_60_seconds", "gap_gt_3600_seconds"]])}<p>Roughly one-second cadence is consistent with periodic reporting. Most accepted step pairs increase, which is also compatible with change-triggered export. Neither pattern distinguishes acquisition, cached callback values or delayed batches. Zero export-order reversals can reflect exporter sorting.</p><details><summary>Counter evidence and why the activity probe is inconclusive</summary>{table(pd.DataFrame([counters]))}<p>Notebook 18 had 591 supported active versus only five inactive bins, with four participants in both states. Its all-active predictions had balanced accuracy 0.5. This is insufficient ground truth for physiological validation. Counter resets, rejected endpoints and boundary-crossing intervals remain excluded; absence of reports is not inactivity.</p></details>""",
        )
    )
    sections.append(
        (
            "Coverage and selection",
            f"""<p>At 60 minutes, <b>{int(sixty.any_accepted):,}/3,706</b> responses have any accepted HR. At 24 hours, <b>{int(day.any_accepted):,}/3,706</b> do, but median occupied-minute fraction is only <b>{100 * day.occupied_fraction_median:.1f}%</b>. Occupied bins do not imply continuous recording.</p>{image("coverage", "Every response remains in the denominator, including no Garmin and missing outcomes.")}{image("survey_coverage", "Separate aggregates by participant and four-hour local-start strata are saved alongside these figures.")}{image("availability", "Participant bootstrap, 2,000 attempted draws. Sparse strata leave only 672 / 1,067 fully estimable standardized draws at ≥5 / ≥10; intervals are fragile and exploratory.")}<p>Availability means ≥1 accepted HR/PulseOx reading or ≥1 supported counter pair for steps/floors. PulseOx remains provisional. These are selection diagnostics, not predictors or causal effects.</p>""",
        )
    )
    sections.append(
        (
            "Uncertainty without clock repair",
            f"""{image("sensitivity", "Hypothetical −60 / 0 / +60 s shifts; original timestamps remain unchanged. Stored DURING/POST records cannot enter even under −60 s.")}<p>This asymmetric restriction deliberately prevents future-record leakage. It is not an estimated latency distribution or recommended correction. No AUROC is computed and no offset is selected. At 60 minutes, any-HR availability is 1,723 / 1,725 / 1,662 respectively.</p>""",
        )
    )
    sections.append(
        (
            "A tested pipeline, with real gates closed",
            """<p><b>Path A completed: 3,706 recall-aligned rows BLOCKED; 3,706 prior-answer availability gates BLOCKED.</b> Missing evidence is not zero accuracy or proof that physiology cannot work.</p><ol><li>Import only reviewed contract and event-file hashes; validate canonical development keys, offsets, precision, clock order, versions, retries and original-row lineage.</li><li>Require complete delivered-prompt/wake scope, an available boundary and verified acquisition + receipt before cutoff. Preserve conflicting/rejected endpoints and counter predecessor links.</li><li>Keep interval duration, HR occupancy/gaps/freshness and supported activity separate from physiological features.</li><li>Test exact-version history: later edits, current answers and future answers cannot be applied backwards.</li></ol><p>Synthetic tests cover successful and rejected imports, version edits, pre-cutoff containment, counter breaks and participant-held-out injected-signal recovery through the actual extractor/model path. Synthetic success establishes implementation behavior, not real predictive validity.</p><details><summary>Frozen conditional experiment</summary><p>Same eligible rows/folds for A context; B original 60-min HR; C aligned HR; D context + aligned HR; E D + supported activity. Add context + original HR reference and separately labeled coverage controls. Binary composites ≥5/≥10; nervousness≥3 secondary. L2 logistic C={0.1,1}; three participant-grouped inner folds; five frozen outer folds; no imputation or expanded search. Report pooled/fold ranking, AP, balanced accuracy, sensitivity/specificity, calibration and paired participant uncertainty.</p><p>No new real model was fit. History-model activation remains conditional on exact version clocks and a reviewed chronological fit-time schedule, separate from unseen-participant evaluation.</p></details>""",
        )
    )
    sections.append(
        (
            "What we can conclude today",
            """<p>Notebook 18’s matched 1,355-response cohort gave nervousness AUROC 0.4875 for context and 0.5107 for context+HR; paired difference +0.0233 (95% participant interval −0.0210 to +0.0663). This does not establish a physiological increment. Do not compare this metric directly with experiments on different cohorts.</p><p>Earlier HR/activity regression and classification tested prediction from recorded pre-EMA data. The separate limited IBI pilot tested numerical variability calculations on a small qualifying-run sample; it did not establish broad normal-beat HRV availability. Neither expanded features nor longer windows resolve missing source semantics.</p><p>The 53.5% within-person score variation partitions observed variance. It can include unmeasured events, changing recall, measurement error and physiology; it does not quantify wearable-predictable variation.</p><p>No new current-pipeline defect has been established here. A draft-only Notebook 19 availability query was corrected to use counter supported-pair counts rather than the HR-specific reading column; the superseded draft is retained and is not a change to earlier features or results.</p>""",
        )
    )
    sections.append(
        (
            "Provider request and next action",
            """<p><b>Next step: one reviewed development payload/log slice plus the collector/exporter function.</b> We need the actual API/SDK/version, timestamp meanings, batching/order, prompt delivery and missed events, exact answer versions/receipt times, independent waking where available, quality/wear flags and step-counter semantics.</p><p>A generic Garmin API description cannot identify this export. Confirm whether “last prompt” includes the current notification. If acquisition times and boundaries are verified, run the already frozen matched logistic comparison; otherwise retain the blocked reasons.</p><p><a href="provider_data_request.md">Focused provider request (draft, not sent)</a> · <a href="source_contract.md">Typed contract and limitations</a> · <a href="verification.md">Verification</a> · <a href="specification.md">Frozen specification</a></p><p>All 32 final-test participants remain sealed. No raw-data changes, prior-artifact replacement, commit, push or provider message.</p>""",
        )
    )
    css = """*{box-sizing:border-box}body{margin:0;background:#f3f5f4;color:#20343e;font:17px/1.65 system-ui,sans-serif}header{background:#183c4b;color:white;padding:65px max(5vw,24px)}header p{max-width:850px;color:#d8e7e8}h1{font-size:clamp(32px,5vw,58px);line-height:1.13;max-width:1000px}h2{font-size:30px;line-height:1.25}h3{font-size:23px}nav{display:flex;flex-wrap:wrap;gap:12px;max-width:1120px;margin:25px auto;padding:0 20px}nav a{background:white;padding:7px 13px;border-radius:8px}a{color:#276e83}main{max-width:1120px;margin:auto;padding:0 20px}section{background:white;border-radius:16px;padding:34px;margin:25px 0;box-shadow:0 3px 12px #19323b09}figure{margin:24px 0}img{max-width:100%;height:auto}figcaption{color:#566871;font-size:14px}.table{overflow-x:auto;max-width:100%;margin:22px 0}table{border-collapse:collapse;width:max-content;min-width:100%;font-size:13px}td,th{padding:9px 12px;border-bottom:1px solid #dce5e5;text-align:left;min-width:110px;max-width:320px;white-space:normal;overflow-wrap:break-word}th{background:#eaf1f2}details{background:#f5f8f8;padding:18px;margin:20px 0;border-radius:9px}summary{cursor:pointer;font-weight:700}.tag{letter-spacing:.12em;font-size:13px;text-transform:uppercase;color:#a8d5d2}@media(max-width:600px){section{padding:22px}body{font-size:16px}h2{font-size:25px}header{padding:35px 24px}}@media print{section{break-inside:avoid;box-shadow:none}nav{display:none}body{background:white}}"""
    doc = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Garmin + EMA — Measurement alignment</title><style>'
        + css
        + '</style><header><div class="tag">Notebook 19 · Development only · Path A</div><h1>What time do our measurements actually represent?</h1><p>Recovering the evidence needed for valid recall alignment. Available diagnostics are complete; prospective aligned prediction remains blocked by missing measurement metadata.</p></header><nav>'
        + "".join(f'<a href="#s{i}">{i}. {t}</a>' for i, (t, _) in enumerate(sections, 1))
        + "</nav><main>"
        + "".join(
            f'<section id="s{i}"><div class="tag">{i:02}</div><h2>{t}</h2>{body}</section>'
            for i, (t, body) in enumerate(sections, 1)
        )
        + "</main></html>"
    )
    (base / "index.html").write_text(doc)
    summary = f"""# Notebook 19 — briefing for Will

**We have precise exported start clocks, but not verified sensor acquisition or delivered-prompt clocks.**
All available Path A diagnostics were executed; no new real predictive models were fit.

- All 3,706 development responses have actual/scheduled starts: 1,868 minute-resolution Morning/Endday and 1,838 second-resolution Midday.
- Median actual−scheduled delay: Morning 10.00 min; Midday 6.39 min; Endday 5.00 min. Starts >60 min after schedule: 140, 209 and 127 respectively. No missing starts among these exported responses; unanswered prompts are not observable from this denominator.
- At 60 min, {int(sixty.any_accepted):,} responses have any accepted HR; at 24 h, {int(day.any_accepted):,}. Median 24-h occupied-minute fraction is {100 * day.occupied_fraction_median:.1f}%. Longer availability is not a verified recall interval or continuous coverage.
- HR and steps show roughly 1.05-second median participant cadence. Step pairs are overwhelmingly increasing; gaps, repeated values and fractional clocks cannot identify callback/batch/acquisition semantics. {counters["negative_reset_pairs"]:,} negative counter pairs remain breaks, not negative activity.
- Availability/label differences are descriptive and affected by survey/time composition and sparse strata. Participant-bootstrap standardized intervals use only 672/2,000 valid draws at ≥5 and 1,067/2,000 at ≥10; interpret cautiously. Tables retain crude and standardized results rather than claiming independence.
- The fixed ±60 s support sensitivity preserves stored clocks and excludes original post-cutoff readings in every scenario. No model performance selects a shift.

Morning recall is since waking; Midday/Endday is since the last prompt. Current-response waking is retrospective.
A previous completed response is not a delivery log; Endday does not mean all day. See the exact questionnaire
citations in source_contract.md. The minute precision buffer remains 60 s; second-resolution starts are unbuffered.

**All 3,706 prospective recall and history gates remain BLOCKED.** The reviewed-hash importer, conservative
boundary/feature extractor, original-adjacency counter handling, exact-version history bridge and fixed matched
logistic runner are implemented and synthetic-tested. No real metadata have been invented. The history-model
schedule remains to be reviewed if independent version availability becomes available.

Notebook 18, on the same 1,355-response matched cohort, found context versus context+HR nervousness AUROC
0.4875 versus 0.5107; paired difference +0.0233 (95% interval −0.0210 to +0.0663). No established HR increment.
Do not compare scores across different cohorts. The 53.5% within-person score variance does not establish
wearable predictability. The limited IBI variability pilot was separate from HR/activity prediction and did not
establish generally usable normal-beat HRV. The activity probe had only five inactive versus 591 active bins.

**Recommendation:** obtain one development-only original payload/log slice plus collector/exporter code and
complete prompt/answer-version semantics. If the gates open, execute the frozen A–E matched logistic comparison;
do not expand the model search. The provider request is a draft only.

Outputs: `{ptr["public"]}/` (aggregate tables/figures), `{ptr["run"]}/` (ignored diagnostic rows and exact source catalog),
`notebooks/19_measurement_alignment.ipynb`, this directory's self-contained index.html, source_contract.md and
provider_data_request.md. Previous artifacts/folds/raw files are preserved; final test remains sealed.
"""
    (base / "summary_for_will.md").write_text(summary)
    nb = nbformat.v4.new_notebook(
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            garmin_public_aggregate_notebook=True,
        )
    )
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    nb.cells = [
        md(
            "# Measurement recovery and recall alignment\n\nNotebook 19 · Development only · Path A.\n\nWe execute the diagnostics supported by existing data. Missing prompt, acquisition and answer-version clocks keep prospective models blocked. Synthetic examples test behavior, not physiological validity. No final-test data or participant rows are displayed."
        ),
        code(
            "from pathlib import Path\nimport sys, json\nroot = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\nsys.path.insert(0, str(root / 'src'))\nfrom garmin_measurement_alignment.run import run\nfrom IPython.display import display, Image, Markdown\nimport pandas as pd\npointer = run(root)\npublic = root / pointer['public']\ndisplay(Markdown('Aggregate sources: `' + pointer['public'] + '`; private provenance: `' + pointer['run'] + '`'))\ndef table(name):\n    return pd.read_csv(public / (name + '.csv')) if (public / (name + '.csv')).exists() else pd.read_json(public / (name + '.json'))\ndef figure(name):\n    display(Image(filename=str(public / 'figures' / (name + '.png'))))"
        ),
        md(
            "## Actual starts are not prompt delivery\nAll exported starts are available; this does not count unanswered prompts. Offset is local minus UTC, in minutes."
        ),
        code("display(table('timing'))\nfigure('start_delays')"),
        md(
            "## Recall boundaries and the precision buffer\nMinute start 09:22 → cutoff 09:21 → [09:16,09:21). Second start 09:22:35 → [09:17:35,09:22:35). These are synthetic examples. Morning asks since waking; Midday/Endday since last prompt. Never substitute a completed response or arbitrary 24-hour interval."
        ),
        code(
            "figure('synthetic_timeline')\ndisplay(table('field_audit').fillna(''))\ndisplay(table('sensor_field_availability'))"
        ),
        md(
            "## Recording evidence\nCadence, repeats and gaps are observations. Periodic reporting, change-triggered output and delayed uploads remain competing interpretations. Step reset/rejection breaks remain intact; no missing period becomes zero activity."
        ),
        code(
            "display(table('cadence'))\ndisplay(pd.DataFrame([json.loads((public / 'counters.json').read_text())]))\nfigure('cadence')"
        ),
        md(
            "## Coverage and selection\nAll 3,706 responses remain counted. Accepted HR readings and supported counter pairs are distinct availability definitions. Sparse survey/time strata limit adjusted bootstrap inference."
        ),
        code(
            "coverage = table('coverage')\ndisplay(coverage[coverage.grouping.eq('sensor/lookback_minutes')].dropna(axis=1, how='all'))\nfigure('coverage')\nfigure('survey_coverage')\ndisplay(table('availability_associations'))\nfigure('availability')"
        ),
        md(
            "## Prespecified clock sensitivity\n−60/0/+60 seconds changes support only. Original DURING/POST records cannot be pulled backwards into any scenario. No offset is selected and no AUROC is computed."
        ),
        code("display(table('timing_sensitivity'))\nfigure('sensitivity')"),
        md(
            "## Verified behavior versus missing evidence\nThe typed importer requires separately approved hashes. Synthetic tests reject wrong identities, uncertain order, conflicting versions and future receipt; retain breaks; recover an injected signal across held-out synthetic participants. This does not validate real acquisition timestamps."
        ),
        code(
            "status = json.loads((public / 'status.json').read_text())\nassert status['new_real_predictive_fits'] == 0\nassert status['blocked_aligned'] == status['blocked_history'] == 3706\ndisplay(pd.DataFrame([status]))"
        ),
        md(
            "## Briefing and next step\nThe 53.5% within-person variation does not imply a wearable signal. Previous results are compared only on matching cohorts. Request the actual collector contract and complete prompt/version logs before the frozen conditional comparison. No provider messages were sent."
        ),
        code(
            "display(Markdown((root / 'reports/measurement_alignment/summary_for_will.md').read_text()))"
        ),
    ]
    nbformat.write(nb, root / "notebooks/19_measurement_alignment.ipynb")
    return dict(
        figures=7, sections=len(sections), code_cells=sum(c.cell_type == "code" for c in nb.cells)
    )


if __name__ == "__main__":
    print(build(Path.cwd()))
