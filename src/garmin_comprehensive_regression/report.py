"""Aggregate-only notebook and self-contained presentation; no individual plots."""

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

from garmin_comprehensive.report import markdown_table
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import verify_bundle, write_json
from .models import MAIN, FAMILIES

LABEL = dict(
    context="Context only",
    original60="Original HR4 + steps · 60m",
    hr60="Expanded HR · 60m",
    hr_all="Expanded HR · eight windows",
    hr_activity60="HR + activity · 60m",
    hr_activity_all="HR + activity · eight windows",
    full="Full supported · 438 candidates",
    full_corr="Full · correlation filter",
    full_l1select="Full · Lasso selection",
    full_enetselect="Full · Elastic Net selection",
    full_no_flags="Full · without missingness flags",
    pulseox_full="Full + provisional PulseOx",
    quality="Context + quality only",
    missingness="Context + missingness only",
)
FAMILY = dict(
    ridge="Ridge",
    lasso="Lasso",
    elastic="Elastic Net",
    huber="Huber",
    hgb="Histogram boosting",
    xgb="XGBoost",
    lgbm="LightGBM",
    dummy_mean="Mean dummy",
    dummy_median="Median dummy",
    fixed_ridge="Fixed Ridge α=1",
)
COLORS = ["#126c76", "#3f66a1", "#c88735", "#855c99", "#588351", "#ae5c51", "#7d7772"]


def render(root):
    root = Path(root)
    run = (
        root
        / json.loads((root / "private/comprehensive_regression/current.json").read_text())["run"]
    )
    out = root / "reports/comprehensive_regression"
    public = out / run.name
    if (out / "completion.json").exists():
        sealed = json.loads((out / "completion.json").read_text())
        assert sealed["run"] == run.name, "Publish future runs in a new presentation directory."
        verify_bundle(public)
        assert all(sha256(root / p) == h for p, h in sealed["presentation_sha256"].items())
        return out

    def load(name):
        return json.loads((public / (name + ".json")).read_text())

    metrics = pd.DataFrame(load("metrics"))
    diff = pd.DataFrame(load("paired_differences"))
    strata = pd.DataFrame(load("strata"))
    settings = pd.DataFrame(load("settings"))
    prediction = pd.read_parquet(run / "all_predictions.parquet")
    accounting = pd.read_parquet(run / "ema_accounting.parquet")
    collapse = []
    for name, f in prediction.groupby("model"):
        fold_sd = f.groupby("n_fold").prediction.std(ddof=0)
        collapse.append(
            dict(
                model=name,
                constant_folds=int(fold_sd.lt(1e-10).sum()),
                min_within_fold_sd=float(fold_sd.min()),
                max_within_fold_sd=float(fold_sd.max()),
                max_difference_from_training_mean=float(
                    np.max(np.abs(f.prediction - f.dummy_mean))
                ),
                max_difference_from_training_median=float(
                    np.max(np.abs(f.prediction - f.dummy_median))
                ),
            )
        )
    write_json(public / "prediction_collapse.json", collapse)
    primary = metrics[metrics.cohort.eq("all_valid") & metrics.complete_outer_folds]
    main = primary[primary["view"].isin(MAIN) & primary.family.isin(FAMILIES)]
    full = primary[primary["view"].eq("full")].set_index("family").reindex(FAMILIES).reset_index()
    median = primary[primary.family.eq("dummy_median")].iloc[0]
    figures = public / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def save(name):
        plt.savefig(figures / name, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close()

    y = accounting.pss_combo.dropna()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    counts = y.value_counts().reindex(range(3, 16), fill_value=0)
    ax.bar(
        counts.index,
        counts.values,
        color=[COLORS[0] if i <= 4 else COLORS[1] if i <= 9 else COLORS[2] for i in counts.index],
    )
    ax.set(
        xticks=range(3, 16),
        xlabel="Observed three-item composite",
        ylabel="Responses",
        title="The continuous outcome remains unchanged",
    )
    save("01_outcome.png")
    grid = main.pivot(index="view", columns="family", values="mae").reindex(
        index=MAIN, columns=FAMILIES
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    distance = float(np.nanmax(np.abs(grid.to_numpy() - median.mae)))
    im = ax.imshow(
        grid.to_numpy() - median.mae, cmap="RdBu_r", aspect="auto", vmin=-distance, vmax=distance
    )
    for i in range(len(grid)):
        for j in range(len(grid.columns)):
            v = grid.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(7), [FAMILY[x] for x in FAMILIES], rotation=20, ha="right")
    ax.set_yticks(range(len(MAIN)), [LABEL[x] for x in MAIN])
    ax.set_title("Matched OOF MAE: lower is better; numbers are absolute MAE")
    fig.colorbar(im, ax=ax, label="MAE minus matching median dummy")
    save("02_model_features.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, base, color in [(-0.1, "context", COLORS[0]), (0.1, "original60", COLORS[2])]:
        for i, family in enumerate(FAMILIES):
            f = diff[
                diff.model.eq(f"all_valid__full__{family}")
                & diff.baseline.eq(f"all_valid__{base}__{family}")
            ]
            if len(f):
                r = f.iloc[0]
                ax.errorbar(
                    r.mae_difference,
                    i + offset,
                    xerr=[[r.mae_difference - r.mae_low], [r.mae_high - r.mae_difference]],
                    fmt="o",
                    color=color,
                    capsize=3,
                    label=base if i == 0 else None,
                )
    ax.axvline(0, color="#333", ls="--", lw=1)
    ax.set_yticks(range(7), [FAMILY[x] for x in FAMILIES])
    ax.invert_yaxis()
    ax.set(
        xlabel="Full-model MAE minus baseline MAE (negative favors full)",
        title="Paired participant-bootstrap 95% intervals",
    )
    ax.legend(title="Matching baseline")
    save("03_paired.png")
    density_max = max(
        np.histogram2d(
            g.pss_combo, g.prediction, bins=[np.arange(2.5, 16), np.linspace(0, 18, 55)]
        )[0].max()
        for _, g in prediction[
            prediction.model.isin(
                [f"all_valid__full__{family}" for family in ["ridge", "hgb", "xgb", "lgbm"]]
            )
        ].groupby("model")
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, family in zip(axes.flat, ["ridge", "hgb", "xgb", "lgbm"]):
        f = prediction[prediction.model.eq(f"all_valid__full__{family}")]
        if len(f):
            hist, xe, ye = np.histogram2d(
                f.pss_combo, f.prediction, bins=[np.arange(2.5, 16), np.linspace(0, 18, 55)]
            )
            density_image = ax.pcolormesh(
                xe, ye, np.ma.masked_less(hist.T, 5), cmap="YlGnBu", vmin=5, vmax=density_max
            )
            ax.plot([3, 15], [3, 15], "--", color="#b37821")
            ax.set_title(FAMILY[family])
            outside = int(((f.prediction < 0) | (f.prediction > 18)).sum())
            ax.text(
                0.02,
                0.96,
                f"Outside display: {outside}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )
        ax.set(xlabel="Actual PSS", ylabel="Predicted PSS", ylim=(0, 18))
    fig.suptitle("Aggregate actual–prediction density; cells with fewer than 5 responses hidden")
    fig.tight_layout()
    fig.subplots_adjust(right=0.87)
    color_ax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
    fig.colorbar(density_image, cax=color_ax, label="Responses per bin (shared scale)")
    save("04_actual_predicted.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.arange(2.75, 15.51, 0.5)
    axes[0].hist(y, bins=bins, histtype="step", lw=2, color="#333", density=True, label="Actual")
    for family, color in zip(["ridge", "hgb", "xgb", "lgbm"], COLORS):
        f = prediction[prediction.model.eq(f"all_valid__full__{family}")]
        axes[0].hist(
            f.prediction,
            bins=bins,
            histtype="step",
            density=True,
            label=FAMILY[family],
            color=color,
        )
        g = strata[
            strata.model.eq(f"all_valid__full__{family}") & strata.dimension.eq("score_band")
        ].set_index("stratum")
        if len(g):
            axes[1].plot(
                range(3),
                g.reindex(["3–4", "5–9", "10–15"]).signed_error,
                "o-",
                label=FAMILY[family],
                color=color,
            )
    axes[0].set(
        xlabel="PSS score (display range 3–15)",
        ylabel="Density within display",
        title="Do predictions collapse?",
    )
    axes[0].legend(fontsize=9)
    axes[1].set_xticks(range(3), ["3–4", "5–9", "10–15"])
    axes[1].axhline(0, color="#333", lw=1)
    axes[1].set(
        xlabel="Observed score band",
        ylabel="Mean prediction − actual",
        title="Negative residuals indicate underprediction",
    )
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save("05_predictions_residuals.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for family, color in [("ridge", COLORS[0]), ("hgb", COLORS[1])]:
        vals = []
        for m in [5, 15, 30, 60, 120, 360, 720, 1440]:
            view = "hr60" if m == 60 else f"hr_{m}m"
            r = primary[primary.family.eq(family) & primary["view"].eq(view)]
            vals.append(r.mae.iloc[0] if len(r) else np.nan)
        axes[0].plot(range(8), vals, "o-", label=FAMILY[family], color=color)
    axes[0].axhline(median.mae, color="#777", ls="--", label="Median dummy")
    axes[0].set_xticks(range(8), [5, 15, 30, 60, 120, 360, 720, 1440], rotation=30)
    axes[0].set(
        xlabel="Lookback minutes (categorical)",
        ylabel="OOF MAE",
        title="Single-window HR on identical responses",
    )
    axes[0].legend()
    for family, color in [("ridge", COLORS[0]), ("hgb", COLORS[1])]:
        views = [
            "full",
            "full_corr",
            "full_l1select",
            "full_enetselect",
            "full_no_flags",
            "pulseox_full",
        ]
        f = primary[primary.family.eq(family)].set_index("view").reindex(views)
        axes[1].plot(range(6), f.mae, "o-", label=FAMILY[family], color=color)
    axes[1].set_xticks(
        range(6),
        ["Full", "Corr.", "L1 select", "EN select", "No flags", "+PulseOx"],
        rotation=35,
        ha="right",
    )
    axes[1].set(ylabel="OOF MAE", title="Selection and separately labeled sensitivities")
    axes[1].legend()
    fig.tight_layout()
    save("06_windows_selection.png")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(7)
    tr = []
    for family in FAMILIES:
        f = settings[settings.model.eq(f"all_valid__full__{family}")]
        tr.append(
            float(np.average(f.train_mae, weights=f.training_responses)) if len(f) else np.nan
        )
    ax.bar(x - 0.18, tr, 0.36, color=COLORS[0], label="Training (repeated memberships)")
    ax.bar(x + 0.18, full.mae, 0.36, color=COLORS[2], label="Held-out OOF")
    ax.axhline(median.mae, color="#333", ls="--", label="OOF median dummy")
    ax.set_xticks(x, [FAMILY[f] for f in FAMILIES], rotation=20, ha="right")
    ax.set(ylabel="MAE", title="Fit quality is not generalization")
    ax.legend()
    save("07_train_oof.png")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    control_views = ["context", "original60", "full", "missingness", "quality"]
    for family, color in [("ridge", COLORS[0]), ("hgb", COLORS[1])]:
        f = primary[primary.family.eq(family)].set_index("view").reindex(control_views)
        ax.plot(range(5), f.mae, "o-", label=FAMILY[family], color=color)
    ax.axhline(median.mae, ls="--", color="#777", label="Median dummy")
    ax.set_xticks(
        range(5), ["Context", "Original", "Full physiology", "Missingness only", "Quality only"]
    )
    ax.set(ylabel="OOF MAE", title="Controls for recording availability")
    ax.legend()
    save("08_controls.png")
    # Reuse prior aggregate quality descriptions without reprocessing any sensor records.
    previous = root / "reports/comprehensive_features/818ddc22d6f28fdc"
    for name in ["recording_support", "feature_summary"]:
        write_json(public / (name + ".json"), json.loads((previous / (name + ".json")).read_text()))
    candidates = main[main["view"].eq("full")].sort_values("mae")
    best = candidates.iloc[0]
    contrast = diff[
        diff.model.eq(best.model) & diff.baseline.eq(f"all_valid__context__{best.family}")
    ].iloc[0]
    dummy_delta = diff[
        diff.model.eq(best.model) & diff.baseline.eq("all_valid__dummy_median")
    ].iloc[0]
    high = strata[
        strata.model.eq(best.model) & strata.dimension.eq("score_band") & strata.stratum.eq("10–15")
    ].iloc[0]
    evidence = (
        f"The lowest point MAE among the prespecified full-feature models is {FAMILY[best.family]} "
        f"at {best.mae:.3f}, compared with {median.mae:.3f} for the matching median dummy. "
        f"Its paired MAE difference from the median is {dummy_delta.mae_difference:+.3f} "
        f"(95% interval {dummy_delta.mae_low:+.3f} to {dummy_delta.mae_high:+.3f}); versus its "
        f"matching context model, {contrast.mae_difference:+.3f} "
        f"({contrast.mae_low:+.3f} to {contrast.mae_high:+.3f}). "
        f"RMSE is {best.rmse:.3f}, R² {best.r2:.3f}, and Spearman correlation {best.spearman:.3f}. "
        f"For actual scores 10–15, MAE is {high.mae:.3f} and mean prediction−actual is {high.signed_error:+.3f}."
    )
    conclusion = (
        "The conditional intervals support a lower error than both matching baselines for this exploratory comparison."
        if dummy_delta.mae_high < 0 and contrast.mae_high < 0
        else "These comparisons do not establish a reliable improvement over both the matching median and context baselines."
    )
    conclusion += " Point minima are descriptive, not a new selected model or final-test result. Repeated development exploration remains a limitation."
    collapse_row = next(x for x in collapse if x["model"] == best.model)
    collapse_text = (
        f"{FAMILY[best.family]} selects zero coefficients in every full-feature outer fold and reproduces "
        "the matching training-mean dummy. Its small pooled prediction variation comes only from different "
        "training-fold means; it has no within-fold physiological discrimination. The median dummy predicts "
        "6 throughout and has undefined Spearman correlation, not zero correlation."
        if collapse_row["constant_folds"] == 5
        and collapse_row["max_difference_from_training_mean"] < 1e-10
        else "Within-fold prediction spread and equality to matching training dummies are explicitly audited."
    )
    xgb_delta = diff[
        diff.model.eq("all_valid__full__xgb") & diff.baseline.eq("all_valid__full__ridge")
    ].iloc[0]
    lowest_learned = main.sort_values("mae").iloc[0]
    full_context = diff[diff.model.isin(full.model) & diff.baseline.str.contains("__context__")]
    selected = (
        primary[primary["view"].isin(["full_corr", "full_l1select", "full_enetselect"])]
        .sort_values("mae")
        .iloc[0]
    )
    answers = [
        dict(
            question="Do expanded features improve continuous prediction?",
            answer=f"No full-feature model has a lower point MAE than the median dummy ({median.mae:.3f}). "
            f"Even the lowest learned main comparison, {LABEL[lowest_learned['view']]} / {FAMILY[lowest_learned.family]}, "
            f"has MAE {lowest_learned.mae:.3f}.",
        ),
        dict(
            question="Do physiological values add beyond context?",
            answer=f"{int(full_context.mae_high.lt(0).sum())} of seven full-versus-context paired intervals supports lower MAE. "
            "The matched full models often worsen error; the quality/missingness controls do not establish a physiological signal.",
        ),
        dict(
            question="Do selection or longer windows solve this?",
            answer=f"The lowest selected-feature point MAE is {selected.mae:.3f}, still above the median baseline. "
            "Single-window and multiwindow results do not justify an optimal scale. PulseOx stays provisional and does not rescue the full benchmark.",
        ),
        dict(
            question="Do nonlinear models outperform Ridge?",
            answer=f"Full XGBoost minus full Ridge MAE is {xgb_delta.mae_difference:+.3f} "
            f"(95% interval {xgb_delta.mae_low:+.3f} to {xgb_delta.mae_high:+.3f}). "
            "That modest conditional advantage does not beat the matching median/context baseline. "
            "The compared losses also differ; this is not isolated evidence for nonlinearity.",
        ),
        dict(
            question="Can the models identify higher continuous scores?",
            answer="For the 515 responses scoring 10–15, actual mean PSS is 11.16. Full-model mean predictions are about 6.1–6.5, with MAE about 4.7–5.1. Higher-score underprediction remains severe.",
        ),
        dict(
            question="Is the problem implementation, sparsity or weak physiology?",
            answer="Synthetic controls, exact legacy replays and estimator replays passed. This rules out the tested computational errors, not all possible errors. Sparse/stale readings, recall mismatch and source semantics remain plausible limitations; this benchmark cannot identify their causal contribution.",
        ),
    ]
    write_json(public / "research_answers.json", answers)
    baselines = (
        primary[primary.family.isin(["dummy_mean", "dummy_median"])]
        .set_index("family")
        .loc[["dummy_median", "dummy_mean"]]
        .reset_index()
    )
    result = pd.concat([baselines, full], ignore_index=True)[
        [
            "family",
            "mae",
            "mae_low",
            "mae_high",
            "rmse",
            "r2",
            "spearman",
            "participant_mae",
            "prediction_sd",
        ]
    ].copy()
    result["family"] = result.family.map(FAMILY)
    result = result.rename(
        columns={
            "family": "Model",
            "mae": "MAE",
            "mae_low": "95% low",
            "mae_high": "95% high",
            "rmse": "RMSE",
            "r2": "R²",
            "spearman": "Spearman",
            "participant_mae": "Participant MAE",
            "prediction_sd": "Prediction SD",
        }
    )
    write_json(
        public / "headline.json",
        dict(
            evidence=evidence,
            conclusion=conclusion,
            collapse=collapse_text,
            best_full_point_model=best.model,
            primary_responses=int(best.responses),
            primary_participants=int(best.participants),
        ),
    )

    def table(f, cols=None, limit=30):
        f = f[cols] if cols else f
        return (
            '<div class="table">'
            + f.head(limit).to_html(
                index=False, border=0, float_format=lambda x: f"{x:.3f}", escape=True
            )
            + "</div>"
        )

    def image(name, caption):
        b = base64.b64encode((figures / name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'

    def details(title, text):
        return f"<details><summary>{title}</summary>{text}</details>"

    def section(n, title, text):
        return f'<section id="s{n}"><p class="eyebrow">{n:02d} / METHODS & FINDINGS</p><h2>{title}</h2>{text}</section>'

    body = []
    body.append(
        section(
            1,
            "Return to the continuous question.",
            "<p>Predict the unchanged three-item perceived-stress composite, 3–15. This is exploratory prediction from recorded pre-EMA data, not validated real-time prediction or a clinical diagnosis.</p>"
            '<div class="cards"><div><b>3,706</b><span>development EMA rows retained</span></div><div><b>3,689</b><span>valid-score/context responses modeled</span></div><div><b>125</b><span>development participants; 32 final-test participants sealed</span></div></div>'
            + image(
                "01_outcome.png",
                "Score bands are error-analysis groups only. No binary target or new score coding is introduced.",
            ),
        )
    )
    body.append(
        section(
            2,
            "The sensor features are reused, not reinvented.",
            '<div class="flow">Sealed Notebook 13 matrix → training-fold transform → grouped inner tuning → held-out participant predictions → aggregate evaluation</div>'
            "<p>The full main block offers 438 observed-summary candidates: 240 HR, 56 steps, 56 floors, 48 earlier-HR reference/deviation, 24 HR/activity co-observation and 14 adjacent-window contrasts. Eight existing lookbacks span five minutes to 24 hours. PulseOx adds 240 provisional columns only in a separate sensitivity.</p>"
            "<p>No raw records were reprocessed. Existing timestamps, buffers, support rules, row keys and folds remain unchanged. No unvalidated HRV, Garmin stress, outcomes, identifiers or quality measures enter physiological blocks. Context is actual-local time and survey category.</p>"
            '<div class="synthetic"><b>Synthetic teaching example:</b> a minute-resolution start at 09:22 gives cutoff 09:21. Its 60-minute interval is [08:21,09:21). Observed HR summaries and disjoint earlier history form features. Items 4,2,3 give 4+(6−2)+(6−3)=11. A prediction of 7 has absolute error 4 and signed residual −4. The held-out participant’s answers never enter feature construction or tuning.</div>'
            + details(
                "Feature availability and accounting",
                table(pd.DataFrame(load("feature_availability")), limit=28),
            ),
        )
    )
    body.append(
        section(
            3,
            "A matched, nested benchmark.",
            "<p>Seven model families × eight main feature blocks, followed by prespecified selection, missingness, quality, PulseOx, lookback and complete-case checks: 98 comparisons, five frozen outer participant folds each. Every inner split is grouped by participant. All tuning minimizes mean inner MAE.</p>"
            "<p>Means, medians, scaling, category encoding, correlation filtering and supervised selection are fitted inside each training split. Feature-level imputation never fills raw readings or targets. HGB, XGBoost and LightGBM compare bounded squared-error and absolute-error settings; no row-level early stopping, prediction clipping or post-result search expansion.</p>"
            "<p>Separate Lasso/Elastic Net selectors retain context and fit only training outcomes. Failed convergence is reported, not replaced with a dummy or an altered fold. Complete grids and loss choices are in specification.md and the aggregate settings table.</p>"
            + details(
                "Selected settings and convergence",
                table(
                    settings,
                    [
                        "view",
                        "family",
                        "fold",
                        "setting",
                        "inner_mae",
                        "selected_features",
                        "nonconverged_candidates",
                    ],
                    limit=40,
                ),
            ),
        )
    )
    body.append(
        section(
            4,
            "Do expanded features reduce held-out error?",
            '<p class="lead">'
            + html.escape(evidence)
            + '</p><div class="note">'
            + html.escape(conclusion)
            + "</div>"
            + table(result)
            + image(
                "02_model_features.png",
                "Every main cell uses the same eligible responses and frozen outer folds. Color is relative to the matching median baseline.",
            )
            + image(
                "03_paired.png",
                "Negative differences favor full features. Intervals resample participants with their saved OOF predictions; they do not include retuning uncertainty or multiplicity correction.",
            )
            + details(
                "All paired comparisons",
                table(
                    diff,
                    ["model", "baseline", "responses", "mae_difference", "mae_low", "mae_high"],
                    limit=400,
                ),
            ),
        )
    )
    body.append(
        section(
            5,
            "Can predictions follow higher scores?",
            '<div class="note">'
            + html.escape(collapse_text)
            + "</div>"
            + image(
                "04_actual_predicted.png",
                "Ideal predictions follow the diagonal. Small cells are hidden for presentation/privacy; every response remains in metrics. Axes show 0–18; out-of-display counts are annotated.",
            )
            + image(
                "05_predictions_residuals.png",
                "Histogram density is normalized within its display range. No predictions are clipped in fitting or metrics. Full prediction ranges and out-of-range fractions are reported numerically.",
            )
            + details(
                "Within-fold prediction-collapse audit", table(pd.DataFrame(collapse), limit=110)
            )
            + details(
                "Full-model prediction ranges",
                table(
                    primary[primary["view"].eq("full")],
                    [
                        "family",
                        "prediction_min",
                        "prediction_max",
                        "prediction_sd",
                        "outside_target_range",
                    ],
                ),
            )
            + details(
                "Errors by score band and survey",
                table(
                    strata[strata["view"].eq("full") & strata.cohort.eq("all_valid")],
                    [
                        "family",
                        "dimension",
                        "stratum",
                        "responses",
                        "mae",
                        "rmse",
                        "signed_error",
                        "prediction_mean",
                        "actual_mean",
                    ],
                    limit=50,
                ),
            ),
        )
    )
    body.append(
        section(
            6,
            "Separate generalization from fitting and feature breadth.",
            image(
                "07_train_oof.png",
                "Training MAE averages repeated outer-training memberships. Lower training error alone does not establish predictive value.",
            )
            + image(
                "06_windows_selection.png",
                "These comparisons assess prespecified scales and procedures; they do not select an optimal physiological window.",
            )
            + image(
                "08_controls.png",
                "Quality-only and missingness-only models are recording-process controls, not physiological predictors.",
            )
            + details(
                "Feature-selection stability (training estimates only)",
                table(
                    pd.DataFrame(load("feature_stability"))
                    .query("model == 'all_valid__full_l1select__ridge'")
                    .sort_values("present_folds", ascending=False),
                    limit=20,
                ),
            ),
        )
    )
    body.append(
        section(
            7,
            "Compare old regressions only where responses match.",
            "<p>The original 60-minute HR+steps complete-case cohort is evaluated separately. Context, original and full feature models are refitted on identical training and validation rows. Fixed-alpha Ridge is independently compared against saved Notebook 05 and 09 predictions by EMA key, participant, fold and target.</p>"
            "<p>All-valid predictions restricted to these validation responses are labeled separately: they were trained on a larger cohort. They must not be mistaken for the complete-case refit. The 3,689-row headline is never directly compared with an older 1,355-row score.</p>"
            + details("Exact legacy replay", table(pd.DataFrame(load("legacy_replay"))))
            + details(
                "Matched complete-case results",
                table(
                    metrics[metrics.cohort.eq("complete60")],
                    [
                        "view",
                        "family",
                        "responses",
                        "participants",
                        "mae",
                        "rmse",
                        "r2",
                        "spearman",
                    ],
                    limit=20,
                ),
            )
            + details(
                "Saved all-valid fits on complete-case validation responses",
                table(pd.DataFrame(load("restricted_evaluation")), limit=30),
            ),
        )
    )
    body.append(
        section(
            8,
            "What this resolves—and what it cannot.",
            "".join(
                f"<h3>{html.escape(a['question'])}</h3><p>{html.escape(a['answer'])}</p>"
                for a in answers
            )
            + "<p>The benchmark tests whether broader observed summaries, regularization, supervised selection and bounded nonlinear models improve prediction under the same participant-held-out evaluation. The synthetic positive control and legacy replays check implementation; they do not establish physiological validity.</p>"
            "<p>Point samples may be sparse or stale. Occupied minutes are not continuous monitoring. Morning asks since waking; Midday and Endday refer to the last prompt, whose actual timing remains unavailable. Acquisition versus receipt clocks, batching and signal-quality semantics remain uncertain. These are plausible limitations, not demonstrated causes of an error difference.</p>"
            "<p>High within-person outcome variation does not imply that the wearable measurements explain it. Subject grouping also means prediction for unseen participants after whatever unlabeled wearable history is available. No earlier or current EMA answer is a predictor.</p>"
            '<div class="note"><b>Next experiment:</b> resolve exporter timestamp/quality evidence and prompt/recall boundaries before adding more features or a wider search. Any promising matched effect here needs a separately specified confirmation; keep the final test sealed until the measurement and analysis contract is fixed.</div>',
        )
    )
    body.append(
        section(
            9,
            "Reproducible, private and reviewable.",
            "<p>Private versioned jobs contain estimators, training/inner memberships and OOF predictions. Public outputs contain aggregates and synthetic examples only. All 3,706 development responses remain in an accounting table. Prior notebooks, bundles, raw data and assignments are preserved.</p>"
            "<p>The XGBoost metadata serializer explicitly records its nonfinite default missing-value sentinel. This is a serialization correction, not sensor imputation or a model change. Compatible completed jobs are reused only after input hashes and fit-path code are verified.</p>"
            "<p>Read verification.md for tests, Ruff, notebook execution, replay, privacy and integrity checks. Read README.md for exact paths and commands. No commit, push, final-test evaluation or provider messages.</p>",
        )
    )
    css = """*{box-sizing:border-box}body{margin:0;background:#f1f5f5;color:#1c3742;font:17px/1.65 system-ui,sans-serif}header,main,footer{max-width:1180px;margin:auto;padding:36px}header{padding-top:70px}h1,h2{font-family:Georgia,serif;line-height:1.13;letter-spacing:-.025em}h1{font-size:58px;margin:15px 0}h2{font-size:36px;margin:12px 0 28px}p{margin:18px 0}.eyebrow{font-size:12px;letter-spacing:.18em;color:#087c7b;font-weight:800}.lead{font-size:21px;line-height:1.5}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.cards div,.flow{background:#edf4f6;border-radius:9px;padding:25px}.cards b{display:block;font-size:34px}.cards span{display:block;font-size:14px}.flow{font-weight:700}section{background:white;padding:40px;border:1px solid #d5e0e5;border-radius:16px;margin-bottom:28px}figure{margin:28px 0}img{display:block;width:100%;height:auto}figcaption{font-size:14px;color:#526c78;margin-top:12px}.table{overflow:auto;width:100%;border:1px solid #d5e0e5;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px 12px;border-bottom:1px solid #dde5e9;white-space:nowrap}th{background:#edf4f6;text-align:left}details{margin:20px 0;border:1px solid #d5e0e5;padding:16px;border-radius:8px}summary{cursor:pointer;font-weight:700}.note,.synthetic{padding:22px;border-left:4px solid #087c7b;background:#ebf6f3}.synthetic{background:#f0f4fc;border-color:#3f66a1}nav a{display:inline-block;margin-right:18px;color:#365f92;text-decoration:none}button{padding:10px 16px;background:white;border:1px solid #bacdd7;border-radius:5px;cursor:pointer}footer{font-size:14px;color:#526c78}@media(max-width:700px){header,main,footer{padding:20px}h1{font-size:41px}h2{font-size:29px}section{padding:22px}.cards{grid-template-columns:1fr}.lead{font-size:19px}}@media print{body{background:white}header,main{padding:12px}section{break-inside:auto}nav,button{display:none}details{break-inside:avoid}figure{break-inside:avoid}}"""
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Comprehensive PSS regression · Garmin + EMA</title><style>'
        + css
        + "</style></head><body>"
    )
    page += (
        '<header><p class="eyebrow">GARMIN × EMA / NOTEBOOK 14 / EXPLORATORY DEVELOPMENT RESEARCH</p><h1>More physiological information.<br>A rigorous return to regression.</h1><p class="lead">438 supported candidate features. Seven model families. Matched participant-held-out predictions.</p><nav>'
        + "".join(f'<a href="#s{i}">{i:02d}</a>' for i in range(1, 10))
        + '</nav><p><button onclick="window.print()">Print / PDF</button> <button onclick="document.querySelectorAll(\'details\').forEach(d=>d.open=!d.open)">Toggle technical details</button></p></header><main>'
        + "".join(body)
        + "</main><footer>Continuous three-item composite, not a diagnostic scale or validated real-time prediction. Final-test participants remain sealed.</footer></body></html>"
    )
    (out / "index.html").write_text(page)
    summary = (
        "# Comprehensive continuous PSS regression — summary for Will\n\n"
        + evidence
        + "\n\n"
        + conclusion
        + "\n\n"
        + collapse_text
        + "\n\n"
        + markdown_table(result)
        + "\n\n"
        + markdown_table(pd.DataFrame(answers))
    )
    summary += "\n\nAll main models use the same 3,689 valid-score/context responses from 125 participants. All 3,706 EMA rows are retained in accounting. These are newly tuned regressions, not conversions of classification probabilities.\n\n438 supported summaries are reused; provisional PulseOx stays separate. Eight main blocks, seven model families and prespecified secondary checks produce 98 comparisons across five frozen outer folds. Every transform/selector and hyperparameter decision stays inside grouped training splits. MAE is the selection objective; boosting losses are explicitly compared.\n\nThe full tables report all feature groups, score-band errors, calibration of continuous predictions, prediction spread, convergence, missingness/quality controls and matched legacy comparisons. Point minima do not identify an optimal window, model or physiological mechanism. High-score residuals and weak correlation must not be hidden by pooled MAE.\n\nIntervals condition on fixed development predictions and omit retuning uncertainty and multiple-comparison correction. Timing, recall alignment, source quality, sparse observations and participant heterogeneity remain unresolved. No new HRV or proprietary Garmin stress is included.\n\nNext: resolve measurement timing/quality and prompt/recall boundaries, then specify a narrow confirmation of any supported matched effect. Do not automatically expand the search or open the final test. See verification.md and README.md for checks and exact output paths.\n"
    (out / "summary.md").write_text(summary)
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        garmin_public_aggregate_notebook=True,
    )
    cells = []

    def md(x):
        cells.append(nbf.v4.new_markdown_cell(x))

    def code(x):
        cells.append(nbf.v4.new_code_cell(x))

    md(
        "# 14 · Comprehensive continuous PSS regression\n\nAn exploratory development benchmark. All displayed tables/figures are aggregates or synthetic examples. The final test remains sealed. No raw sensor processing is repeated."
    )
    code(
        "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nROOT=Path.cwd() if (Path.cwd()/'reports').exists() else Path.cwd().parent\nREPORT=ROOT/'reports/comprehensive_regression/"
        + run.name
        + "'\ndef load(n): return json.loads((REPORT/(n+'.json')).read_text())\ndef table(n,columns=None,limit=25):\n    f=pd.DataFrame(load(n)); display((f[columns] if columns else f).head(limit))\ndef figure(n): display(Image(filename=str(REPORT/'figures'/n)))"
    )
    md(
        "## 1 · Outcome and accounting\n\nThe unchanged three-item composite requires all valid items. All 3,706 responses remain accounted for; 3,689 valid-score/context responses enter matched main models. The 3–4 / 5–9 / 10–15 bands are evaluation strata, not new targets."
    )
    code("figure('01_outcome.png')\ntable('cohort_composition')")
    md(
        "## 2 · Reused features and a synthetic example\n\n438 supported summaries across eight windows; 240 provisional PulseOx columns only in a separate sensitivity. Example: minute start 09:22 → cutoff 09:21 → 60-minute interval [08:21,09:21). Items 4,2,3 give 11. Prediction 7 has MAE contribution 4 and residual −4. No current answer reaches the predictors."
    )
    code(
        "display(pd.DataFrame([{'synthetic_nervous':4,'synthetic_cope':2,'synthetic_control':3,'composite':4+(6-2)+(6-3),'prediction':7,'absolute_error':4,'signed_residual':-4}]))\ntable('feature_availability',limit=28)"
    )
    md(
        "## 3 · Nested evaluation\n\nFive unchanged participant folds; three grouped inner folds. Training-only imputation, scaling, encoding and selection. Inner MAE selects settings. No row-level early stopping, target imputation, clipping or model expansion based on held-out results. Boosting families compare both absolute and squared-error losses."
    )
    code(
        "table('settings',['view','family','fold','setting','inner_mae','selected_features','nonconverged_candidates'])\ntable('failures')"
    )
    md(
        "## 4 · Matched model and feature comparisons\n\nRead MAE beside RMSE, R², correlation and prediction spread. The median dummy must be fitted on the matching training cohort."
    )
    code(
        "figure('02_model_features.png')\nf=pd.DataFrame(load('metrics'))\ndisplay(f[f.cohort.eq('all_valid') & (f['view'].eq('full') | f.family.str.startswith('dummy_'))][['family','mae','mae_low','mae_high','rmse','r2','spearman','prediction_sd']])\nfigure('03_paired.png')"
    )
    md(
        "## 5 · Higher outcomes and collapsed predictions\n\nThe ideal line is actual = predicted. Density cells with fewer than five responses are hidden; all observations remain in metrics. Display ranges do not clip model predictions. Negative residuals indicate underprediction."
    )
    code(
        "figure('04_actual_predicted.png')\nfigure('05_predictions_residuals.png')\ns=pd.DataFrame(load('strata'));display(s[s.cohort.eq('all_valid') & s['view'].eq('full') & s.dimension.eq('score_band')][['family','stratum','responses','mae','signed_error','prediction_mean','actual_mean']])"
    )
    md(
        "## 6 · Selection, time windows, and recording controls\n\nMissingness and quality are distinct from physiological values. Selection stability is descriptive; correlated predictors can substitute. Point minima do not establish an optimal window."
    )
    code(
        "figure('06_windows_selection.png')\nfigure('07_train_oof.png')\nfigure('08_controls.png')"
    )
    md(
        "## 7 · Compare with older regressions fairly\n\nExact fixed-Ridge replays verify linkage and preprocessing against Notebook 05 and 09. Complete-case refits and restricted validation of larger-training-cohort models are reported separately. Never compare their headline metrics across unequal evaluation rows."
    )
    code(
        "table('legacy_replay')\ndisplay(f[f.cohort.eq('complete60')][['view','family','responses','participants','mae','rmse','r2']])\ntable('restricted_evaluation')"
    )
    md(
        "## 8 · Findings and limitations\n\n"
        + evidence
        + "\n\n"
        + conclusion
        + "\n\nSparse or stale readings and survey recall mismatch remain possible limitations. These results cannot identify the cause without new measurement evidence. Participant-bootstrap intervals condition on fixed OOF predictions, repeated exploration and prespecified grids."
    )
    code(
        "display(Markdown((ROOT/'reports/comprehensive_regression/summary.md').read_text()))\nif (REPORT/'verification.json').exists(): display(pd.Series(load('verification'),name='verification').to_frame())"
    )
    nb.cells = cells
    nbf.write(nb, root / "notebooks/14_comprehensive_pss_regression.ipynb")
    return out


if __name__ == "__main__":
    print(render(Path.cwd()))
