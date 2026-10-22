"""Build the short code-only overview; execution uses the current environment."""

from pathlib import Path
import nbformat as nb


def build(root):
    cells = [
        """# Sources and scope
from pathlib import Path
from os.path import relpath
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, FileLink
root = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(root / "src"))
from garmin_central_overview import overview as ov  # noqa: E402
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.precision", 3)
data = ov.prepare(root)
display(data["provenance"].set_index("source"))""",
        """# Dataset overview
inventory = data["inventory"]
display(pd.DataFrame([
    ["Original EMA inventory (structural totals only)", 4680, 157, "115–137 raw fields by survey"],
    ["Development EMA / candidate matrix", 3706, 125, "678 candidates; 693 saved columns incl. metadata"],
    ["Development participants with Garmin", pd.NA, 123, "Two participants have no Garmin; their EMA rows remain"],
    ["Model cohort: valid score and context", 3689, 125, "438 supported candidates + context"],
], columns=["scope", "EMA rows", "participants", "dimensions / note"]).set_index("scope"))
display(data["sensors"].set_index("modality"))""",
        """# Preprocessing and retained records
n = data["normalized"]
display(pd.DataFrame([
    ["Physical sensor records preserved", n["counts"]["physical_sensor_rows"], "No raw rows deleted"],
    ["Exact duplicate excess excluded from logical view", n["counts"]["exact_duplicate_excess_rows"], "Source records remain in physical tables"],
    ["Logical sensor records", n["counts"]["logical_sensor_rows"], "Repeated values at different times retained"],
    ["Conflict-flagged physical records", n["counts"]["conflicting_physical_rows"], "Retained and flagged; not a separate subtraction"],
    ["Mapped using mobile code", n["mapping_rows"]["content_mbl_cod"], "Verified crosswalk"],
    ["Mapped using respondent key", n["mapping_rows"]["content_rsp_id"], "Verified crosswalk"],
    ["Mapped using source content hash", n["mapping_rows"]["reference_content_hash"], "Verified crosswalk"],
    ["Development EMA rows retained", 3706, "One stable EMA key; original folds"],
    ["Missing composite; excluded only from modeling", data["missing_scores"], "All three valid 1–5 items required; no outcome imputation"],
], columns=["stage", "rows", "rule"]).set_index("stage"))""",
        """# Clocks and feature acceptance
rules = [
    ["EMA UTC", "Actual local start minus row local-minus-UTC offset; 3,706 parsed"],
    ["Garmin UTC", "Unix ts unchanged; dte_tme retained separately; discrepancy flags preserved"],
    ["Minute start (1,868 EMAs)", "09:22 → cutoff 09:21 → [09:16, 09:21), synthetic example"],
    ["Second start (1,838 EMAs)", "09:22:35 → [09:17:35, 09:22:35), synthetic example"],
    ["Windows", "Left-inclusive, right-exclusive; 5, 15, 30, 60, 120, 360, 720, 1440 minutes"],
    ["HR", "30–240 bpm; conflicting timestamp groups / bad clocks excluded; no gap filling"],
    ["Mean / SD; slope", "≥2 readings and ≥1 s span; slope ≥3 readings and ≥60 s span"],
    ["Steps / floors", "Original adjacent counter pairs, no reset, gap ≤300 s; whole interval inside window"],
    ["History", "Earlier 24 h excluding current window; ≥30 observed UTC minutes across ≥3 hours"],
    ["PulseOx", "Provisional 70–100 percent-like values; separate sensitivity, not validated units"],
]
display(pd.DataFrame(rules, columns=["item", "implementation"]).set_index("item"))""",
        """# Missingness: feature cells, not unique responses
display(data["missing_modality"].set_index("modality"))
fig = ov.missing_plot(data)
display(fig)
plt.close(fig)
display(FileLink(relpath(root / ov.PUBLIC / "missingness_lookback.csv")))""",
        """# Complete feature inventory and per-feature missingness
features = data["dictionary"]
display(features.pivot_table(index=["status", "modality"], columns="lookback_minutes", values="name", aggfunc="count", fill_value=0))
examples = ["hr__mean__60m", "hr__sd__60m", "hr__latest__60m", "hr__slope__60m", "steps__added_supported__60m"]
display(features[features.name.isin(examples)][["name", "requirements", "responses", "missing_count", "missing_pct"]].set_index("name"))
display(features.groupby("modality").agg(candidate_columns=("name", "size"), distinct_statistics=("metric", "nunique")))
display(pd.DataFrame([
    ["Level / distribution", "mean, median, min/max, SD, variance, percentiles, range, IQR, MAD, CV"],
    ["Timing / change", "first/latest, slope, net/half-window change, original-adjacent changes"],
    ["Shape / observed bins", "skewness, excess kurtosis, observed-minute and rolling-five-minute summaries"],
    ["Activity / history", "supported counter increments, co-observed activity, earlier HR references / deviations"],
], columns=["statistic group", "examples"]).set_index("statistic group"))
display(FileLink(relpath(root / ov.PUBLIC / "complete_feature_inventory.csv")))""",
        """# Modeling matrix and actual training inputs
display(data["shapes"].set_index("dataset"))
display(pd.DataFrame({"complete cases across all 438": [data["complete_supported"]], "missing target": [data["missing_scores"]], "provisional columns in primary models": [0]}))
display(data["folds"][["fold", "training_rows", "validation_rows", "training_participants", "validation_participants", "retained_physiology", "context_numeric", "survey_encoded", "missing_indicators", "actual_inputs"]].set_index("fold"))
display(FileLink(relpath(root / ov.PUBLIC / "actual_model_inputs_by_fold.csv")))""",
        """# Training-only preprocessing and nested tuning
display(pd.DataFrame([
    ["Split", "Five frozen participant folds; three grouped inner folds within each training fold"],
    ["Numeric inputs", "Drop training-all-missing / constant columns; training median imputation; StandardScaler"],
    ["Context", "Actual local time sine/cosine + training-fitted survey one-hot encoder"],
    ["Missingness", "Training-variable missingness flags; separate from physiological measurements"],
    ["Logistic regression (13)", "L2; C ∈ {0.01, 0.1, 1, 10}; select inner macro F1; no outer-fold selection"],
    ["Selection comparisons", "Training-only correlation >0.95 filtering, L1 / elastic regularization; separate branches"],
    ["Binary follow-up (15)", "Same full block; inner AUROC selection; separate from Notebook 13's objective"],
    ["This overview", "Reuses saved fits; recomputes metrics; no new model search"],
], columns=["step", "implementation"]).set_index("step"))
display(data["folds"][["fold", "selected_C", "dropped_numeric"]].set_index("fold"))""",
        """# Score labels and train / held-out development class support
display(pd.DataFrame([
    ["Composite", "nervous + (6 − cope) + (6 − control); integer 3–15; all three valid items"],
    ["Three classes", "0: 3–4; 1: 5–9; 2: 10–15"],
    ["Binary tasks", "≥5 versus ≤4; ≥10 versus ≤9; exploratory, not clinical diagnoses"],
    ["Validation", "Held-out development participants, not the sealed final test"],
], columns=["item", "definition"]).set_index("item"))
display(data["support"])
display(data["support"].filter(like="validation").sum().rename("pooled OOF support").to_frame())""",
        """# Three-class classification: matched 3,689 responses / 125 participants
metrics = ["roc_auc", "macro_f1", "balanced_accuracy", "elevated_precision", "elevated_recall"]
display(data["comparisons"][["view", "responses", "participants"] + metrics].set_index("view").rename(columns={"roc_auc": "macro AUROC (OvR)"}))
fig = ov.confusion_plot(data)
display(fig)
plt.close(fig)""",
        """# Binary classification and matched uncertainty
binary = data["metrics"].query('task != "three_class"')
display(binary[["checkpoint", "task", "view"] + metrics].set_index(["checkpoint", "task", "view"]))
display(data["paired"][["task", "roc_auc_difference", "roc_auc_low", "roc_auc_high", "macro_f1_difference", "macro_f1_low", "macro_f1_high"]].set_index("task"))
display(pd.DataFrame({"reading": ["A = ≥5; B = ≥10. All rows use 3,689 responses / 125 participants.", "Paired 95% participant-bootstrap intervals: full minus context, Notebook 13.", "F1 gains do not establish useful ranking: AUROC difference intervals include zero.", "Intervals condition on fixed OOF predictions and repeated exploratory development."]}))""",
        """# Earlier boosting comparison: separate matched cohort, not a ranking against Notebook 13
display(data["earlier"][["model", "view", "responses", "participants"] + metrics].set_index(["model", "view"]))
display(pd.DataFrame({"scope": ["Notebook 12: 60-minute complete cases, 1,355 responses / 113 participants.", "Only original HR + steps + context; boosting was not fitted to all 438 here.", "Compare models within this table; do not compare their scores directly with the 3,689-row cohort."]}))""",
        """# Continuous-score results: same 3,689 responses / 125 participants
regression = pd.read_csv(root / ov.REVIEW / "regression_metrics.csv")
display(regression.pivot(index="family", columns="view", values="mae").rename_axis(columns="Notebook 14 · OOF MAE"))
display(pd.DataFrame({"interpretation": ["Full = context + 438 supported candidates; no stable improvement over matching median baseline.", "Within-person score variation is not evidence that wearable measurements predict that variation."]}))""",
        """# Recording support and remaining limitations
coverage = data["coverage"].copy()
coverage["accepted_pct"] = 100 * coverage.any_accepted / coverage.responses
coverage["occupied_minute_pct_median"] = 100 * coverage.occupied_fraction_median
display(coverage[["lookback_minutes", "responses", "any_recorded", "any_accepted", "accepted_pct", "occupied_minute_pct_median", "gap_median_seconds", "freshness_median_seconds"]].set_index("lookback_minutes"))
display(pd.DataFrame([
    ["Recording support", "Any reading is not full-window coverage; no interpolation or carry-forward"],
    ["Gap / freshness medians", "Maximum gaps include window edges / empty windows; freshness excludes no-reading rows"],
    ["Missingness", "Median-filled features / missingness flags may reflect recording patterns, not physiology"],
    ["Timing", "Acquisition versus receipt/batching remains unknown; no validated real-time claim"],
    ["Recall", "Morning: since waking; Midday/Endday: since last prompt. Delivery logs unavailable"],
    ["IBI / HRV", "Limited ≥240 s run pilot only; normal-beat status unverified; not among 678 predictors"],
    ["Next evidence", "Collector/SDK version, original payloads, acquisition/receipt clocks, prompt logs, quality flags"],
], columns=["limitation", "status"]).set_index("limitation"))""",
        """# Verification
from garmin_central_overview.verify import verify_inputs
display(pd.Series(verify_inputs(root, data["checks"]), name="verified").to_frame())""",
    ]
    notebook = nb.v4.new_notebook(cells=[nb.v4.new_code_cell(c) for c in cells])
    notebook.metadata.update(
        kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
        garmin_public_aggregate_notebook=True,
    )
    destination = root / "notebooks/21_central_research_overview.ipynb"
    if destination.exists():
        raise FileExistsError(
            "Preserve executed overview; remove only an unvalidated draft explicitly."
        )
    nb.write(notebook, destination)
    return len(cells)


if __name__ == "__main__":
    print(build(Path.cwd()))
