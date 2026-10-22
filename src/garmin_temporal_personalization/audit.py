"""Aggregate evidence and retrospective diagnostics; no history predictors manufactured."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from garmin_binary_6hour.metrics import score
from garmin_comprehensive_regression.evaluate import finite
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import write_json
from .data import data, RAW, PERSONAL, EXTRA


def alpha(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.var(x.sum(axis=1), ddof=1) == 0:
        return None
    return float(
        x.shape[1]
        / (x.shape[1] - 1)
        * (1 - np.var(x, axis=0, ddof=1).sum() / np.var(x.sum(axis=1), ddof=1))
    )


def run(root):
    root = Path(root)
    out = root / "reports/temporal_personalization/audit"
    if (root / "reports/temporal_personalization/completion.json").exists():
        seal = json.loads((root / "reports/temporal_personalization/completion.json").read_text())
        assert all(sha256(root / p) == h for p, h in seal["presentation_sha256"].items())
        return out
    out.mkdir(exist_ok=True)
    f, cfg = data(root)
    source = root / cfg["normalized_bundle"] / "ema/development_ema.parquet"
    schema = pq.read_schema(source).names
    available = [
        s
        for s in schema
        if any(
            k in s.lower()
            for k in ["submit", "receipt", "available", "finish", "completion", "end_time"]
        )
    ]
    e = pd.read_parquet(
        source,
        columns=["actual_start_local", "timezone_offset", "n_start_precision", "n_cutoff_epoch"],
    )
    clock_checks = 0
    for r in e.itertuples():
        actual = None
        for fmt in ["%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
            try:
                actual = datetime.strptime(r.actual_start_local, fmt)
                break
            except ValueError:
                pass
        if actual is None:
            raise ValueError("Unrecognized local clock; values withheld.")
        offset = float(r.timezone_offset)
        # Raw offset is in minutes, confirmed against the existing parser.
        utc = (actual - timedelta(minutes=offset)).replace(tzinfo=timezone.utc)
        cutoff = utc - timedelta(seconds=60 if r.n_start_precision == "minute" else 0)
        assert abs(cutoff.timestamp() - r.n_cutoff_epoch) < 1e-5
        clock_checks += 1
    items = f[["item_nervous", "item_cope", "item_control"]].copy()
    oriented = items.copy()
    oriented.iloc[:, 1:] = 6 - oriented.iloc[:, 1:]
    distribution = []
    variation = []
    for name, values in [("composite", f.pss_combo), *[(c, oriented[c]) for c in oriented]]:
        z = pd.DataFrame({"person": f.n_participant, "value": values}).dropna()
        center = z.groupby("person").value.transform("mean")
        total = float(((z.value - z.value.mean()) ** 2).sum())
        within = float(((z.value - center) ** 2).sum())
        variation.append(
            dict(
                measure=name,
                responses=len(z),
                participants=z.person.nunique(),
                within_fraction=within / total,
                between_fraction=1 - within / total,
                persons_with_variation=int(z.groupby("person").value.nunique().gt(1).sum()),
            )
        )
        for v, n in z.value.value_counts().sort_index().items():
            distribution.append(
                dict(measure=name, value=float(v), responses=int(n), fraction=n / len(z))
            )
    complete = oriented.dropna()
    centered = complete - complete.groupby(f.loc[complete.index, "n_participant"]).transform("mean")
    consistency = dict(
        complete_responses=len(complete),
        alpha=alpha(complete),
        person_centered_alpha=alpha(centered),
        limitation="Descriptive internal covariance, not test-retest reliability or validated composite/clinical change.",
    )
    # Start-ordered labels are retrospective only; no answer availability inferred.
    pairs = []
    potential = []
    for person, g in f.groupby("n_participant"):
        g = g.sort_values("n_actual_epoch")
        ambiguous = g.n_actual_epoch.duplicated(keep=False)
        z = g[~ambiguous & g.pss_combo.notna()]
        potential.append(
            dict(
                n_participant=person,
                responses=len(z),
                later_after_five=max(0, len(z) - 5),
                ambiguous_starts=int(ambiguous.sum()),
            )
        )
        for a, b in zip(z.iloc[:-1].itertuples(), z.iloc[1:].itertuples()):
            pairs.append(
                dict(
                    n_participant=person,
                    previous=a.pss_combo,
                    current=b.pss_combo,
                    gap_hours=(b.n_actual_epoch - a.n_actual_epoch) / 3600,
                    change=b.pss_combo - a.pss_combo,
                    survey=b.n_survey_category,
                )
            )
    p = pd.DataFrame(pairs)
    pot = pd.DataFrame(potential)
    transitions = []
    for label, cut in [("A", 5), ("B", 10)]:
        for prev in [0, 1]:
            z = p[p.previous.ge(cut).eq(bool(prev))]
            transitions.append(
                dict(
                    task=label,
                    previous_class=prev,
                    pairs=len(z),
                    participants=z.n_participant.nunique(),
                    current_positive=int(z.current.ge(cut).sum()),
                    positive_fraction=float(z.current.ge(cut).mean()),
                )
            )
    person_mean = f.groupby("n_participant").pss_combo.mean()
    centered_previous = p.previous - p.n_participant.map(person_mean)
    centered_current = p.current - p.n_participant.map(person_mean)
    longitudinal = dict(
        retrospective_pairs=len(p),
        participants=p.n_participant.nunique(),
        lag1_correlation=float(p.previous.corr(p.current)),
        retrospective_person_centered_lag1=float(centered_previous.corr(centered_current)),
        median_gap_hours=float(p.gap_hours.median()),
        increase_at_least_two=int(p.change.ge(2).sum()),
        decrease_at_least_two=int(p.change.le(-2).sum()),
        potential_later_responses=int(pot.later_after_five.sum()),
        potential_calibratable_people=int(pot.later_after_five.gt(0).sum()),
        verified_history_responses=0,
        verified_taskB_responses=0,
        clock_checks=clock_checks,
        candidate_availability_fields=available,
        unavailable_reason="No explicit completion/submission or answer-receipt timestamp",
        source_schema_sha256=sha256(source),
        ambiguous_starts=int(pot.ambiguous_starts.sum()),
    )
    prior = pd.read_parquet(root / cfg["prior_run"] / "all_predictions.parquet")
    prior_metrics = []
    folds = []
    for (task, model), g in prior.groupby(["task", "model"]):
        numerator = denominator = 0
        for fold, z in g.groupby("n_fold"):
            s = score(z.target, z.probability)
            pairs_count = int(z.target.sum() * (len(z) - z.target.sum()))
            if s["auroc"] is not None:
                numerator += s["auroc"] * pairs_count
                denominator += pairs_count
            folds.append(
                dict(
                    task=task,
                    model=model,
                    fold=int(fold),
                    responses=len(z),
                    prevalence=float(z.target.mean()),
                    mean_probability=float(z.probability.mean()),
                    **s,
                )
            )
        pooled = roc_auc_score(g.target, g.probability)
        total = int(g.target.sum() * (len(g) - g.target.sum()))
        prior_metrics.append(
            dict(
                task=task,
                model=model,
                responses=len(g),
                pooled_auroc=pooled,
                within_fold_pair_weighted_auroc=numerator / denominator,
                cross_fold_pair_auroc=(pooled * total - numerator) / (total - denominator),
                cross_fold_pair_fraction=(total - denominator) / total,
            )
        )
    feature_status = [
        dict(feature=c, status="available_saved_observed_feature")
        for c in dict.fromkeys(RAW + PERSONAL)
    ]
    feature_status += [
        dict(feature=c, status="new_cached_ledger_summary_provisional_clock") for c in EXTRA
    ]
    feature_status += [
        dict(feature=c, status="blocked", reason=reason)
        for c, reason in [
            ("EMA history/online personalization", "No verified answer availability timestamp"),
            (
                "Sleep duration/fragmentation",
                "No validated sleep stream; current Morning answers unavailable before cutoff",
            ),
            ("IBI HRV", "Unresolved interval acquisition/anchoring/quality"),
            ("Respiration/PulseOx", "Unsupported stream / provisional measurement quality"),
            ("Prior-prompt recall windows", "No full delivered-prompt log"),
        ]
    ]
    for name, obj in [
        ("item_distribution", distribution),
        ("variation", variation),
        ("consistency", consistency),
        ("longitudinal", longitudinal),
        ("transitions_retrospective", transitions),
        ("prior_pooling", prior_metrics),
        ("prior_fold_calibration", folds),
        ("feature_status", feature_status),
    ]:
        write_json(out / f"{name}.json", finite(obj))
    return out


if __name__ == "__main__":
    print(run(Path.cwd()))
