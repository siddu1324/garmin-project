"""Fixed, matched logistic experiment; callable only after validated alignment."""

import numpy as np
import pandas as pd
from garmin_binary_6hour.metrics import score, bootstrap, interval_fields
from garmin_root_cause.data import HR
from garmin_root_cause.models import fit_model
from .metadata import MetadataError

ALIGNED = ["aligned_" + c for c in HR]
QUALITY = [
    "coverage_occupied_fraction",
    "coverage_largest_gap_seconds",
    "coverage_freshness_seconds",
    "coverage_supported_activity_fraction",
]


def specifications():
    return [
        dict(name=n, columns=c, context=k, missing=False)
        for n, c, k in [
            ("A_context", [], True),
            ("B_original_hr", HR, False),
            ("C_aligned_hr", ALIGNED, False),
            ("D_context_aligned", ALIGNED, True),
            ("E_context_aligned_activity", ALIGNED + ["aligned_steps_rate"], True),
            ("reference_context_original", HR, True),
            ("control_context_coverage", QUALITY, True),
            ("control_context_aligned_coverage", ALIGNED + QUALITY, True),
        ]
    ]


def evaluate(frame, cfg):
    """Private outputs include predictions and membership; no split can bypass the gates."""
    if (
        not frame.n_partition.eq("development").all()
        or frame.groupby("n_participant").n_fold.nunique().max() != 1
    ):
        raise MetadataError("invalid_experiment_partition")
    required = (
        HR + ALIGNED + ["aligned_steps_rate"] + QUALITY + ["local_time_sin", "local_time_cos"]
    )
    mask = (
        frame.alignment_status.eq("RECONSTRUCTIBLE")
        & frame.context_available
        & np.isfinite(frame[required]).all(axis=1)
    )
    predictions = []
    support = []
    trials = []
    members = []
    skips = []
    for target in ["target_composite5", "target_composite10", "target_nervous"]:
        g = frame[mask & frame[target].notna()].sort_values("n_ema_id")
        for fold in range(5):
            tr, va = g[g.n_fold.ne(fold)], g[g.n_fold.eq(fold)]
            assert not set(tr.n_participant) & set(va.n_participant)
            if len(tr) == 0 or len(va) == 0 or tr[target].nunique() != 2:
                skips.append(dict(target=target, fold=fold, reason="infeasible_frozen_fold"))
                continue
            # Stage all models; a failed fit cannot create mismatched comparator membership.
            pending = []
            try:
                for spec in specifications():
                    fit, t, m = fit_model(tr, target, spec, "binary", cfg)
                    pending.append((spec, fit, t, m))
            except ValueError:
                skips.append(
                    dict(target=target, fold=fold, reason="infeasible_inner_fold_or_nonconvergence")
                )
                continue
            for role, z in [("train", tr), ("validation", va)]:
                support.append(
                    dict(
                        target=target,
                        fold=fold,
                        role=role,
                        responses=len(z),
                        participants=z.n_participant.nunique(),
                        positive=int(z[target].sum()),
                        negative=int(z[target].eq(0).sum()),
                    )
                )
            for spec, fit, t, m in pending:
                pred = va[["n_ema_id", "n_participant", "n_fold"]].assign(
                    target=va[target], probability=fit.predict(va), task=target, model=spec["name"]
                )
                predictions.append(pred)
                for trial in t:
                    trials.append(dict(target=target, fold=fold, model=spec["name"], **trial))
                for membership in m:
                    members.append(membership.assign(target=target, outer=fold, model=spec["name"]))
            predictions.append(
                va[["n_ema_id", "n_participant", "n_fold"]].assign(
                    target=va[target],
                    probability=tr[target].mean(),
                    task=target,
                    model="training_prevalence",
                )
            )
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    metrics = []
    differences = []
    calibration = []
    if len(pred):
        for (task, model), z in pred.groupby(["task", "model"]):
            z = z.sort_values("n_ema_id")
            boot = bootstrap(z, cfg)
            ranking = []
            for _, h in z.groupby("n_fold"):
                auc = score(h.target, h.probability)["auroc"]
                if auc is not None:
                    ranking.append((auc, int(h.target.eq(1).sum() * h.target.eq(0).sum())))
            within_auc = (
                sum(a * w for a, w in ranking) / sum(w for _, w in ranking) if ranking else np.nan
            )
            metrics.append(
                dict(
                    task=task,
                    model=model,
                    fold="pooled",
                    within_fold_pair_weighted_auroc=within_auc,
                    responses=len(z),
                    participants=z.n_participant.nunique(),
                    **score(z.target, z.probability),
                    **interval_fields(boot),
                )
            )
            for fold, h in z.groupby("n_fold"):
                metrics.append(
                    dict(
                        task=task,
                        model=model,
                        fold=str(fold),
                        responses=len(h),
                        participants=h.n_participant.nunique(),
                        **score(h.target, h.probability),
                    )
                )
            base = pred[pred.task.eq(task) & pred.model.eq("A_context")].sort_values("n_ema_id")
            assert z.n_ema_id.tolist() == base.n_ema_id.tolist()
            differences.append(
                dict(
                    task=task,
                    model=model,
                    reference="A_context",
                    auroc_difference=score(z.target, z.probability)["auroc"]
                    - score(base.target, base.probability)["auroc"],
                    **interval_fields(boot - bootstrap(base, cfg), "difference_"),
                )
            )
            for b, h in z.groupby(
                pd.cut(z.probability, np.linspace(0, 1, 6), include_lowest=True), observed=True
            ):
                calibration.append(
                    dict(
                        task=task,
                        model=model,
                        bin=str(b),
                        responses=len(h),
                        mean_probability=h.probability.mean(),
                        observed_fraction=h.target.mean(),
                    )
                )
    return dict(
        predictions=pred,
        support=pd.DataFrame(support),
        metrics=pd.DataFrame(metrics),
        differences=pd.DataFrame(differences),
        calibration=pd.DataFrame(calibration),
        trials=trials,
        membership=pd.concat(members, ignore_index=True) if members else pd.DataFrame(),
        skips=skips,
        eligible=mask,
    )
