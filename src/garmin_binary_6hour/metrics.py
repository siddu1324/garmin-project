"""Binary ranking metrics and exact weighted participant bootstrap arithmetic."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def score(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    cm = confusion_matrix(y, p >= 0.5, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    def divide(a, b):
        return float(a / b) if b else 0.0

    sensitivity, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    return dict(
        auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        ap=float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        balanced_accuracy=(sensitivity + specificity) / 2 if len(np.unique(y)) == 2 else None,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=divide(tp, tp + fp),
        f1=divide(2 * tp, 2 * tp + fp + fn),
        negative_precision=divide(tn, tn + fn),
        negative_f1=divide(2 * tn, 2 * tn + fp + fn),
        accuracy=divide(tp + tn, len(y)),
        brier=float(np.mean((p - y) ** 2)),
        probability_sd=float(p.std()),
        positive_support=int(y.sum()),
        negative_support=int((y == 0).sum()),
        predicted_positive=int((p >= 0.5).sum()),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def weighted_ranking(y, p, weights):
    """One row per draw; exact tie-aware AUROC and noninterpolated AP."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    pos = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    neg = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    cp, cn = np.cumsum(pos, axis=1), np.cumsum(neg, axis=1)
    totalp, totaln = cp[:, -1], cn[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (pos * (totaln[:, None] - cn + 0.5 * neg)).sum(axis=1) / (totalp * totaln)
        precision = np.divide(cp, cp + cn, out=np.zeros_like(cp, dtype=float), where=cp + cn > 0)
        ap = (pos * precision).sum(axis=1) / totalp
    ap[(totalp == 0) | (totaln == 0)] = np.nan
    return auc, ap


def bootstrap(f, cfg):
    people, inv = np.unique(f.n_participant, return_inverse=True)
    y, p = f.target.to_numpy(int), f.probability.to_numpy(float)
    rng = np.random.default_rng(cfg["seed"])
    multiplicities = rng.multinomial(
        len(people), np.full(len(people), 1 / len(people)), cfg["bootstrap_draws"]
    )
    parts = []
    for counts in np.array_split(multiplicities, max(1, len(multiplicities) // 100)):
        w = counts[:, inv].astype(float)
        auc, ap = weighted_ranking(y, p, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            sen = (w * ((p >= 0.5) & (y == 1))).sum(axis=1) / (w * y).sum(axis=1)
            spe = (w * ((p < 0.5) & (y == 0))).sum(axis=1) / (w * (1 - y)).sum(axis=1)
        parts.append(np.column_stack([auc, ap, (sen + spe) / 2]))
    return np.concatenate(parts)


def interval_fields(values, prefix=""):
    result = {}
    for i, name in enumerate(["auroc", "ap", "balanced_accuracy"]):
        good = values[:, i][np.isfinite(values[:, i])]
        lo, hi = np.quantile(good, [0.025, 0.975]) if len(good) else [np.nan, np.nan]
        result.update(
            {
                prefix + name + "_low": float(lo),
                prefix + name + "_high": float(hi),
                prefix + name + "_valid_draws": len(good),
            }
        )
    return result
