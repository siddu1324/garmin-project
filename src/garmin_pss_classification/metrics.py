"""Fixed-label classification scoring and paired participant uncertainty."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion(y, prediction, k, weights=None):
    out = np.zeros((k, k), float)
    np.add.at(out, (np.asarray(y, int), np.asarray(prediction, int)),
              1. if weights is None else np.asarray(weights, float))
    return out


def from_confusion(cm):
    cm = np.asarray(cm, float)
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
    total = support.sum(axis=-1)
    return dict(macro_f1=f1.mean(axis=-1), weighted_f1=(f1 * support).sum(axis=-1) / total,
                accuracy=tp.sum(axis=-1) / total,
                balanced_accuracy=recall.sum(axis=-1) / (support > 0).sum(axis=-1),
                precision=precision, recall=recall, f1=f1, support=support)


def macro_f1(y, p, k):
    return float(from_confusion(confusion(y, p, k))['macro_f1'])


def score(y, p, probability, groups, k):
    cm = confusion(y, p, k)
    summary = from_confusion(cm)
    values = {key: float(summary[key]) for key in ['macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy']}
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    equal = from_confusion(confusion(y, p, k, 1. / counts[inverse]))
    values['equal_participant_macro_f1'] = float(equal['macro_f1'])
    values['equal_participant_accuracy'] = float(equal['accuracy'])
    values['roc_auc'] = float(roc_auc_score(y, probability[:, 1] if k == 2 else probability,
        multi_class='ovr', average='macro')) if len(np.unique(y)) == k else None
    positive = np.asarray(y) == k - 1
    values['elevated_pr_auc'] = float(average_precision_score(positive, probability[:, k - 1])) if len(np.unique(positive)) == 2 else None
    values['elevated_precision'] = float(summary['precision'][-1])
    values['elevated_recall'] = float(summary['recall'][-1])
    values['elevated_f1'] = float(summary['f1'][-1])
    values['probability_sd'] = float(probability[:, -1].std())
    values['predicted_elevated_fraction'] = float((np.asarray(p) == k - 1).mean())
    return values, cm.astype(int), summary


def paired_bootstrap(y, model, baseline, participants, k, draws=2000, seed=20261015):
    groups = np.unique(participants)
    a = np.stack([confusion(y[participants == g], model[participants == g], k) for g in groups])
    b = np.stack([confusion(y[participants == g], baseline[participants == g], k) for g in groups])
    sample = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    delta = from_confusion(a[sample].sum(axis=1))['macro_f1'] - from_confusion(b[sample].sum(axis=1))['macro_f1']
    low, high = np.quantile(delta, [.025, .975])
    return dict(macro_f1_difference=macro_f1(y, model, k) - macro_f1(y, baseline, k),
                difference_low=float(low), difference_high=float(high),
                bootstrap_participants=len(groups), bootstrap_draws=draws)
