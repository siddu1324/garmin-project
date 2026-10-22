"""Training-only resampling with private, replayable donor lineage.

Standard SMOTE is numeric-only; the separate conditional variant preserves context
and only mixes a person's same-category, same-missing-pattern observations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def resample(x, y, people, categories, missing, method, numeric_indices, seed, k=5):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    counts = np.bincount(y, minlength=2)
    if counts.min() == 0:
        raise ValueError("Both training classes required.")
    minority = int(np.argmin(counts))
    small, large = np.flatnonzero(y == minority), np.flatnonzero(y != minority)
    lineage = pd.DataFrame(columns=["base", "neighbor", "fraction", "synthetic"])
    xr, yr = x.copy(), y.copy()
    if method == "under":
        ix = np.sort(np.r_[small, rng.choice(large, len(small), replace=False)])
        xr, yr = x[ix], y[ix]
        lineage = pd.DataFrame(dict(base=ix, neighbor=ix, fraction=0.0, synthetic=False))
    elif method in ["over", "smote", "conditional"]:
        number = int(counts.max() - counts.min())
        base = rng.choice(small, number, replace=True)
        neighbor = base.copy()
        fraction = np.zeros(number)
        if method != "over" and len(small) > 1 and len(numeric_indices):
            distances = pairwise_distances(x[small][:, numeric_indices])
            np.fill_diagonal(distances, np.inf)
            if method == "conditional":
                # IDs constrain augmentation only; they never enter a predictor.
                ok = people[small, None] == people[None, small]
                ok &= categories[small, None] == categories[None, small]
                ok &= (missing[small, None, :] == missing[None, small, :]).all(axis=2)
                distances[~ok] = np.inf
            nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]
            lookup = {int(v): i for i, v in enumerate(small)}
            for i, b in enumerate(base):
                row = lookup[int(b)]
                choices = nearest[row][np.isfinite(distances[row, nearest[row]])]
                if len(choices):
                    neighbor[i] = small[rng.choice(choices)]
                    fraction[i] = rng.uniform()
        synthetic = x[base].copy()
        synthetic[:, numeric_indices] += fraction[:, None] * (
            x[neighbor][:, numeric_indices] - x[base][:, numeric_indices]
        )
        xr = np.vstack([x, synthetic])
        yr = np.r_[y, np.full(number, minority)]
        lineage = pd.DataFrame(
            dict(base=base, neighbor=neighbor, fraction=fraction, synthetic=fraction > 0)
        )
    elif method not in ["none", "weighted"]:
        raise ValueError("Unspecified sampler.")
    info = dict(
        method=method,
        original_counts=counts.tolist(),
        resampled_counts=np.bincount(yr, minlength=2).tolist(),
        augmented_rows=max(0, len(yr) - len(y)),
        synthetic_rows=0,
        cross_person_rows=0,
        donor_missing_rows=0,
        cross_category_rows=0,
    )
    if len(lineage) and method != "under":
        a, b = lineage.base.to_numpy(int), lineage.neighbor.to_numpy(int)
        genuine = lineage.synthetic.to_numpy(bool)
        info.update(
            synthetic_rows=int(genuine.sum()),
            cross_person_rows=int((genuine & (people[a] != people[b])).sum()),
            cross_category_rows=int((genuine & (categories[a] != categories[b])).sum()),
            donor_missing_rows=int((missing[a].any(axis=1) | missing[b].any(axis=1)).sum()),
        )
        assert np.all(
            synthetic[:, numeric_indices]
            >= np.minimum(x[a][:, numeric_indices], x[b][:, numeric_indices]) - 1e-10
        )
        assert np.all(
            synthetic[:, numeric_indices]
            <= np.maximum(x[a][:, numeric_indices], x[b][:, numeric_indices]) + 1e-10
        )
    return xr, yr, info, lineage


def prior_correct(p, original_prevalence, effective_prevalence):
    """Analytic prior-shift sensitivity; not proof of calibrated probabilities."""
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    ratio = (original_prevalence / (1 - original_prevalence)) / (
        effective_prevalence / (1 - effective_prevalence)
    )
    return ratio * p / (1 - p + ratio * p)
