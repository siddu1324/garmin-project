"""Train-only transformations and deterministic grouped nested classification."""
import warnings

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from garmin_inventory.privacy import AuditError
from .data import feature_set
from .metrics import macro_f1


class Transform:
    """Feature-level median imputation; no sensor data or outcomes are modified."""

    def __init__(self, view, minutes, handling):
        self.numeric, self.wearable = feature_set(view, minutes)
        self.handling = handling

    def fit(self, frame):
        self.kept = [c for c in self.numeric if frame[c].notna().any()]
        self.dropped = [c for c in self.numeric if c not in self.kept]
        self.medians = frame[self.kept].median()
        x = frame[self.kept].to_numpy(float)
        if self.handling == 'complete' and np.isnan(x).any():
            raise AuditError('Missing feature in complete-case training cohort.')
        x = frame[self.kept].fillna(self.medians).to_numpy(float)
        self.scaler = StandardScaler().fit(x)
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(
            frame[['n_survey_category']])
        self.names = self.kept + list(self.encoder.get_feature_names_out())
        self.groups = {c: [i] for i, c in enumerate(self.kept)}
        self.groups['survey_category'] = list(range(len(self.kept), len(self.names)))
        if self.handling == 'imputed':
            self.groups['recording_missingness'] = list(range(len(self.names), len(self.names) + len(self.wearable)))
            self.names += ['missing__' + c for c in self.wearable]
        return self

    def transform(self, frame):
        if self.handling == 'complete' and frame[self.kept].isna().any().any():
            raise AuditError('Missing feature in complete-case validation cohort.')
        blocks = [self.scaler.transform(frame[self.kept].fillna(self.medians).to_numpy(float)),
                  self.encoder.transform(frame[['n_survey_category']])]
        if self.handling == 'imputed' and self.wearable:
            blocks.append(frame[self.wearable].isna().to_numpy(float))
        x = np.column_stack(blocks)
        if not np.isfinite(x).all():
            raise AuditError('Nonfinite transformed features.')
        return x


def weights(y, k, weighting):
    counts = np.bincount(y, minlength=k)
    if (counts == 0).any():
        raise AuditError('Training set lacks an outcome class.')
    return np.ones(len(y)) if weighting == 'none' else len(y) / (k * counts[y])


def classifier(family, params, k, seed):
    if family == 'logistic':
        return LogisticRegression(**params, solver='lbfgs', max_iter=2000,
                                  random_state=seed)
    if family == 'xgboost':
        return XGBClassifier(**params, objective='multi:softprob' if k == 3 else 'binary:logistic',
            **({'num_class': k} if k == 3 else {}), tree_method='hist', n_jobs=1,
            random_state=seed, verbosity=0, eval_metric='mlogloss' if k == 3 else 'logloss')
    if family == 'lightgbm':
        return LGBMClassifier(**params, objective='multiclass' if k == 3 else 'binary',
            n_jobs=1, random_state=seed, verbosity=-1, deterministic=True,
            force_col_wise=True, subsample_freq=1)
    raise AuditError('Unknown classifier.')


def fit_model(family, params, x, y, k, weighting, seed):
    model = classifier(family, params, k, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x, y, sample_weight=weights(y, k, weighting))
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise AuditError('Classifier failed to converge; no silent iteration change.')
    return model


def probabilities(model, x, k):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='X does not have valid feature names')
        p = model.predict_proba(x)
    if not np.array_equal(model.classes_, np.arange(k)) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise AuditError('Invalid probability/class mapping.')
    return p


def inner_splits(frame, target, k, config):
    splitter = StratifiedGroupKFold(n_splits=config['inner_splits'], shuffle=True,
                                   random_state=config['seed'])
    splits = list(splitter.split(frame, frame[target], frame.n_participant))
    for a, b in splits:
        if set(frame.iloc[a].n_participant) & set(frame.iloc[b].n_participant):
            raise AuditError('Participant leakage in inner folds.')
        if any(len(np.unique(frame.iloc[ix][target])) != k for ix in [a, b]):
            return None
    return splits


def tune(family, prepared, grid, k, weighting, seed):
    scores = []
    for params in grid:
        fold_scores = []
        for x, y, xv, yv in prepared:
            model = fit_model(family, params, x, y, k, weighting, seed)
            p = probabilities(model, xv, k).argmax(axis=1)
            fold_scores.append(macro_f1(yv, p, k))
        scores.append(fold_scores)
    best = int(np.argmax(np.mean(scores, axis=1)))
    return grid[best], scores, best
