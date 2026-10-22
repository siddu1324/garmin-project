"""Bounded grouped tuning with training-only transforms and supervised selection."""

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from garmin_comprehensive.models import CONTEXT, Transform
from garmin_inventory.privacy import AuditError

FAMILIES = ['ridge', 'lasso', 'elastic', 'huber', 'hgb', 'xgb', 'lgbm']
MAIN = ['context', 'original60', 'hr60', 'hr_all', 'hr_activity60',
        'hr_activity_all', 'full', 'full_corr']
TARGET = 'pss_combo'


def specifications():
    out = []

    def add(view, family, cohort='all_valid', **kw):
        block = 'full' if view in ['full_corr', 'full_l1select', 'full_enetselect',
                                   'full_no_flags', 'missingness'] else view
        out.append(dict(name=f'{cohort}__{view}__{family}', view=view, family=family,
                        block=block, cohort=cohort, flags=view != 'full_no_flags', **kw))

    for view, family in itertools.product(MAIN, FAMILIES):
        add(view, family)
    for view, family in itertools.product(
        ['full_l1select', 'full_enetselect', 'pulseox_full', 'full_no_flags'],
        ['ridge', 'hgb', 'xgb', 'lgbm']
    ):
        add(view, family)
    for view, family in itertools.product(['missingness', 'quality'], ['ridge', 'hgb']):
        add(view, family)
    for m, family in itertools.product([5, 15, 30, 120, 360, 720, 1440], ['ridge', 'hgb']):
        add(f'hr_{m}m', family)
    for view, family in itertools.product(['context', 'original60', 'full'], ['ridge', 'hgb']):
        add(view, family, 'complete60')
    for view in ['context', 'original60']:
        add(view, 'fixed_ridge', 'complete60')
    assert len(out) == len({s['name'] for s in out})
    return out


def grid(family, cfg):
    if family in ['ridge', 'fixed_ridge']:
        return [dict(alpha=a) for a in ([1] if family == 'fixed_ridge' else cfg['ridge_alphas'])]
    if family in ['lasso', 'elastic']:
        return [dict(alpha=a) for a in cfg['sparse_alphas']]
    if family == 'huber':
        return [dict(alpha=a, epsilon=e) for a, e in itertools.product(
            cfg['huber_alphas'], cfg['huber_epsilons'])]
    if family == 'hgb':
        return [dict(loss=loss, max_depth=d, max_iter=n) for loss, d, n in itertools.product(
            ['squared_error', 'absolute_error'], cfg['hgb_depths'], cfg['hgb_iterations'])]
    if family == 'xgb':
        return [dict(objective=loss, max_depth=d, n_estimators=n, min_child_weight=w, reg_lambda=r)
                for loss in ['reg:squarederror', 'reg:absoluteerror']
                for d, n, w, r in cfg['xgb_profiles']]
    if family == 'lgbm':
        return [dict(objective=loss, num_leaves=d, n_estimators=n, min_child_samples=w, reg_lambda=r)
                for loss in ['regression', 'regression_l1']
                for d, n, w, r in cfg['lgbm_profiles']]
    raise ValueError('Unknown family.')


def estimator(family, setting, cfg):
    sparse = dict(max_iter=cfg['max_iter_linear'], tol=cfg['linear_tolerance'],
                  selection='cyclic', random_state=cfg['seed'])
    if family in ['ridge', 'fixed_ridge']:
        return Ridge(**setting, solver='cholesky')
    if family == 'lasso':
        return Lasso(**setting, **sparse)
    if family == 'elastic':
        return ElasticNet(**setting, l1_ratio=.5, **sparse)
    if family == 'huber':
        return HuberRegressor(**setting, max_iter=cfg['huber_max_iter'], tol=cfg['huber_tolerance'])
    if family == 'hgb':
        return HistGradientBoostingRegressor(**setting, learning_rate=cfg['boosting_learning_rate'],
            min_samples_leaf=20, l2_regularization=1, early_stopping=False,
            categorical_features=None, random_state=cfg['seed'])
    common = dict(learning_rate=cfg['boosting_learning_rate'], colsample_bytree=.8,
                  subsample=1, n_jobs=1, random_state=cfg['seed'])
    if family == 'xgb':
        return XGBRegressor(**setting, **common, tree_method='hist', max_bin=64, verbosity=0)
    if family == 'lgbm':
        return LGBMRegressor(**setting, **common, max_bin=63, deterministic=True,
                             force_col_wise=True, verbosity=-1)
    raise ValueError('Unknown estimator.')


def fit_checked(model, x, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x, y)
    ok = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, ok, sorted({w.category.__name__ for w in caught})


def predict(model, x):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='X does not have valid feature names')
        p = np.asarray(model.predict(x), float)
    if not np.isfinite(p).all():
        raise AuditError('Nonfinite regression prediction.')
    return p


@dataclass
class Fitted:
    transform: object
    selected: np.ndarray
    selector: object
    model: object

    def predict(self, f):
        return predict(self.model, self.transform.transform(f)[:, self.selected])

    @property
    def names(self):
        return np.array(self.transform.names)[self.selected].tolist()


def prepare(train, valid, spec, columns, cfg):
    transform = Transform(columns, handling='complete' if spec['cohort'] == 'complete60'
                          and spec['block'] in ['context', 'original60'] else 'imputed',
                          flags=spec['flags'] and spec['cohort'] != 'complete60',
                          missing_only=spec['view'] == 'missingness',
                          correlation=cfg['correlation_threshold']
                          if spec['view'] == 'full_corr' else None).fit(train)
    # Full features in complete60 still need imputation; retain its ordinary flags.
    if spec['cohort'] == 'complete60' and spec['block'] == 'full':
        transform = Transform(columns, flags=True).fit(train)
    x, xv = transform.transform(train), transform.transform(valid)
    keep = np.ones(x.shape[1], dtype=bool)
    selector = None
    if spec['view'] in ['full_l1select', 'full_enetselect']:
        selector = ElasticNet(alpha=cfg['selector_alpha'],
            l1_ratio=1 if spec['view'] == 'full_l1select' else .5,
            max_iter=cfg['max_iter_linear'], tol=cfg['linear_tolerance'], selection='cyclic')
        selector, ok, _ = fit_checked(selector, x, train[TARGET].to_numpy(float))
        if not ok:
            raise AuditError('Training-only feature selector did not converge.')
        keep = np.abs(selector.coef_) > cfg['selector_coefficient_threshold']
        keep |= np.array([n in CONTEXT or n.startswith('n_survey_category_')
                          for n in transform.names])
    return transform, selector, keep, x[:, keep], xv[:, keep]


def inner_splits(train, cfg):
    splits = list(GroupKFold(cfg['inner_splits'], shuffle=False).split(
        train, groups=train.n_participant))
    for a, b in splits:
        assert not set(train.iloc[a].n_participant) & set(train.iloc[b].n_participant)
        assert len(a) and len(b) and train.iloc[a].n_participant.nunique() >= 2
    return splits


def nested_fit(train, valid_x, spec, columns, cfg):
    if TARGET in valid_x:
        raise AuditError('Outer validation outcomes must not enter tuning.')
    if set(train.n_participant) & set(valid_x.n_participant):
        raise AuditError('Outer participant leakage.')
    splits = inner_splits(train, cfg)
    prepared = []
    for a, b in splits:
        tr, va = train.iloc[a], train.iloc[b]
        _, _, _, x, xv = prepare(tr, va.drop(columns=TARGET), spec, columns, cfg)
        prepared.append((x, tr[TARGET].to_numpy(float), xv, va[TARGET].to_numpy(float)))
    trials = []
    for setting in grid(spec['family'], cfg):
        maes, warnings_seen = [], []
        ok = True
        for x, y, xv, yv in prepared:
            model, converged, warning_names = fit_checked(estimator(spec['family'], setting, cfg), x, y)
            warnings_seen.extend(warning_names)
            if not converged:
                ok = False
                break
            maes.append(float(np.abs(predict(model, xv) - yv).mean()))
        trials.append(dict(setting=setting, converged=ok, inner_maes=maes,
                           inner_mae=float(np.mean(maes)) if ok else None,
                           warning_categories=sorted(set(warnings_seen))))
    eligible = [t for t in trials if t['converged']]
    if not eligible:
        raise AuditError('All fixed tuning candidates failed convergence.')
    best = min(eligible, key=lambda t: t['inner_mae'])
    transform, selector, keep, x, _ = prepare(train, valid_x, spec, columns, cfg)
    model, converged, warn = fit_checked(estimator(spec['family'], best['setting'], cfg),
                                       x, train[TARGET].to_numpy(float))
    if not converged:
        raise AuditError('Selected outer refit did not converge.')
    fitted = Fitted(transform, keep, selector, model)
    return fitted, best, trials, splits, warn
