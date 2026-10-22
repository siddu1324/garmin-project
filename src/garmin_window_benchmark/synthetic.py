"""Clearly synthetic positive control through production clocks, summaries, linkage and models."""

import json

import numpy as np
import pandas as pd

from garmin_ema_baseline.linkage import META, link_outcomes
from garmin_hr_activity.features import CLOCK_FLAGS, prepare, window
from garmin_preprocessing.normalize import parse_ema_clock
from .models import fit_predict


def positive_control(hr_config, model_config):
    features, emas, assignments = [], [], []
    for person in range(30):
        participant = f"synthetic_person_{person:02}"
        assignments.append(dict(participant=participant, partition="development", fold=person % 5))
        for occasion in range(10):
            level = (occasion + person) % 5
            actual = pd.Timestamp("2020-01-01 12:00") + pd.Timedelta(days=occasion)
            raw = pd.DataFrame(
                dict(
                    actual_start_local=[str(actual)],
                    scheduled_start_local=[str(actual)],
                    timezone_offset=["-240"],
                    instance_id=["0"],
                )
            )
            clock = parse_ema_clock(raw).iloc[0].to_dict()
            cutoff = clock["n_cutoff_epoch"]
            times = np.array([cutoff - 3601, cutoff - 3599, cutoff - 1800, cutoff - 1, cutoff])
            values = np.array([230, 60 + 5 * level - 1, 60 + 5 * level, 60 + 5 * level + 1, 230])
            source = pd.DataFrame(
                dict(
                    n_epoch_s=times,
                    n_value=values,
                    n_logical_row=np.arange(5),
                    q_conflicting_timestamp_value=False,
                    **{flag: False for flag in CLOCK_FLAGS},
                )
            )
            g, _ = prepare(source, "hr", hr_config)
            v, _, _ = window(g, cutoff - 3600, cutoff, "hr", hr_config)
            meta = dict(
                n_ema_id=f"synthetic_ema_{person:02}_{occasion:02}",
                n_participant=participant,
                n_partition="development",
                n_fold=person % 5,
                n_survey_category=["Morning", "Midday", "Endday"][occasion % 3],
                n_cutoff_epoch=cutoff,
                n_start_precision=clock["n_start_precision"],
                n_invalid_clock=False,
            )
            features.append({**meta, **{f"hr_{k}__60m": val for k, val in v.items()}})
            emas.append(
                {
                    **meta,
                    **clock,
                    "n_source_columns": json.dumps(["PSS_NERVOUS", "PSS_COPE", "PSS_CONTROL"]),
                    "n_source_sha256": "synthetic_source",
                    "n_source_row": occasion + 1,
                    "PSS_NERVOUS": str(1 + level),
                    "PSS_COPE": str(5 - level),
                    "PSS_CONTROL": str(5 - level),
                }
            )
    ema = pd.DataFrame(emas)
    feature = pd.DataFrame(features)
    assert set(META) <= set(ema)
    linked = link_outcomes(
        ema, feature.sample(frac=1, random_state=1), pd.DataFrame(assignments), expected_rows=300
    )
    errors = {k: [] for k in ["fixed_ridge", "hgb", "median"]}
    for fold in range(5):
        train, valid = linked[linked.n_fold.ne(fold)], linked[linked.n_fold.eq(fold)]
        for family, setting in [
            ("fixed_ridge", {"alpha": 1}),
            ("hgb", {"max_depth": 2, "max_iter": 150}),
        ]:
            p, _, _ = fit_predict(train, valid, "context_hr", family, setting, model_config)
            errors[family].extend(np.abs(p - valid.pss_combo).tolist())
        errors["median"].extend(np.abs(train.pss_combo.median() - valid.pss_combo).tolist())
    result = {
        "synthetic_only": True,
        "responses": 300,
        "participants": 30,
        "outer_folds": 5,
        **{k + "_mae": float(np.mean(v)) for k, v in errors.items()},
    }
    assert result["fixed_ridge_mae"] < 0.1 and result["hgb_mae"] < 0.3 and result["median_mae"] > 3
    return result
