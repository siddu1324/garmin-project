"""Availability-gated EMA features. Unknown availability never becomes an assumed delay."""

import numpy as np
import pandas as pd

HISTORY = [
    "previous_pss",
    "previous_nervous",
    "previous_cope",
    "previous_control",
    "previous2_pss",
    "previous3_pss",
    "recent_change",
    "recent_trend_per_hour",
    "historical_mean",
    "historical_sd",
    "hours_since_previous_start",
    "prior_frequency_5",
    "prior_frequency_10",
]


def eligible_history(frame, current):
    required = {"n_available_epoch", "n_availability_verified"}
    if not required <= set(frame):
        return frame.iloc[:0].copy()
    f = frame[
        frame.n_participant.eq(current.n_participant)
        & frame.n_ema_id.ne(current.n_ema_id)
        & frame.n_availability_verified.eq(True)
        & frame.n_available_epoch.lt(current.n_cutoff_epoch)
        & frame.n_available_epoch.ge(frame.n_actual_epoch)
        & frame.n_actual_epoch.lt(current.n_cutoff_epoch)
        & frame.pss_combo.notna()
    ].copy()
    # No arbitrary tie breaking of two distinct answers at the same start.
    return f[~f.n_actual_epoch.duplicated(keep=False)].sort_values("n_actual_epoch")


def build_history(frame):
    rows, lineage = [], []
    for r in frame.itertuples(index=False):
        h = eligible_history(frame, r)
        out = {
            "n_ema_id": r.n_ema_id,
            **dict.fromkeys(HISTORY, np.nan),
            "q_verified_history_count": len(h),
            "q_history_reason": "available" if len(h) else "no_verified_available_prior_answer",
        }
        if len(h):
            last = h.iloc[-1]
            for dest, source in [
                ("previous_pss", "pss_combo"),
                ("previous_nervous", "item_nervous"),
                ("previous_cope", "item_cope"),
                ("previous_control", "item_control"),
            ]:
                out[dest] = last[source]
            for lag in [2, 3]:
                if len(h) >= lag:
                    out[f"previous{lag}_pss"] = h.pss_combo.iloc[-lag]
            out.update(
                historical_mean=h.pss_combo.mean(),
                historical_sd=h.pss_combo.std(),
                hours_since_previous_start=(r.n_cutoff_epoch - last.n_actual_epoch) / 3600,
                prior_frequency_5=h.pss_combo.ge(5).mean(),
                prior_frequency_10=h.pss_combo.ge(10).mean(),
            )
            if len(h) >= 2:
                last3 = h.tail(3)
                x = (last3.n_actual_epoch - last3.n_actual_epoch.iloc[0]) / 3600
                out["recent_change"] = h.pss_combo.iloc[-1] - h.pss_combo.iloc[-2]
                out["recent_trend_per_hour"] = np.polyfit(x, last3.pss_combo, 1)[0]
            lineage.extend(
                {
                    "n_ema_id": r.n_ema_id,
                    "source_ema_id": v.n_ema_id,
                    "source_available_epoch": v.n_available_epoch,
                    "cutoff_epoch": r.n_cutoff_epoch,
                }
                for v in h.itertuples()
            )
        rows.append(out)
    return pd.DataFrame(rows), pd.DataFrame(
        lineage, columns=["n_ema_id", "source_ema_id", "source_available_epoch", "cutoff_epoch"]
    )


def personalized_probability(population_prevalence, previous_targets, strength=5):
    """Fixed Beta-prior shrinkage; caller must provide verified available labels only."""
    y = np.asarray(previous_targets, dtype=float)
    if not np.isin(y, [0, 1]).all() or not 0 <= population_prevalence <= 1:
        raise ValueError("Invalid Bernoulli inputs.")
    return (y.sum() + strength * population_prevalence) / (len(y) + strength)


def chronological_population(frame, heldout_people, fit_epoch):
    """Exclude held-out people and population answers unavailable at fitting time."""
    if not {"n_available_epoch", "n_availability_verified"} <= set(frame):
        return frame.iloc[:0].copy()
    return frame[
        ~frame.n_participant.isin(heldout_people)
        & frame.n_availability_verified.eq(True)
        & frame.n_available_epoch.lt(fit_epoch)
        & frame.n_available_epoch.ge(frame.n_actual_epoch)
        & frame.pss_combo.notna()
    ].copy()


def online_baseline(frame, heldout_people, threshold, minimum_history=5):
    """Verified prequential baseline; real exports currently yield no eligible rows.

    Population prevalence is recomputed from only other people's answers available
    at that individual cutoff. Personal state updates only after verified receipt.
    """
    rows = []
    for r in (
        frame[frame.n_participant.isin(heldout_people)].sort_values("n_cutoff_epoch").itertuples()
    ):
        prior = eligible_history(frame, r)
        population = chronological_population(frame, heldout_people, r.n_cutoff_epoch)
        if len(prior) < minimum_history or not len(population):
            continue
        y = prior.pss_combo.ge(threshold).to_numpy(int)
        rows.append(
            dict(
                n_ema_id=r.n_ema_id,
                n_participant=r.n_participant,
                probability=personalized_probability(population.pss_combo.ge(threshold).mean(), y),
                persistence=int(y[-1]),
                historical_responses=len(prior),
                target_change=float(r.pss_combo - prior.pss_combo.iloc[-1] >= 2)
                if np.isfinite(r.pss_combo)
                else np.nan,
            )
        )
    return pd.DataFrame(rows)
