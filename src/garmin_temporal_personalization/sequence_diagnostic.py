"""Retrospective changes between successive pre-EMA sensor summaries, never predictors.

Added as a descriptive coverage check after the fixed model fits. No selection or
new predictive feature/model results are based on it. Previous starts are not prompts.
"""

import json
from pathlib import Path
import pandas as pd
from garmin_preprocessing.common import write_json
from garmin_comprehensive_regression.evaluate import finite


def describe(root):
    root = Path(root)
    ptr = json.loads((root / "private/temporal_personalization/current.json").read_text())
    out = root / "reports/temporal_personalization" / Path(ptr["run"]).name
    if (out / "manifest.json").exists():
        return out
    f = pd.read_parquet(root / ptr["features"] / "development.parquet")
    pairs = []
    for person, g in f.groupby("n_participant"):
        z = g.sort_values("n_actual_epoch")
        for a, b in zip(z.iloc[:-1].itertuples(), z.iloc[1:].itertuples()):
            if a.n_actual_epoch >= b.n_cutoff_epoch:
                continue
            prev = getattr(a, "hr__mean__60m")
            now = getattr(b, "hr__mean__60m")
            pairs.append(
                dict(
                    person=person,
                    gap=(b.n_cutoff_epoch - a.n_cutoff_epoch) / 3600,
                    nonoverlapping=a.n_cutoff_epoch <= b.n_cutoff_epoch - 3600,
                    hr_change=now - prev,
                    score_change=b.pss_combo - a.pss_combo,
                )
            )
    p = pd.DataFrame(pairs)
    rows = []
    for name, g in [
        ("all_successive_starts", p),
        ("nonoverlapping_hour_windows", p[p.nonoverlapping]),
    ]:
        observed = g.dropna(subset=["hr_change", "score_change"])
        rows.append(
            dict(
                view=name,
                start_ordered_pairs=len(g),
                participants=g.person.nunique(),
                paired_observed_changes=len(observed),
                paired_participants=observed.person.nunique(),
                median_gap_hours=float(g.gap.median()),
                median_absolute_hr_change=float(observed.hr_change.abs().median())
                if len(observed)
                else None,
                retrospective_change_correlation=float(
                    observed.hr_change.corr(observed.score_change)
                )
                if len(observed) > 2
                else None,
                limitation="Current and previous fixed-hour summaries, not full interresponse/recall intervals; unknown answer availability; descriptive only",
            )
        )
    write_json(out / "successive_window_changes.json", finite(rows))
    return out


if __name__ == "__main__":
    print(describe(Path.cwd()))
