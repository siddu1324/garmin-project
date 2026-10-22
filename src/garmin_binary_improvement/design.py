"""Outcome-independent experiment registry and exact feature allowlists."""

from garmin_comprehensive.augment import history_name

WINDOWS = [60, 120, 360, 720, 1440]


def blocks(entries):
    result = {"context": []}
    for m in WINDOWS:

        def hr(s):
            return f"hr__{s}__{m}m"

        result[f"original{m}"] = [hr(s) for s in ["mean", "sd", "latest", "slope"]]
        for b, mods in [
            ("hr", ["hr"]),
            ("activity", ["hr", "steps", "floors"]),
            ("full", ["hr", "steps", "floors", "interaction"]),
        ]:
            result[f"{b}{m}"] = [
                d["name"] for d in entries if d["lookback_minutes"] == m and d["modality"] in mods
            ]
        result[f"minute{m}"] = [
            hr(s) for s in ["observed_minute_mean", "observed_minute_sd", "latest", "slope"]
        ]
        result[f"history{m}"] = result[f"original{m}"] + [
            history_name(s, m) for s in ["reference_mean", "mean_deviation", "latest_deviation"]
        ]
        result[f"quality{m}"] = [
            f"quality__hr__{s}__{m}m"
            for s in [
                "accepted_readings",
                "occupied_fraction",
                "span_seconds",
                "maximum_gap_seconds",
                "latest_age_seconds",
            ]
        ]
    result["temporal360"] = result["original360"] + [
        f"hr__{s}__360m" for s in ["half_mean_change", "observed_minute_sd", "rolling5_within_sd"]
    ]
    result["adjacent360"] = result["original360"] + [
        f"multiscale__adjacent_mean_difference__{m}m" for m in [60, 360, 720]
    ]
    result["interaction360"] = result["original360"] + [
        d["name"]
        for d in entries
        if d["lookback_minutes"] == 360 and d["modality"] == "interaction"
    ]
    result["full_quality360"] = result["full360"] + result["quality360"]
    result["multiscale"] = sum([result[f"original{m}"] for m in [60, 360, 1440]], []) + [
        history_name("reference_mean", 60),
        "multiscale__adjacent_mean_difference__360m",
        "multiscale__adjacent_mean_difference__720m",
    ]
    result["compact360"] = result["original360"] + [
        "steps__supported_rate__360m",
        "hr__observed_minute_sd__360m",
        "interaction__hr_zero_step_mean__360m",
        "hr__half_mean_change__360m",
    ]
    result["wake"] = [f"retrospective__{s}" for s in ["mean", "sd", "latest", "slope"]]
    return result


def specifications():
    rows = []

    def add(name, block, **kw):
        rows.append(
            dict(
                name=name,
                block=block,
                cohort="all",
                mode="main",
                sampler="none",
                context=True,
                compact=False,
                flags=True,
                permutation=None,
                reuse=None,
            )
            | kw
        )

    add("context", "context", reuse="context_u")
    for m in WINDOWS:
        for b in ["original", "hr", "activity", "full", "sparse"]:
            add(
                f"{b}{m}",
                f"full{m}" if b == "sparse" else f"{b}{m}",
                mode="sparse" if b == "sparse" else "main",
                reuse=f"{b}{m}_u" if m <= 360 else None,
            )
    for name in [
        "minute360",
        "minute1440",
        "temporal360",
        "history360",
        "history1440",
        "adjacent360",
        "interaction360",
        "full_quality360",
        "multiscale",
    ]:
        add(name, name)
    add("quality360", "quality360")
    add("missing360", "full360", mode="missing_only", reuse="missing_only360_u")
    add("no_flags360", "full360", flags=False, reuse="no_flags360_u")
    for context in [False, True]:
        for sampler in ["none", "weighted", "over", "under", "conditional" if context else "smote"]:
            add(
                f"{'context_' if context else ''}compact_{sampler}",
                "compact360",
                compact=True,
                context=context,
                flags=False,
                sampler=sampler,
            )
    for block in ["context", "original360", "full360"]:
        add(f"support_{block}", block, cohort="supported")
    for scenario in ["nominal", "later"]:
        for block in ["context", "original60", "wake"]:
            add(f"retro_{scenario}_{block}", block, cohort=f"retro_{scenario}")
    for seed in [7, 19, 31]:
        add(f"permuted_{seed}", "full360", permutation=seed)
    return rows
