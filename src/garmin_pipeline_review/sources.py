"""Literal canonical source registry for the review notebook."""

SECTIONS = [
    ("Raw sources and participant linkage", ["garmin_inventory.sources", "garmin_inventory.audit"]),
    (
        "Raw-preserving preprocessing and corrected clocks",
        [
            "garmin_preprocessing.common",
            "garmin_preprocessing.provenance",
            "garmin_preprocessing.normalize",
            "garmin_hr_activity.boundaries",
            "garmin_hr_activity.features",
            "garmin_hr_activity.pipeline",
        ],
    ),
    (
        "PSS scoring, labels and context",
        ["garmin_ema_baseline.linkage", "garmin_pss_classification.data"],
    ),
    (
        "Complete candidate feature generation",
        [
            "garmin_comprehensive.features",
            "garmin_comprehensive.dictionary",
            "garmin_comprehensive.extract",
            "garmin_comprehensive.augment",
        ],
    ),
    (
        "Comprehensive logistic training and evaluation",
        [
            "garmin_comprehensive.models",
            "garmin_comprehensive.run",
            "garmin_pss_classification.metrics",
            "garmin_comprehensive.evaluate",
        ],
    ),
    (
        "Full-feature binary logistic workflow",
        [
            "garmin_binary_6hour.models",
            "garmin_binary_6hour.run",
            "garmin_binary_6hour.metrics",
            "garmin_binary_6hour.evaluate",
        ],
    ),
    (
        "Comprehensive regression and training-only selection",
        [
            "garmin_comprehensive_regression.models",
            "garmin_comprehensive_regression.run",
            "garmin_comprehensive_regression.evaluate",
        ],
    ),
    (
        "Notebook 16 bounded experiments",
        [
            "garmin_binary_improvement.design",
            "garmin_binary_improvement.sampling",
            "garmin_binary_improvement.models",
            "garmin_binary_improvement.run",
        ],
    ),
    (
        "Notebook 17 temporal references and gated EMA history",
        [
            "garmin_temporal_personalization.data",
            "garmin_temporal_personalization.history",
            "garmin_temporal_personalization.models",
            "garmin_temporal_personalization.run",
        ],
    ),
    (
        "Notebook 18 measurement and model diagnostics",
        ["garmin_root_cause.data", "garmin_root_cause.models", "garmin_root_cause.run"],
    ),
    (
        "Notebook 19 conditional measurement alignment",
        [
            "garmin_measurement_alignment.metadata",
            "garmin_measurement_alignment.alignment",
            "garmin_measurement_alignment.provider",
            "garmin_measurement_alignment.experiment",
        ],
    ),
]


def source_path(root, module):
    allowed = {m for _, modules in SECTIONS for m in modules}
    if module not in allowed:
        raise ValueError("Unregistered source module")
    return root / "src" / (module.replace(".", "/") + ".py")
