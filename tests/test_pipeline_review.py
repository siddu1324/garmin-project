"""Review packaging and exact-source safeguards; no real data in fixtures."""

from pathlib import Path
import pytest
from garmin_pipeline_review.package import allowed
from garmin_pipeline_review.sources import SECTIONS, source_path


@pytest.mark.parametrize(
    "name",
    [
        "private/a.parquet",
        "data/a.csv",
        "src/__pycache__/x.py",
        "../src/x.py",
        "/src/x.py",
        "config/local.toml",
        "config/local.json",
        "config/credentials.json",
        ".env",
        "estimator.joblib",
    ],
)
def test_archive_rejects_private_or_unlisted_files(name):
    assert not allowed(name)


def test_archive_accepts_only_source_contract():
    for name in [
        "src/garmin_comprehensive/features.py",
        "tests/test_comprehensive.py",
        "config/comprehensive_features.json",
        "uv.lock",
        "notebooks/20_complete_pipeline_for_review.ipynb",
    ]:
        assert allowed(name)


def test_review_includes_both_feature_stages_and_latest_gates():
    modules = [m for _, group in SECTIONS for m in group]
    assert len(modules) == len(set(modules))
    for m in [
        "garmin_comprehensive.extract",
        "garmin_comprehensive.augment",
        "garmin_comprehensive.models",
        "garmin_comprehensive_regression.models",
        "garmin_binary_6hour.models",
        "garmin_measurement_alignment.alignment",
    ]:
        assert m in modules
    with pytest.raises(ValueError):
        source_path(Path.cwd(), "private.credentials")
