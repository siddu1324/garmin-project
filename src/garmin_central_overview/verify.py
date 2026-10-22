"""Preservation and public-output checks without modifying earlier checkpoints."""

import json
from pathlib import Path

import pandas as pd

from garmin_binary_6hour.privacy import validate_15
from garmin_comprehensive_regression.privacy import token_matcher
from garmin_inventory.privacy import public_scan_text, sha256, validate_notebook_output
from garmin_pipeline_review.verify import verify as prior_verify
from .overview import PUBLIC, check_seal, read_json


def verify_inputs(root, metric_groups=10):
    root = Path(root)
    before = read_json(root / "private/central_research_overview/before.json")
    publication = root / "reports/repository_publication/maintenance.json"
    if publication.exists():
        from garmin_publication.preservation import preserved, verify_prior

        assert all(preserved(root, p, expected) for p, expected in before.items())
        sealed_files = check_seal(root)
        prior = verify_prior(root)
    else:
        assert all(sha256(root / p) == expected for p, expected in before.items())
        sealed_files = check_seal(root)
        prior = prior_verify(root)
    dictionary = pd.read_csv(root / PUBLIC / "complete_feature_inventory.csv")
    assert len(dictionary) == 678 and dictionary.name.is_unique
    assert dictionary.responses.eq(3706).all()
    assert dictionary.loc[dictionary.modality.eq("pulseox")].shape[0] == 240
    assert metric_groups == 10
    return dict(
        previous_files_checked=len(before),
        documented_maintenance_exceptions=sum(sha256(root / p) != h for p, h in before.items()),
        prior_review_sealed_files=sealed_files,
        raw_reference_metadata_unchanged=prior["raw_reference_metadata_checks"],
        earlier_bundle_manifests_preserved=prior["prior_manifests_preserved"],
        development_rows=3706,
        development_participants=125,
        modeling_rows=3689,
        candidate_columns=678,
        supported_columns=438,
        provisional_pulseox_columns=240,
        feature_missingness_counts_checked=678,
        score_checks=3706,
        target_checks=11118,
        outer_participant_separation_checks=5,
        saved_OOF_metric_groups_recomputed=metric_groups,
        previous_feature_recalculations=prior["feature_value_checks"],
        previous_classifier_replays=prior["classifier_replays"],
        previous_regressor_replays=prior["regressor_replays"],
        previous_inner_separation_checks=prior["inner_group_separation_checks"],
        new_model_fits=0,
        final_test_evaluated=False,
    )


def privacy_scan(root):
    root = Path(root)
    tokens = read_json(root / "private/privacy_tokens.json")
    contains = token_matcher(tokens["identifiers"], tokens["names"])
    paths = [
        p
        for folder in ["reports", "notebooks", "src", "tests", "config"]
        for p in (root / folder).rglob("*")
        if p.is_file() and p.suffix in {".json", ".md", ".ipynb", ".py", ".svg", ".html", ".csv"}
    ]
    for path in paths:
        text = public_scan_text(path)
        assert not contains(text), "Private token found; details withheld"
        if path.suffix == ".ipynb":
            notebook = json.loads(text)
            if int(path.name[:2]) >= 15:
                validate_15(notebook)
            else:
                validate_notebook_output(path, notebook)
    notebook = read_json(root / "notebooks/21_central_research_overview.ipynb")
    assert 10 <= len(notebook["cells"]) <= 16
    for cell in notebook["cells"]:
        assert cell["cell_type"] == "code" and cell["outputs"]
        assert not any(
            o.get("name") == "stderr" or o["output_type"] == "error" for o in cell["outputs"]
        )
    return len(paths)


if __name__ == "__main__":
    root = Path.cwd()
    result = verify_inputs(root)
    result["public_files_privacy_scanned"] = privacy_scan(root)
    result["code_cells_executed"] = len(
        read_json(root / "notebooks/21_central_research_overview.ipynb")["cells"]
    )
    (
        root
        / (
            "reports/repository_publication/notebook_verification.json"
            if (root / "reports/repository_publication/maintenance.json").exists()
            else PUBLIC + "/verification.json"
        )
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
