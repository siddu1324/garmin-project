"""Synthetic-only boundary, importer, leakage and grouped recovery checks."""

import copy
import json
import numpy as np
import pandas as pd
import pytest
from garmin_inventory.privacy import sha256
from garmin_measurement_alignment.metadata import (
    clock,
    validate_record,
    validate_contract,
    validate_events,
    import_metadata,
    MetadataError,
)
from garmin_measurement_alignment.alignment import aligned_features, reconstruct, versioned_history
from garmin_measurement_alignment.diagnostics import perturb
from garmin_measurement_alignment.synthetic import T, contract, event, parse, fixture
from garmin_measurement_alignment.experiment import evaluate, HR, QUALITY


def test_clock_offsets_precision_and_unix():
    a = clock(dict(value="2030-01-01T09:22", utc_offset_minutes=-240, precision_seconds=60))
    b = clock(dict(value="2030-01-01T13:22Z", precision_seconds=60))
    assert a == b and a["upper"] - a["lower"] == 60
    assert clock(dict(value=T, encoding="unix_seconds", precision_seconds=0))["lower"] == T
    with pytest.raises(MetadataError):
        clock(dict(value="2030-01-01T09:22", precision_seconds=60))
    with pytest.raises(MetadataError):
        clock(dict(value=T, encoding="unix_seconds", precision_seconds=0, utc_offset_minutes=-240))


def test_contract_and_identity():
    c = contract()
    validate_contract(c)
    c["clock_definitions"]["acquired"] = "callback"
    with pytest.raises(MetadataError):
        validate_contract(c)
    _, raw = fixture()
    with pytest.raises(MetadataError):
        validate_record(raw[0], "b" * 64, {"OTHER"}, {})
    e = copy.deepcopy(raw[0])
    e["ema_id"] = "UNKNOWN"
    with pytest.raises(MetadataError):
        validate_record(e, "b" * 64, {"SYNTHETIC_PERSON"}, {})


def test_duplicate_conflict_and_bad_order():
    _, raw = fixture()
    r = validate_record(raw[0], "b" * 64, {"SYNTHETIC_PERSON"}, {})
    physical, logical = validate_events([r, r])
    assert len(physical) == 2 and len(logical) == 1 and physical[1]["import_exact_duplicate"]
    bad = copy.deepcopy(r)
    bad["prompt_id"] = "DIFFERENT"
    with pytest.raises(MetadataError):
        validate_events([r, bad])
    bad = copy.deepcopy(raw[0])
    bad["clocks"]["received"]["value"] = T - 1
    with pytest.raises(MetadataError):
        validate_record(bad, "b" * 64, {"SYNTHETIC_PERSON"}, {})


def test_import_immutable_and_explicit_approval(tmp_path):
    e, raw = fixture()
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "contract.json").write_text(json.dumps(contract()))
    for r in raw:
        r["source_contract_sha256"] = sha256(inbox / "contract.json")
    path = inbox / "events.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in raw + [raw[0]]))
    approval = tmp_path / "approved.json"
    approval.write_text(
        json.dumps(
            dict(
                review_status="approved",
                reviewer="synthetic_test",
                contract_sha256=[sha256(inbox / "contract.json")],
                event_file_sha256=[sha256(path)],
            )
        )
    )
    dev = pd.DataFrame([e])
    dest = import_metadata(inbox, approval, dev, tmp_path / "out")
    assert dest == import_metadata(inbox, approval, dev, tmp_path / "out")
    m = json.loads((dest / "manifest.json").read_text())
    assert m["physical_events"] == len(raw) + 1 and m["logical_events"] == len(raw)
    approval.write_text("{}")
    with pytest.raises(MetadataError):
        import_metadata(inbox, approval, dev, tmp_path / "out")


def test_recall_no_completed_response_substitute_and_latest_receipt():
    ema, raw = fixture()
    events = list(map(parse, raw))
    assert reconstruct(ema, [], contract())["status"] == "BLOCKED"
    assert reconstruct(ema, events, contract())["duration_seconds"] == 3600
    latest = event("LATE", "prompt_delivered", 3500, prompt_id="LATE")
    latest["clocks"]["received"]["value"] = T + 3700
    assert (
        reconstruct(ema, events + [parse(latest)], contract())["reason"]
        == "latest_boundary_not_available"
    )
    assert (
        reconstruct(ema, [e for e in events if e["event_type"] != "coverage"], contract())["status"]
        == "BLOCKED"
    )


def test_current_prompt_rule_and_uncertain_boundary():
    ema, raw = fixture()
    new = event("CURRENT", "prompt_delivered", 3500, prompt_id="CURRENT")
    events = list(map(parse, raw + [new]))
    c = contract()
    assert reconstruct(ema, events, c)["duration_seconds"] == 100
    c["recall_prompt_rule"] = "exclude_current_prompt"
    assert reconstruct(ema, events, c)["duration_seconds"] == 3600
    new["clocks"]["delivered"]["precision_seconds"] = 100
    # Invalid precision is caught before alignment.
    with pytest.raises(MetadataError):
        parse(new)


def test_wake_requires_independent_date():
    ema, raw = fixture()
    ema["n_survey_category"] = "Morning"
    assert (
        reconstruct(ema, list(map(parse, raw)), contract())["reason"]
        == "verified_waking_date_missing"
    )
    ema["verified_local_date"] = "2030-01-01"
    w = event("WAKE", "wake_observed", 0, verified_local_date="2030-01-01")
    cover = copy.deepcopy(raw[1])
    cover["scope"] = "wake"
    assert reconstruct(ema, list(map(parse, [w, cover])), contract())["status"] == "RECONSTRUCTIBLE"


def test_during_post_and_receipt_excluded_and_quality_separate():
    ema, raw = fixture()
    _, features, q = aligned_features(ema, list(map(parse, raw)), contract())
    bad = event(
        "POST",
        "sensor_measurement",
        3600,
        sensor="hr",
        value=230,
        quality="accepted",
        wear="worn",
        source_sha256="e" * 64,
        source_row=1,
    )
    late = copy.deepcopy(bad)
    late["event_id"] = "LATE"
    late["clocks"]["acquired"]["value"] = T + 3599
    late["clocks"]["received"]["value"] = T + 3700
    _, f2, q2 = aligned_features(ema, list(map(parse, raw + [bad, late])), contract())
    assert features == f2 and q2["accepted_hr"] == q["accepted_hr"]
    assert not set(features) & set(q)


def test_counter_rejection_reset_missing_predecessor_breaks():
    ema, raw = fixture()
    raw = [e for e in raw if e["event_type"] != "sensor_measurement" or e.get("source_row", 0) <= 3]
    _, f, q = aligned_features(ema, list(map(parse, raw)), contract())
    assert q["supported_activity_pairs"] == 2
    for e in raw:
        if e.get("sensor") == "steps" and e["source_row"] == 2:
            e["quality"] = "rejected"
    _, f, q = aligned_features(ema, list(map(parse, raw)), contract())
    assert q["supported_activity_pairs"] == 0 and np.isnan(f["aligned_steps_rate"])
    for e in raw:
        if e.get("sensor") == "steps":
            e["quality"] = "accepted"
            e["counter_epoch"] = str(e["source_row"])
    _, _, q = aligned_features(ema, list(map(parse, raw)), contract())
    assert q["supported_activity_pairs"] == 0
    for e in raw:
        if e.get("sensor") == "steps":
            e["counter_epoch"] = "ONE"
            e["previous_event_id"] = "MISSING"
    _, _, q = aligned_features(ema, list(map(parse, raw)), contract())
    assert q["supported_activity_pairs"] == 0


def test_conflicting_hr_and_boundary_precision():
    ema, raw = fixture()
    r = copy.deepcopy(next(e for e in raw if e.get("sensor") == "hr"))
    r["value"] = 130
    r["event_id"] = "CONFLICT"
    r["source_row"] = 999
    _, _, q = aligned_features(ema, list(map(parse, raw + [r])), contract())
    assert q["accepted_hr"] == 58
    last = copy.deepcopy(r)
    last["clocks"]["acquired"]["value"] = T + 3590
    last["clocks"]["acquired"]["precision_seconds"] = 10
    last["clocks"]["received"]["value"] = T + 3595
    _, _, q2 = aligned_features(ema, list(map(parse, raw + [r, last])), contract())
    assert q2["accepted_hr"] == q["accepted_hr"]


def test_uncertainty_diagnostic_never_pulls_post_cutoff():
    t = np.array([T + 50, T + 100, T + 300, T + 350])
    assert perturb(t, T + 300, 5, -60) == (1, 1)
    assert perturb(t, T + 300, 5, 0) == (2, 2)
    assert perturb(t, T + 300, 5, 60) == (2, 2)


def test_versioned_history_current_future_and_edits():
    current, _ = fixture()
    prior = {
        **current,
        "n_ema_id": "PRIOR",
        "n_actual_epoch": T + 100,
        "n_cutoff_epoch": T + 100,
        "pss_combo": 15,
    }
    current["pss_combo"] = 15
    dev = pd.DataFrame([prior, current])
    a = event(
        "V1",
        "answer_submitted",
        200,
        ema_id="PRIOR",
        answer_version=1,
        items=dict(PSS_NERVOUS=1, PSS_COPE=5, PSS_CONTROL=5),
    )
    b = event(
        "V2",
        "answer_submitted",
        3700,
        ema_id="PRIOR",
        answer_version=2,
        items=dict(PSS_NERVOUS=5, PSS_COPE=1, PSS_CONTROL=1),
    )
    h, _ = versioned_history(dev, list(map(parse, [a, b])), current)
    assert h.previous_pss.iloc[0] == 3
    current["pss_combo"] = 3
    h2, _ = versioned_history(dev, list(map(parse, [a, b])), current)
    pd.testing.assert_frame_equal(h, h2)


def test_end_to_end_synthetic_grouped_injected_signal():
    rows = []
    for p in range(30):
        for y in [0, 1]:
            ema, raw = fixture(f"SYNTHETIC_{p}", hr=65 + 50 * y)
            ema["n_fold"] = p % 5
            ema["n_ema_id"] += str(y)
            boundary, f, q = aligned_features(ema, list(map(parse, raw)), contract())
            row = {
                **ema,
                **f,
                **dict(zip(HR, [f["aligned_" + k] for k in HR])),
                **{"coverage_" + k: q[k] for k in [s.removeprefix("coverage_") for s in QUALITY]},
                "alignment_status": boundary["status"],
                "context_available": True,
                "local_time_sin": np.sin(p),
                "local_time_cos": np.cos(p),
            }
            row.update(
                {k: y for k in ["target_composite5", "target_composite10", "target_nervous"]}
            )
            rows.append(row)
    result = evaluate(pd.DataFrame(rows), dict(seed=19, C_grid=[0.1, 1], bootstrap_draws=20))
    metrics = result["metrics"]
    a = metrics[metrics.fold.eq("pooled")]
    assert a.loc[a.model.eq("D_context_aligned"), "auroc"].min() > 0.99
    assert a.loc[a.model.eq("A_context"), "auroc"].max() < 0.6
    assert not result["skips"]
    for _, g in result["predictions"].groupby("task"):
        assert g.groupby("model").n_ema_id.apply(lambda s: tuple(sorted(s))).nunique() == 1
    for _, g in result["membership"].groupby(["target", "outer", "model", "inner"]):
        assert not set(g.loc[g.role.eq("train"), "n_participant"]) & set(
            g.loc[g.role.eq("validation"), "n_participant"]
        )


def test_counter_availability_uses_supported_pairs_not_absent_hr_column():
    from garmin_measurement_alignment.diagnostics import availability_indicator

    g = pd.DataFrame(
        dict(
            sensor=["steps", "floors", "hr", "hr"],
            accepted_readings=[np.nan, np.nan, 0, 3],
            supported_pairs=[2, 0, np.nan, np.nan],
        )
    )
    assert availability_indicator(g).tolist() == [True, False, False, True]


def test_reviewed_provider_entrypoint_preserves_rows_and_blocks_missing_boundaries(tmp_path):
    from garmin_measurement_alignment.provider import build

    ema, raw = fixture()
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "contract.json").write_text(json.dumps(contract()))
    for r in raw:
        r["source_contract_sha256"] = sha256(inbox / "contract.json")
    (inbox / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            dict(
                review_status="approved",
                reviewer="SYNTHETIC_TEST",
                contract_sha256=[sha256(inbox / "contract.json")],
                event_file_sha256=[sha256(inbox / "events.jsonl")],
            )
        )
    )
    morning = {**ema, "n_ema_id": "SYNTHETIC_MORNING", "n_survey_category": "Morning"}
    dev = pd.DataFrame([ema, morning]).assign(pss_combo=3.0)
    output = build(
        tmp_path, inbox, approval, dev, dict(seed=19, C_grid=[0.1, 1], bootstrap_draws=20)
    )
    frame = pd.read_parquet(output / "aligned_development.parquet")
    assert frame.n_ema_id.tolist() == dev.n_ema_id.tolist()
    assert frame.alignment_status.tolist() == ["RECONSTRUCTIBLE", "BLOCKED"]
    assert frame.n_cutoff_epoch.tolist() == dev.n_cutoff_epoch.tolist()
    assert not (output / "predictions.parquet").exists()


def test_incomplete_latest_version_is_not_backfilled():
    current, _ = fixture()
    prior = {
        **current,
        "n_ema_id": "PRIOR",
        "n_actual_epoch": T + 100,
        "n_cutoff_epoch": T + 100,
        "pss_combo": 15,
    }
    current["pss_combo"] = 15
    a = event(
        "V1",
        "answer_submitted",
        200,
        ema_id="PRIOR",
        answer_version=1,
        items=dict(PSS_NERVOUS=1, PSS_COPE=5, PSS_CONTROL=5),
    )
    b = event(
        "V2",
        "answer_submitted",
        300,
        ema_id="PRIOR",
        answer_version=2,
        items=dict(PSS_NERVOUS=None, PSS_COPE=5),
    )
    v = validate_record(b, "b" * 64, {"SYNTHETIC_PERSON"}, {"PRIOR": "SYNTHETIC_PERSON"})
    h, _ = versioned_history(pd.DataFrame([prior, current]), [parse(a), v], current)
    assert np.isnan(h.previous_pss.iloc[0]) and h.q_verified_history_count.iloc[0] == 0
