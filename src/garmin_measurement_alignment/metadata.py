"""Read-only provider import with explicit approval, precision bounds and provenance."""

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle, write_json
from garmin_inventory.privacy import sha256

EVENTS = {
    "prompt_scheduled": "scheduled",
    "prompt_delivered": "delivered",
    "prompt_suppressed": "event",
    "prompt_missed": "event",
    "prompt_dismissed": "event",
    "prompt_unanswered": "event",
    "survey_started": "start",
    "answer_saved": "saved",
    "answer_submitted": "submitted",
    "wake_observed": "wake",
    "sensor_measurement": "acquired",
    "coverage": "coverage_start",
}
ITEMS = ["PSS_NERVOUS", "PSS_COPE", "PSS_CONTROL"]
HASH = re.compile(r"^[0-9a-f]{64}$")


class MetadataError(ValueError):
    """Errors contain rule names only, never participant values."""


def clock(value):
    """A stated precision is a possible-time interval [lower, upper], never invented seconds."""
    if not isinstance(value, dict) or set(value) - {
        "value",
        "encoding",
        "precision_seconds",
        "utc_offset_minutes",
    }:
        raise MetadataError("invalid_clock_schema")
    precision = value.get("precision_seconds")
    if (
        type(precision) not in (int, float)
        or not math.isfinite(precision)
        or not 0 <= precision <= 60
    ):
        raise MetadataError("invalid_precision")
    raw = value.get("value")
    encoding = value.get("encoding", "iso8601")
    if encoding == "unix_seconds":
        if type(raw) not in (int, float) or not math.isfinite(raw) or "utc_offset_minutes" in value:
            raise MetadataError("invalid_unix_clock")
        epoch = float(raw)
    elif encoding == "iso8601":
        if not isinstance(raw, str):
            raise MetadataError("invalid_iso_clock_type")
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            raise MetadataError("invalid_iso_clock") from None
        offset = value.get("utc_offset_minutes")
        if offset is not None and (
            type(offset) not in (int, float) or not math.isfinite(offset) or abs(offset) > 840
        ):
            raise MetadataError("invalid_explicit_offset")
        if dt.tzinfo is None:
            if offset is None:
                raise MetadataError("naive_clock_without_offset")
            dt = (dt - timedelta(minutes=offset)).replace(tzinfo=timezone.utc)
        elif offset is not None and dt.utcoffset().total_seconds() / 60 != offset:
            raise MetadataError("conflicting_utc_offsets")
        epoch = dt.timestamp()
    else:
        raise MetadataError("unsupported_clock_encoding")
    if not math.isfinite(epoch) or not 946684800 <= epoch <= 4102444800:
        raise MetadataError("clock_outside_supported_epoch_range")
    # Precision is truncation width, defined by the approved contract. Zero means exact event time.
    return dict(lower=epoch, upper=epoch + precision, precision=precision)


def validate_contract(contract):
    if contract.get("schema_version") != 1:
        raise MetadataError("unsupported_contract_version")
    if contract.get("precision_convention") != "lower_bound_plus_resolution":
        raise MetadataError("unsupported_precision_convention")
    if contract.get("recall_prompt_rule") not in [
        "last_delivered_before_cutoff",
        "exclude_current_prompt",
    ]:
        raise MetadataError("recall_prompt_rule_unverified")
    for key in ["clock_definitions", "collector", "hr_acceptance", "steps"]:
        if not isinstance(contract.get(key), dict):
            raise MetadataError("missing_contract_section")
    if contract["clock_definitions"].get("received") != "server_received_this_exact_event_version":
        raise MetadataError("receipt_semantics_unverified")
    if contract["clock_definitions"].get("acquired") != "sensor_acquisition_time":
        raise MetadataError("acquisition_semantics_unverified")
    for key in ["api_sdk", "api_sdk_version", "collector_version", "device_model", "firmware"]:
        if (
            not isinstance(contract["collector"].get(key), str)
            or not contract["collector"][key].strip()
        ):
            raise MetadataError("missing_collector_identity")
    if contract["hr_acceptance"].get("accepted_quality") != ["accepted"] or contract[
        "hr_acceptance"
    ].get("accepted_wear") != ["worn"]:
        raise MetadataError("unsupported_quality_contract")
    if contract["steps"] != {
        "units": "steps",
        "mode": "cumulative",
        "reset_rule": "explicit_epoch_and_negative_change_break",
        "maximum_pair_gap_seconds": 300,
    }:
        raise MetadataError("unsupported_counter_contract")


def validate_record(raw, contract_hash, participants, ema_lookup):
    if not isinstance(raw, dict):
        raise MetadataError("event_not_object")
    # Gate identity before interpreting clocks, outcomes or sensor values.
    person = raw.get("participant_id")
    if not isinstance(person, str) or person not in participants:
        raise MetadataError("nondevelopment_or_unknown_participant")
    if raw.get("source_contract_sha256") != contract_hash:
        raise MetadataError("contract_hash_mismatch")
    typ = raw.get("event_type")
    if typ not in EVENTS or not isinstance(raw.get("event_id"), str) or not raw["event_id"]:
        raise MetadataError("unknown_event_or_missing_stable_id")
    if not HASH.fullmatch(str(raw.get("payload_sha256", ""))):
        raise MetadataError("missing_payload_hash")
    ema = raw.get("ema_id")
    if ema is not None and (ema not in ema_lookup or ema_lookup[ema] != person):
        raise MetadataError("ambiguous_or_wrong_ema_join")
    if typ in ["survey_started", "answer_saved", "answer_submitted"] and ema is None:
        raise MetadataError("missing_ema_key")
    if typ.startswith("prompt_") and not isinstance(raw.get("prompt_id"), str):
        raise MetadataError("missing_prompt_key")
    raw_clocks = raw.get("clocks", {})
    if (
        not isinstance(raw_clocks, dict)
        or EVENTS[typ] not in raw_clocks
        or "received" not in raw_clocks
    ):
        raise MetadataError("required_event_clock_missing")
    parsed = {name: clock(v) for name, v in raw_clocks.items()}
    if set(parsed) - {
        "scheduled",
        "delivered",
        "event",
        "start",
        "saved",
        "submitted",
        "received",
        "wake",
        "acquired",
        "callback",
        "uploaded",
        "coverage_start",
        "coverage_end",
    }:
        raise MetadataError("unsupported_clock_name")

    def precedes(a, b):
        if a in parsed and b in parsed and parsed[a]["upper"] > parsed[b]["lower"]:
            raise MetadataError("inconsistent_or_ambiguous_clock_order")

    for a, b in [
        (EVENTS[typ], "received"),
        ("scheduled", "delivered"),
        ("start", "saved"),
        ("saved", "submitted"),
        ("submitted", "received"),
        ("acquired", "callback"),
        ("callback", "received"),
        ("received", "uploaded"),
        ("coverage_start", "coverage_end"),
    ]:
        precedes(a, b)
    if typ in ["answer_saved", "answer_submitted"]:
        if type(raw.get("answer_version")) is not int or raw["answer_version"] < 1:
            raise MetadataError("invalid_answer_version")
        items = raw.get("items")
        if (
            not isinstance(items, dict)
            or set(items) - set(ITEMS)
            or any(
                v is not None and (type(v) is not int or v not in range(1, 6))
                for v in items.values()
            )
        ):
            raise MetadataError("invalid_answer_version_items")
        if "submitted" not in parsed:
            # Save-only versions remain preserved but cannot open submitted-answer history.
            pass
    if typ == "sensor_measurement":
        if (
            raw.get("sensor") not in ["hr", "steps"]
            or type(raw.get("value")) not in (int, float)
            or not math.isfinite(raw["value"])
        ):
            raise MetadataError("invalid_sensor_measurement")
        if (
            not HASH.fullmatch(str(raw.get("source_sha256", "")))
            or type(raw.get("source_row")) is not int
            or raw["source_row"] < 1
        ):
            raise MetadataError("missing_export_row_provenance")
        if raw.get("quality") not in ["accepted", "rejected", "unknown"] or raw.get("wear") not in [
            "worn",
            "not_worn",
            "unknown",
        ]:
            raise MetadataError("missing_quality_or_wear_annotation")
        if raw["sensor"] == "steps" and not isinstance(raw.get("counter_epoch"), str):
            raise MetadataError("missing_counter_reset_epoch")
        if raw["sensor"] == "steps" and (
            "previous_event_id" not in raw
            or raw["previous_event_id"] is not None
            and not isinstance(raw["previous_event_id"], str)
        ):
            raise MetadataError("missing_original_counter_adjacency")
    if typ == "coverage":
        if (
            raw.get("scope") not in ["prompt", "wake"]
            or raw.get("complete") is not True
            or "coverage_end" not in parsed
        ):
            raise MetadataError("invalid_coverage_attestation")
        if parsed["coverage_start"]["lower"] >= parsed["coverage_end"]["lower"]:
            raise MetadataError("empty_coverage_attestation")
        # Received may follow coverage end: historical proof of completeness is distinct from event availability.
    return {**raw, "parsed_clocks": parsed}


def validate_events(records):
    """Exact retries survive physically; conflicting stable IDs and versions fail the import."""
    first = {}
    logical = []
    physical = []
    versions = {}
    prompts = {}
    sources = {}
    for index, r in enumerate(records):
        key = (r["participant_id"], r["event_id"])
        content = digest_object({k: v for k, v in r.items() if not k.startswith("import_")})
        if key in first and first[key][0] != content:
            raise MetadataError("conflicting_duplicate_event_id")
        duplicate = key in first
        physical.append({**r, "import_exact_duplicate": duplicate})
        if duplicate:
            continue
        first[key] = (content, index)
        logical.append(r)
        if r["event_type"].startswith("answer_"):
            key = (r["participant_id"], r["ema_id"], r["answer_version"])
            if key in versions and versions[key] != r["items"]:
                raise MetadataError("conflicting_answer_version")
            versions[key] = r["items"]
        if r["event_type"] == "prompt_delivered":
            key = (r["participant_id"], r["prompt_id"])
            payload = (r["parsed_clocks"]["delivered"], r.get("ema_id"))
            if key in prompts and prompts[key] != payload:
                raise MetadataError("ambiguous_prompt_delivery")
            prompts[key] = payload
        if r["event_type"] == "sensor_measurement":
            key = (r["participant_id"], r["source_sha256"], r["source_row"])
            payload = (r["sensor"], r["value"], r["parsed_clocks"]["acquired"])
            if key in sources and sources[key] != payload:
                raise MetadataError("conflicting_payload_export_join")
            sources[key] = payload
    for person in {r["participant_id"] for r in logical}:
        answers = [
            r
            for r in logical
            if r["participant_id"] == person and r["event_type"] == "answer_submitted"
        ]
        for ema in {r["ema_id"] for r in answers}:
            ordered = sorted(
                [r for r in answers if r["ema_id"] == ema], key=lambda x: x["answer_version"]
            )
            for a, b in zip(ordered, ordered[1:]):
                if (
                    a["answer_version"] == b["answer_version"]
                    or a["parsed_clocks"]["received"]["upper"]
                    > b["parsed_clocks"]["received"]["lower"]
                ):
                    raise MetadataError("ambiguous_version_order")
    return physical, logical


def import_metadata(inbox, approval, development, destination_parent):
    """Only hashes explicitly supplied by an external review registry authorize ingestion."""
    inbox, approval, destination_parent = Path(inbox), Path(approval), Path(destination_parent)
    approved = json.loads(approval.read_text())
    if approved.get("review_status") != "approved" or not approved.get("reviewer"):
        raise MetadataError("source_review_not_approved")
    contract_path = inbox / "contract.json"
    contract_hash = sha256(contract_path)
    if contract_hash not in approved.get("contract_sha256", []):
        raise MetadataError("unapproved_contract_hash")
    contract = json.loads(contract_path.read_text())
    validate_contract(contract)
    if not development.n_partition.eq("development").all() or not development.n_ema_id.is_unique:
        raise MetadataError("invalid_development_registry")
    participants = set(development.n_participant)
    lookup = development.set_index("n_ema_id").n_participant.to_dict()
    paths = sorted(inbox.glob("*.jsonl"))
    if not paths:
        raise MetadataError("no_event_exports")
    records = []
    hashes = {}
    for path in paths:
        source_hash = sha256(path)
        hashes[path.name] = source_hash
        if source_hash not in approved.get("event_file_sha256", []):
            raise MetadataError("unapproved_event_file_hash")
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                raise MetadataError("blank_event_line")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                raise MetadataError("invalid_json_event") from None
            parsed = validate_record(raw, contract_hash, participants, lookup)
            parsed.update(
                import_source_sha256=source_hash, import_source_row=n, import_raw_json=line
            )
            records.append(parsed)
    physical, logical = validate_events(records)
    indexed = development.set_index("n_ema_id")
    if "n_actual_epoch" in indexed:
        for r in logical:
            if r["event_type"] in ["answer_saved", "answer_submitted"]:
                start = float(indexed.loc[r["ema_id"], "n_actual_epoch"])
                for field in [EVENTS[r["event_type"]], "received"]:
                    if r["parsed_clocks"][field]["lower"] < start:
                        raise MetadataError("answer_event_precedes_original_start")
    identity = dict(
        contract_sha256=contract_hash,
        approval_sha256=sha256(approval),
        source_hashes=hashes,
        source_code_sha256=sha256(Path(__file__)),
        development_registry_sha256=digest_object(
            development[["n_ema_id", "n_participant", "n_fold"]].to_dict("records")
        ),
    )
    dest = destination_parent / digest_object(identity)[:16]
    destination_parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        verify_bundle(dest)
        return dest
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix=".building_", dir=destination_parent))
    # JSON keeps optional/nested originals without coercing types or discarding provider fields.
    for name, rows in [("physical_events", physical), ("logical_events", logical)]:
        (tmp / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        )
    write_json(tmp / "contract.json", contract)
    write_json(tmp / "identity.json", identity)
    publish_bundle(
        tmp,
        dest,
        dict(
            identity=identity,
            physical_events=len(physical),
            logical_events=len(logical),
            final_test_events=0,
        ),
    )
    return dest
