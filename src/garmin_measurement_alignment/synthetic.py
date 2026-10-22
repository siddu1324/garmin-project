"""Explicitly synthetic provider fixture; never used to fill missing real evidence."""

from .metadata import clock

T = 1893456000.0


def contract():
    return dict(
        schema_version=1,
        precision_convention="lower_bound_plus_resolution",
        recall_prompt_rule="last_delivered_before_cutoff",
        clock_definitions=dict(
            received="server_received_this_exact_event_version", acquired="sensor_acquisition_time"
        ),
        collector={
            k: "SYNTHETIC"
            for k in ["api_sdk", "api_sdk_version", "collector_version", "device_model", "firmware"]
        },
        hr_acceptance=dict(accepted_quality=["accepted"], accepted_wear=["worn"]),
        steps=dict(
            units="steps",
            mode="cumulative",
            reset_rule="explicit_epoch_and_negative_change_break",
            maximum_pair_gap_seconds=300,
        ),
    )


def event(name, typ, seconds, person="SYNTHETIC_PERSON", **extra):
    fields = {
        "prompt_delivered": "delivered",
        "wake_observed": "wake",
        "sensor_measurement": "acquired",
        "coverage": "coverage_start",
        "answer_submitted": "submitted",
        "survey_started": "start",
        "answer_saved": "saved",
    }
    value = dict(value=T + seconds, encoding="unix_seconds", precision_seconds=0)
    clocks = {
        fields[typ]: value,
        "received": dict(value=T + seconds + 1, encoding="unix_seconds", precision_seconds=0),
    }
    return dict(
        event_id=name,
        event_type=typ,
        participant_id=person,
        payload_sha256="a" * 64,
        source_contract_sha256="b" * 64,
        clocks=clocks,
        **extra,
    )


def parse(e):
    return {**e, "parsed_clocks": {k: clock(v) for k, v in e["clocks"].items()}}


def fixture(person="SYNTHETIC_PERSON", hr=80.0):
    ema = dict(
        n_ema_id=person + "_EMA",
        n_participant=person,
        n_partition="development",
        n_fold=0,
        n_survey_category="Midday",
        n_cutoff_epoch=T + 3600,
        n_actual_epoch=T + 3660,
        provider_prompt_id="CURRENT",
    )
    boundary = event("PROMPT", "prompt_delivered", 0, person, prompt_id="PREVIOUS")
    cover = event("COVER", "coverage", -100, person, scope="prompt", complete=True)
    cover["clocks"]["coverage_end"] = dict(
        value=T + 3700, encoding="unix_seconds", precision_seconds=0
    )
    cover["clocks"]["received"] = dict(value=T + 3701, encoding="unix_seconds", precision_seconds=0)
    records = [boundary, cover]
    previous = None
    for i, t in enumerate(range(60, 3600, 60)):
        records.append(
            event(
                f"HR{i}",
                "sensor_measurement",
                t,
                person,
                sensor="hr",
                value=hr + i % 3,
                quality="accepted",
                wear="worn",
                source_sha256="c" * 64,
                source_row=i + 1,
            )
        )
        key = f"STEP{i}"
        records.append(
            event(
                key,
                "sensor_measurement",
                t,
                person,
                sensor="steps",
                value=i * 2,
                quality="accepted",
                wear="worn",
                source_sha256="d" * 64,
                source_row=i + 1,
                counter_epoch="ONE",
                previous_event_id=previous,
            )
        )
        previous = key
    return ema, records
