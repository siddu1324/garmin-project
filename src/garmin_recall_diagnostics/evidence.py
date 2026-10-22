"""Local blank-questionnaire evidence and development timing field inventory."""

import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from garmin_ema_baseline.linkage import questionnaire_evidence
from garmin_inventory.privacy import AuditError, sha256
from garmin_inventory.sources import discover, safe_members
from .recall import TIMING_FIELDS

FORMS = {
    "Morning": "3aab0cb8e25586e2aaf233a8c9132d0f416f8b4de608c79143ae3539f218018b",
    "Midday": "bddc0455b203de2abf25409a2acabc5b243cf40d63c2d399254526fe48964f79",
    "Endday": "2197a34a1695053c074af540e8cfd8519003bb0f24d67f6a5cd9582ea26d3667",
}
SUFFIXES = {
    "PSS_NERVOUS": "felt STRESSED or nervous?",
    "PSS_COPE": "felt that you were ABLE to handle or cope with things well?",
    "PSS_CONTROL": "felt that you were ON TOP OF THINGS?",
}


def inspect_sources(root, ema):
    sources = discover(root / "data")
    hashes = {s.content_hash(): s for s in sources[2]}
    catalog = json.loads(
        (root / "private/recall_diagnostics/questionnaire_sources.json").read_text()
    )
    for item in catalog:
        if item["sha256"] not in hashes or sha256(root / item["local_copy"]) != item["sha256"]:
            raise AuditError("Local questionnaire citation does not match raw source bytes.")
    coding = questionnaire_evidence(root)
    evidence = []
    for item in coding:
        category = item["category"]
        prefix = "Since waking up" if category == "Morning" else "Since the last prompt"
        for i, field in enumerate(item["items"]):
            evidence.append(
                {
                    "survey_category": category,
                    **field,
                    "exact_wording": prefix
                    + ", how much have you: "
                    + SUFFIXES[field["export_field"]],
                    "recall_boundary": "final waking today"
                    if category == "Morning"
                    else "last prompt",
                    "rendered_form_sha256": FORMS[category],
                    "rendered_form_page": 3 if category == "Morning" and i == 2 else 2,
                    "question_number": i + (9 if category == "Morning" else 5),
                    "codebook_sha256": item["document_sha256"],
                    "transcription_method": "visual verification of supplied outlined-text blank forms; codebook field/code cross-check",
                }
            )
    field_rows = []
    for category, f in ema.groupby("n_survey_category"):
        headers = set().union(*(set(json.loads(s)) for s in f.n_source_columns.unique()))
        for name in TIMING_FIELDS:
            values = (
                f[name].astype("string").str.strip() if name in f else pd.Series("", index=f.index)
            )
            field_rows.append(
                {
                    "survey_category": category,
                    "field": name,
                    "in_original_schema": name in headers,
                    "responses": len(f),
                    "nonmissing": int((values.notna() & values.ne("")).sum()),
                }
            )
        for name in [
            "n_invalid_clock",
            "n_scheduled_clock_discrepancy",
            "n_actual_before_scheduled",
        ]:
            field_rows.append(
                {
                    "survey_category": category,
                    "field": name,
                    "in_original_schema": False,
                    "responses": len(f),
                    "flagged": int(f[name].sum()),
                }
            )
    # Structural audit only: all supplied CSV schemas, no participant response values read here.
    import csv
    import io

    nested_extensions = Counter()
    nested_sources = {(s.archive, s.nested) for s in sources[1] if s.nested is not None}
    for archive, member in sorted(nested_sources):
        with zipfile.ZipFile(archive) as outer:
            with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
                nested_extensions.update(
                    Path(m.filename).suffix.lower() for m in safe_members(inner)
                )
    if set(nested_extensions) - {".csv"}:
        raise AuditError("Additional nested source types require a timing-log audit.")
    schema_rows = []
    for label, members in [("ema", sources[0]), ("garmin", sources[1])]:
        unique = {}
        for source in members:
            with source.open() as stream:
                header = tuple(next(csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig"))))
            unique[header] = unique.get(header, 0) + 1
        for header, count in unique.items():
            # Only field labels with timing relevance; never source/participant filenames.
            timing = [
                c
                for c in header
                if any(
                    k in c.lower()
                    for k in [
                        "timestamp",
                        "start",
                        "wake",
                        "sleep",
                        "prompt",
                        "notif",
                        "deliver",
                        "submit",
                        "complet",
                        "remind",
                    ]
                )
                or c in ["ts", "dte_tme", "timezone_offset", "instance_id"]
            ]
            schema_rows.append(
                {"source_type": label, "files_with_schema": count, "timing_fields": timing}
            )
    return {
        "questionnaire_items": evidence,
        "timing_fields": field_rows,
        "structural_inventory": {
            "archives": sources[3],
            "nested_archive_count": len(nested_sources),
            "nested_member_extensions": dict(nested_extensions),
            "schemas": schema_rows,
            "ema_sources": len(sources[0]),
            "sensor_sources": len(sources[1]),
            "blank_pdfs": len(sources[2]),
            "dedicated_prompt_or_submission_log_supplied": False,
        },
        "wake_question": {
            "wording": "What time did you wake up today? (If multiple awakenings, select the final time you woke up)",
            "rendered_form_sha256": FORMS["Morning"],
            "page": 2,
            "question": 7,
        },
        "questionnaire_source_hashes": sorted(hashes),
    }
