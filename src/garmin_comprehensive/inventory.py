"""Trace all 54 original sheet rows without claiming an invented scalar feature count."""

from pathlib import Path
import json


def original_inventory(root):
    from garmin_preprocessing.common import verify_bundle

    root = Path(root)
    current = json.loads((root / "private/comprehensive_features/current.json").read_text())
    feature_bundle = root / current["features"]
    verify_bundle(feature_bundle)
    source = json.loads((feature_bundle / "original_sheet.json").read_text())
    if len(source["rows"]) != 55 or source["family_rows"] != 54:
        raise ValueError("Original-sheet inventory changed; review and version the reconciliation.")
    out = []
    for i, row in enumerate(source["rows"][1:], 1):
        name = row[0].split("\n")[0]
        previous = "not implemented as current EMA predictors"
        used = "no"
        status = "blocked"
        new = "none"
        reason = ""
        if i <= 27:
            reason = "Beat timing/order/quality and original adjacency are not established; no new NN/HRV interpretation. VLF and dependent powers remain deferred."
            if i in [1, 2, 3, 4, 7]:
                previous = "limited run-level IBI variability pilot only; no EMA HRV predictors"
            new = "Existing synthetic adapters and limited exploratory pilot preserved; no expanded real HRV"
        elif i in list(range(28, 34)) + [43]:
            status = "excluded"
            reason = "Proprietary Garmin stress excluded by study decision; no sensitivity justified in this checkpoint"
        elif i == 34:
            previous = "9 summaries ×8 windows =72 HR columns"
            used = "4 summaries ×8 windows =32 in Notebook12"
            status = "implemented"
            new = "30 observed-HR features ×8 windows =240"
            reason = "Original summary family now covered; additional features explicitly identified as new"
        elif i == 35:
            status = "provisional_sensitivity"
            new = "30 PulseOx summaries ×8 =240"
            reason = "Percent-like stream, provisional 70–100 gate; source quality/sentinel/acquisition semantics unresolved"
        elif i in [36, 40]:
            status = "implemented"
            previous = "supported additions at all eight windows"
            used = "steps only in Notebook12" if i == 36 else "no"
            new = "Supported additions and six additional supported-pair summaries per window"
            reason = "Reset, source/date/offset and gap controls retained; not total activity"
        elif i == 37:
            status = "implemented_with_explicit_support_definition"
            new = "Supported-time step rate stored per minute; multiply by60 for per-hour units"
            reason = "Whole-window hourly activity rate unsupported; this rate uses observed supported seconds only"
        elif i in [38, 39, 41]:
            reason = "Calorie/intensity units, weighting, counter/sentinel meanings are unverified"
        elif i == 42:
            status = "blocked_exact_measure"
            new = "Supported zero-change-time fraction available separately"
            reason = "Positive counter increment does not locate exact active minutes; no substitution with an active-minute claim"
        elif i in [44, 45]:
            previous = (
                "3 earlier-reference/deviation columns at60min only"
                if i == 44
                else "not implemented"
            )
            used = "separate Notebook10/12 history sensitivity" if i == 44 else "no"
            status = "implemented"
            new = "6 earlier-recording mean/SD/deviation features at each of8 lookbacks =48"
            reason = "Reference disjoint from current window; not resting or full-person normalization; standardized versions require SD>0"
        elif i in [46, 47, 48, 49, 50]:
            previous = "existing HR/activity/inventory quality diagnostics"
            status = "quality_only"
            new = "Separate counts, accepted fraction, occupied bins, gaps, freshness and logical-record zero fraction"
            reason = "Point samples do not establish literal time coverage; occupied minutes/support are named proxies"
            used = "missingness flags only in Notebook12; quality-only control newly separated"
        elif i in [51, 52]:
            previous = "frozen IBI eligibility/flag audits"
            status = "existing_quality_only"
            new = "Reuse corrected IBI audit; not physiological inputs"
            reason = "Flagged interval changes are provisional, not verified artifact or normal-beat labels"
        elif i in [53, 54]:
            previous = "implemented"
            used = "yes"
            status = "implemented"
            new = "Actual-local sine/cosine and training-encoded survey category"
            reason = "Context comparator; no response answer value or inferred prompt/wake boundary"
        out.append(
            dict(
                sheet_row=i + 1,
                original_feature_family=name,
                original_lookback=row[1] if len(row) > 1 else "",
                original_tag=row[2] if len(row) > 2 else "",
                previous_implementation=previous,
                previously_modeled=used,
                current_status=status,
                current_implementation=new,
                evidence_or_blocker=reason,
            )
        )
    return dict(
        source_basename=source["source_basename"],
        source_sha256=source["source_sha256"],
        family_rows=54,
        windowed_ibi_measure_variants=33,
        conditional_ibi_summary_columns=1518,
        counting_note="Rows1–25 expand compound band names to33 measures. Six summaries ×8 windows minus SD/trend at5min =46 per measure:33×46=1518 conditional IBI columns. Two duration-unspecified geometric families and unspecified percentile/level definitions prevent an exact complete scalar-feature total. These are planned, not implemented HRV.",
        previously_computable_wearable_columns=91,
        previously_modeled_distinct_wearable_columns=43,
        previous_count_note="Notebook04:72 HR+8 steps+8 floors=88 supported columns; Notebook10 adds3 history. Notebook12 used32 HR+8 steps+3 separate history=43 distinct wearable columns; primary60-minute model used5, plus context.",
        rows=out,
    )
