"""Explicit reviewed-provider entrypoint; does not activate on inbox presence alone."""

import json
from pathlib import Path
import tempfile
import pandas as pd
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import digest_object, publish_bundle, verify_bundle, write_json
from garmin_comprehensive_regression.evaluate import finite
from .metadata import import_metadata, MetadataError
from .alignment import aligned_features, versioned_history
from .experiment import evaluate, QUALITY


def build(root, inbox, approval, development, cfg, execute_fixed_experiment=False):
    root = Path(root)
    imported = import_metadata(
        inbox, approval, development, root / "private/measurement_alignment/provider_imports"
    )
    verify_bundle(imported)
    events = [json.loads(s) for s in (imported / "logical_events.jsonl").read_text().splitlines()]
    contract = json.loads((imported / "contract.json").read_text())
    identity = dict(
        import_manifest=sha256(imported / "manifest.json"),
        registry=digest_object(development.to_json(orient="split")),
        configuration=cfg,
        execute_fixed_experiment=execute_fixed_experiment,
        code={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    )
    parent = root / "private/measurement_alignment/provider_aligned"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / digest_object(identity)[:16]
    if destination.exists():
        verify_bundle(destination)
        return destination
    rows = []
    histories = []
    lineage = []
    for _, row in development.iterrows():
        ema = row.to_dict()
        # Explicit provider start event binds prompt/date keys to an existing stable EMA.
        starts = [
            e for e in events if e["event_type"] == "survey_started" and e["ema_id"] == row.n_ema_id
        ]
        if len(starts) > 1:
            raise MetadataError("ambiguous_survey_start_binding")
        if starts:
            start = starts[0]
            original = row.n_actual_epoch
            if (
                not start["parsed_clocks"]["start"]["lower"]
                <= original
                <= start["parsed_clocks"]["start"]["upper"]
            ):
                raise MetadataError("provider_start_disagrees_with_preserved_start")
            for k in ["provider_prompt_id", "verified_local_date"]:
                if k in start:
                    ema[k] = start[k]
        boundary, features, quality = aligned_features(ema, events, contract)
        rows.append(
            {
                **row.to_dict(),
                **features,
                **{"coverage_" + k: v for k, v in quality.items()},
                **{"alignment_" + k: v for k, v in boundary.items()},
            }
        )
        h, history_lineage = versioned_history(development, events, ema)
        histories.append(h)
        lineage.append(history_lineage)
    frame = pd.DataFrame(rows)
    # Missing columns stay missing. Neither outcomes nor physiological readings are imputed.
    from .experiment import ALIGNED

    for name in ALIGNED + ["aligned_steps_rate"] + QUALITY:
        if name not in frame:
            frame[name] = float("nan")
    tmp = Path(tempfile.mkdtemp(prefix=".building_", dir=parent))
    frame.to_parquet(tmp / "aligned_development.parquet", index=False)
    pd.concat(histories, ignore_index=True).to_parquet(
        tmp / "available_version_history.parquet", index=False
    )
    pd.concat(lineage, ignore_index=True).to_parquet(tmp / "history_lineage.parquet", index=False)
    write_json(tmp / "identity.json", identity)
    if execute_fixed_experiment:
        result = evaluate(frame, cfg)
        for k in ["predictions", "support", "metrics", "differences", "calibration", "membership"]:
            result[k].to_parquet(tmp / f"{k}.parquet", index=False)
        write_json(tmp / "trials.json", finite(result["trials"]))
        write_json(tmp / "skips.json", result["skips"])
    publish_bundle(
        tmp,
        destination,
        dict(
            identity=identity,
            rows=len(frame),
            reconstructible=int(frame.alignment_status.eq("RECONSTRUCTIBLE").sum()),
            history_prediction="not_automatically_activated",
            real_final_test_evaluated=False,
        ),
    )
    return destination
