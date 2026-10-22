"""Prospective alignment entrypoint, closed until independently verified metadata exist."""

import re

from .data import aligned_features


def align_verified_metadata(ema, ledger, metadata, contract, approved_evidence_hashes):
    """Caller supplies an evidence-reviewed registry, never model-selected clock assumptions.

    The existing EMA key/cutoff and development identity are authoritative. Evidence
    hashes refer to separately reviewed source contracts and original event payloads.
    This does not accept the current survey's self-reported wake answer.
    """
    if ema.get("n_partition") != "development" or ema.get("n_fold") not in range(5):
        raise ValueError("Development-only alignment")
    for key in ["n_ema_id", "n_participant"]:
        if not ema.get(key) or metadata.get(key) != ema[key]:
            raise ValueError("Boundary identity differs from EMA")
    if not {"n_participant", "n_partition"} <= set(ledger.columns):
        raise ValueError("Ledger identity provenance required")
    if (
        not ledger.n_participant.eq(ema["n_participant"]).all()
        or not ledger.n_partition.eq("development").all()
    ):
        raise ValueError("Wrong participant or partition")
    for key in ["contract_sha256", "boundary_payload_sha256"]:
        value = contract.get(key, "") if key == "contract_sha256" else metadata.get(key, "")
        if not re.fullmatch("[0-9a-f]{64}", value) or value not in approved_evidence_hashes:
            raise ValueError("Unapproved source evidence")
    row = {**metadata, "cutoff_epoch": float(ema["n_cutoff_epoch"])}
    # A caller-supplied replacement cutoff is never authoritative.
    if "cutoff_epoch" in metadata and metadata["cutoff_epoch"] != row["cutoff_epoch"]:
        raise ValueError("Cannot alter frozen EMA cutoff")
    return aligned_features(ledger, row, contract)
