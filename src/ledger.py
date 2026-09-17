"""Durable decision ledger and payout settlement files.

`AuditChain` (src/trigger.py) holds the hash chain; this module gives it somewhere to live and a way
to turn verified decisions into an instruction a treasury system can act on:

- append-only JSONL on disk, so the ledger survives a restart,
- a season root hash (the last record's hash) that can be published as evidence of what was decided,
- model and dataset fingerprints bound into every record, so a record names the exact model that made it,
- a signed payment batch generated only from a chain that still verifies.
"""
import csv
import hashlib
import hmac
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .trigger import SECRET_KEY, AuditChain

LEDGER_DIR = Path(__file__).resolve().parents[1] / "data" / "ledger"


def dataset_fingerprint(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def model_fingerprint(model, feature_cols) -> str:
    """Fingerprints the fitted coefficients, so a retrained model cannot silently reuse a version string."""
    ridge = model[-1] if hasattr(model, "__getitem__") else model
    parts = [np.asarray(getattr(ridge, "coef_", [])).round(8).tobytes(),
             np.asarray([getattr(ridge, "intercept_", 0.0)]).round(8).tobytes(),
             ",".join(feature_cols).encode()]
    return hashlib.sha256(b"".join(parts)).hexdigest()[:16]


def model_version(model, feature_cols, dataset_path) -> str:
    return f"ridge-{model_fingerprint(model, feature_cols)}+data-{dataset_fingerprint(dataset_path)}"


def ledger_path(season: int) -> Path:
    return LEDGER_DIR / f"decisions_{season}.jsonl"


def save(chain: AuditChain, season: int) -> Path:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    path = ledger_path(season)
    with path.open("w", encoding="utf-8") as fh:
        for record in chain.chain:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def load(season: int) -> AuditChain | None:
    path = ledger_path(season)
    if not path.exists():
        return None
    chain = AuditChain()
    chain.chain = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return chain or None


def season_root(chain: AuditChain) -> str:
    """The head of the chain: publishing this one string fixes every record beneath it."""
    return chain.chain[-1]["record_hash"] if chain.chain else "EMPTY"


def payment_batch(chain: AuditChain, terms, season: int, reference_prefix: str = "MSIDX") -> tuple[str, str, dict]:
    """Builds a payment instruction file for the payouts in a verified ledger.

    Returns (csv_text, signature, summary). The signature is an HMAC over the file, so the treasury
    system can confirm the batch came from this engine and was not edited in transit.
    """
    valid, broken = chain.verify()
    if not valid:
        raise ValueError(f"ledger does not verify at record #{broken}; refusing to build a payment batch")

    rows, total = [], 0.0
    for record in chain.chain:
        payload = record["payload"]
        if payload["decision"] != 1:
            continue
        amount = float(terms.payout_zar(payload["prediction"])) * terms.hectares_per_parcel
        total += amount
        rows.append({
            "reference": f"{reference_prefix}-{season}-{record['index']:04d}",
            "parcel_pseudonym": payload["parcel_pseudonym"],
            "season": payload["season"],
            "index_value": f"{payload['prediction']:.4f}",
            "payout_share": f"{terms.payout_share(payload['prediction']):.4f}",
            "amount_zar": f"{amount:.2f}",
            "currency": "ZAR",
            "model_version": payload["model_version"],
            "record_hash": record["record_hash"],
        })

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]) if rows else
                            ["reference", "parcel_pseudonym", "season", "index_value", "payout_share",
                             "amount_zar", "currency", "model_version", "record_hash"])
    writer.writeheader()
    writer.writerows(rows)
    csv_text = buffer.getvalue()
    signature = hmac.new(SECRET_KEY, csv_text.encode("utf-8"), hashlib.sha256).hexdigest()
    summary = {
        "season": season,
        "payments": len(rows),
        "total_zar": total,
        "ledger_root": season_root(chain),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signature": signature,
    }
    return csv_text, signature, summary


def batch_manifest(summary: dict) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)
