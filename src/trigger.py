import os
import json
import hmac
import hashlib
import secrets
import warnings
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Single source of truth for thresholds used across the app
LOSS_THRESHOLD = -0.15          # yield anomaly (vs provincial trend) at or below this is a genuine loss event
TEST_START_SEASON = 2018        # planting years before this are used for training; 2018/19 onward is the holdout
RISK_BANDS = [                  # (upper bound on predicted anomaly, label), checked in order
    (-0.25, "Catastrophic"),
    (LOSS_THRESHOLD, "Severe"),
    (-0.05, "Moderate"),
]
GENESIS_HASH = "GENESIS_ROOT_HASH_0000000000000000"


def _load_secret(name: str) -> tuple[bytes, bool]:
    """Returns (secret, is_fallback). Missing secrets get a random per-process value, never a known constant."""
    value = os.getenv(name)
    if value:
        return value.encode("utf-8"), False
    warnings.warn(f"{name} is not set; using a random ephemeral value. Signatures and pseudonyms will not survive a restart.")
    return secrets.token_bytes(32), True


SECRET_KEY, _KEY_FALLBACK = _load_secret("MSINDEX_HMAC_KEY")
PARCEL_SALT, _SALT_FALLBACK = _load_secret("MSINDEX_PARCEL_SALT")
USING_FALLBACK_SECRETS = _KEY_FALLBACK or _SALT_FALLBACK


def classify_risk(anomaly: float) -> str:
    for upper, label in RISK_BANDS:
        if anomaly <= upper:
            return label
    return "Low"


def pseudonymise_parcel(parcel_id: str) -> str:
    """POPIA Data Minimisation: Salted SHA-256 parcel pseudonymisation."""
    raw = PARCEL_SALT + parcel_id.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def evaluate_triggers(test_df: pd.DataFrame, rainfall_cutoff_mm: float):
    """Evaluates baseline rainfall index vs msIndex engine at an exactly matched number of payouts."""
    df = test_df.copy()
    y_true = df["yield_anomaly"] <= LOSS_THRESHOLD

    # 1. Baseline rainfall trigger
    df["trigger_baseline"] = (df["rain_mm"] < rainfall_cutoff_mm).astype(int)
    n_payouts = int(df["trigger_baseline"].sum())

    # 2. msIndex trigger: pay the same number of rows, choosing the worst predicted anomalies.
    # Rank-based selection keeps the count exact even when predictions tie.
    rank = df["pred_yield_anomaly"].rank(method="first")
    df["trigger_msindex"] = (rank <= n_payouts).astype(int)
    pred_cutoff = float(df.loc[df["trigger_msindex"] == 1, "pred_yield_anomaly"].max()) if n_payouts else float("-inf")

    def compute_metrics(payout_col: str):
        payouts = df[payout_col] == 1

        tp = int((payouts & y_true).sum())
        fp = int((payouts & ~y_true).sum())
        fn = int((~payouts & y_true).sum())
        tn = int((~payouts & ~y_true).sum())

        hit_rate = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        false_payout_rate = (fp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0

        return {
            "payout_rate": float(payouts.mean()),
            "payouts": tp + fp,
            "hit_rate": float(hit_rate),
            "false_payout_rate": float(false_payout_rate),
            "missed_losses": fn,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }

    metrics = {
        "baseline": compute_metrics("trigger_baseline"),
        "msindex": compute_metrics("trigger_msindex"),
        "pred_cutoff": pred_cutoff,
        "rainfall_cutoff_mm": rainfall_cutoff_mm
    }
    return df, metrics


class AuditChain:
    """Tamper-evident append-only decision log using an HMAC-SHA256 hash chain."""
    def __init__(self):
        self.chain = []

    def _hash_record(self, payload: dict, prev_hash: str) -> str:
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hmac.new(SECRET_KEY, serialized + prev_hash.encode("utf-8"), hashlib.sha256).hexdigest()

    def append_decision(self, parcel_id: str, season: int, features: dict, prediction: float, cutoff: float, decision: int, model_version: str = "v1.0.0"):
        prev_hash = self.chain[-1]["record_hash"] if self.chain else GENESIS_HASH
        payload = {
            "parcel_pseudonym": pseudonymise_parcel(parcel_id),
            "season": int(season),
            "features": {k: float(v) for k, v in features.items()},
            "prediction": float(round(prediction, 4)),
            "cutoff": float(round(cutoff, 4)),
            "decision": int(decision),
            "model_version": model_version,
            "sealed_at": datetime.now(timezone.utc).isoformat(),
        }
        rec_hash = self._hash_record(payload, prev_hash)
        record = {
            "index": len(self.chain),
            "prev_hash": prev_hash,
            "payload": payload,
            "record_hash": rec_hash
        }
        self.chain.append(record)
        return record

    def verify(self) -> tuple[bool, int]:
        for i, rec in enumerate(self.chain):
            expected_prev = GENESIS_HASH if i == 0 else self.chain[i - 1]["record_hash"]
            if rec["prev_hash"] != expected_prev:
                return False, i
            computed = self._hash_record(rec["payload"], rec["prev_hash"])
            if computed != rec["record_hash"]:
                return False, i
        return True, -1

    def tamper_decision(self, index: int) -> bool:
        """Simulates insider tampering by flipping a historical payout decision. Returns True if a record changed."""
        if 0 <= index < len(self.chain):
            payload = self.chain[index]["payload"]
            payload["decision"] = 1 - payload["decision"]
            return True
        return False
