import numpy as np
import pandas as pd
import pytest

from src.trigger import LOSS_THRESHOLD, AuditChain, classify_risk, evaluate_triggers, pseudonymise_parcel


def make_df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "rain_mm": rng.uniform(200, 800, n),
        "pred_yield_anomaly": rng.uniform(-0.5, 0.3, n),
        "yield_anomaly": rng.uniform(-0.5, 0.3, n),
    })


@pytest.mark.parametrize("cutoff", [250.0, 400.0, 550.0, 700.0])
def test_both_triggers_pay_the_same_number(cutoff):
    scored, metrics = evaluate_triggers(make_df(), cutoff)
    assert scored["trigger_baseline"].sum() == scored["trigger_msindex"].sum()
    assert metrics["baseline"]["payouts"] == metrics["msindex"]["payouts"]


def test_matching_survives_tied_predictions():
    df = make_df(12)
    df["pred_yield_anomaly"] = -0.2  # every prediction identical
    scored, metrics = evaluate_triggers(df, 500.0)
    assert scored["trigger_msindex"].sum() == scored["trigger_baseline"].sum()


def test_msindex_pays_the_worst_predictions():
    df = make_df(30)
    scored, _ = evaluate_triggers(df, 500.0)
    paid = scored.loc[scored["trigger_msindex"] == 1, "pred_yield_anomaly"]
    unpaid = scored.loc[scored["trigger_msindex"] == 0, "pred_yield_anomaly"]
    if len(paid) and len(unpaid):
        assert paid.max() <= unpaid.min()


def test_metrics_add_up():
    _, metrics = evaluate_triggers(make_df(40), 500.0)
    for side in ("baseline", "msindex"):
        m = metrics[side]
        assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 40
        assert m["missed_losses"] == m["fn"]


def test_no_payouts_when_cutoff_below_every_season():
    scored, metrics = evaluate_triggers(make_df(), 0.0)
    assert metrics["baseline"]["payouts"] == 0
    assert metrics["msindex"]["payouts"] == 0
    assert metrics["msindex"]["hit_rate"] == 0.0


def test_risk_bands_follow_thresholds():
    assert classify_risk(-0.30) == "Catastrophic"
    assert classify_risk(LOSS_THRESHOLD) == "Severe"
    assert classify_risk(-0.10) == "Moderate"
    assert classify_risk(0.10) == "Low"


def test_pseudonym_is_stable_and_opaque():
    assert pseudonymise_parcel("PARCEL-1") == pseudonymise_parcel("PARCEL-1")
    assert pseudonymise_parcel("PARCEL-1") != pseudonymise_parcel("PARCEL-2")
    assert "PARCEL" not in pseudonymise_parcel("PARCEL-1")


def seeded_chain(n=4):
    chain = AuditChain()
    for i in range(n):
        chain.append_decision(f"PARCEL-{i}", 2025, {"rain_mm": 400.0 + i}, -0.2, -0.15, int(i % 2 == 0))
    return chain


def test_fresh_chain_verifies():
    assert seeded_chain().verify() == (True, -1)


def test_tampering_is_detected_at_the_edited_record():
    chain = seeded_chain()
    assert chain.tamper_decision(2)
    valid, index = chain.verify()
    assert not valid and index == 2


def test_tampering_flips_rather_than_sets():
    chain = seeded_chain()
    before = chain.chain[1]["payload"]["decision"]
    chain.tamper_decision(1)
    assert chain.chain[1]["payload"]["decision"] != before


def test_rewriting_a_hash_is_also_caught():
    chain = seeded_chain()
    chain.chain[3]["prev_hash"] = "0" * 64
    valid, index = chain.verify()
    assert not valid and index == 3


def test_empty_chain_verifies():
    assert AuditChain().verify() == (True, -1)
