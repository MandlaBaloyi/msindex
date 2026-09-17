import numpy as np
import pandas as pd
import pytest

from src.ledger import payment_batch
from src.pricing import PolicyTerms, apply_policy, basis_metrics, district_correlation, premium, reinsurance_split, risk_stats, season_losses
from src.trigger import AuditChain

TERMS = PolicyTerms(sum_insured_ha=10000, attachment=-0.10, exhaustion=-0.40, min_payout_share=0.0, hectares=1200, parcels=12)


def test_ladder_shape():
    assert TERMS.payout_share(0.05) == 0
    assert TERMS.payout_share(-0.10) == 0
    assert TERMS.payout_share(-0.25) == pytest.approx(0.5)
    assert TERMS.payout_share(-0.40) == pytest.approx(1.0)
    assert TERMS.payout_share(-0.80) == pytest.approx(1.0)  # capped at the limit


def test_ladder_is_monotone():
    idx = np.linspace(-0.6, 0.2, 50)
    payouts = TERMS.payout_share(idx)
    assert np.all(np.diff(payouts) <= 1e-12)


def test_franchise_suppresses_small_payouts():
    terms = PolicyTerms(attachment=-0.10, exhaustion=-0.40, min_payout_share=0.2)
    assert terms.payout_share(-0.14) == 0  # would be 13% of cover
    assert terms.payout_share(-0.30) > 0


def test_limit_caps_payout():
    terms = PolicyTerms(sum_insured_ha=10000, attachment=-0.1, exhaustion=-0.4, limit_share=0.5, min_payout_share=0.0)
    assert terms.payout_zar(-0.9) == pytest.approx(5000)


def sample_scored():
    df = pd.DataFrame({
        "season": [2020, 2020, 2021, 2021],
        "district": ["A", "B", "A", "B"],
        "pred_yield_anomaly": [-0.40, -0.10, 0.05, -0.25],
        "yield_anomaly": [-0.40, -0.20, 0.00, -0.25],
    })
    return apply_policy(df, TERMS)


def test_apply_policy_adds_payout_and_loss():
    scored = sample_scored()
    assert scored["payout_share"].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.5])
    assert scored["loss_share"].tolist() == pytest.approx([1.0, 1 / 3, 0.0, 0.5])
    assert scored["payout_zar_ha"].max() == pytest.approx(TERMS.sum_insured_ha)


def test_season_losses_average_across_districts():
    losses = season_losses(sample_scored())
    assert losses[2020] == pytest.approx(0.5)
    assert losses[2021] == pytest.approx(0.25)


def test_risk_stats_tail_is_at_least_the_mean():
    stats = risk_stats(season_losses(sample_scored()))
    assert stats["expected_loss"] == pytest.approx(0.375)
    assert stats["tvar_995"] >= stats["var_995"] >= stats["expected_loss"]
    assert stats["worst_observed"] == pytest.approx(0.5)


def test_premium_covers_expected_loss_and_capital():
    stats = risk_stats(season_losses(sample_scored()))
    price = premium(stats, TERMS, expense_ratio=0.2, cost_of_capital=0.15)
    assert price["gross_premium_ha"] > price["technical_premium_ha"] >= price["expected_loss_ha"]
    assert price["book_premium"] == pytest.approx(price["gross_premium_ha"] * TERMS.hectares)


def test_zero_loss_history_prices_to_zero():
    losses = pd.Series([0.0, 0.0, 0.0], index=[2018, 2019, 2020])
    price = premium(risk_stats(losses), TERMS)
    assert price["expected_loss_ha"] == 0 and price["gross_premium_ha"] == 0


def test_basis_metrics_flag_undercover():
    bm = basis_metrics(sample_scored())
    assert bm["loss_events"] == 3
    assert 0 <= bm["paid_when_loss"] <= 100
    assert bm["under_half_covered"] > 0  # district B in 2020 suffers a loss and is paid nothing


def test_reinsurance_split_conserves_the_loss():
    losses = season_losses(sample_scored())
    split = reinsurance_split(losses, 0.3, 0.6)
    assert np.allclose(split["retained"] + split["ceded"], split["portfolio_loss"])
    assert (split["ceded"] <= 0.3 + 1e-9).all()


def test_district_correlation_is_a_correlation():
    assert -1 <= district_correlation(sample_scored()) <= 1


def test_payment_batch_refuses_a_broken_ledger():
    chain = AuditChain()
    for i in range(3):
        chain.append_decision(f"P{i}", 2025, {"rain_mm": 400.0}, -0.3, -0.15, 1)
    chain.tamper_decision(1)
    with pytest.raises(ValueError, match="does not verify"):
        payment_batch(chain, TERMS, 2025)


def test_payment_batch_totals_match_the_ladder():
    chain = AuditChain()
    chain.append_decision("P0", 2025, {"rain_mm": 400.0}, -0.40, -0.15, 1)
    chain.append_decision("P1", 2025, {"rain_mm": 700.0}, 0.10, -0.15, 0)
    csv_text, signature, summary = payment_batch(chain, TERMS, 2025)
    assert summary["payments"] == 1  # the no-pay record is not in the batch
    assert summary["total_zar"] == pytest.approx(TERMS.sum_insured_ha * TERMS.hectares_per_parcel)
    assert len(signature) == 64
    assert "P0" not in csv_text  # only the pseudonym travels
