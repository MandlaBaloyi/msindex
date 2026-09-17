"""Contract terms, payout ladder, loss distribution and capital-based pricing.

The engine's index is the predicted yield anomaly (a shortfall against the provincial trend).
A policy pays nothing above the attachment, pays proportionally between attachment and exhaustion,
and pays the limit at or below exhaustion - the standard shape for a parametric crop cover.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

COST_OF_CAPITAL = 0.15   # annual return required on the capital held against this book
CAPITAL_QUANTILE = 0.995  # Solvency II style 1-in-200 tail


@dataclass(frozen=True)
class PolicyTerms:
    sum_insured_ha: float = 12000.0
    attachment: float = -0.10   # index level where cover starts
    exhaustion: float = -0.40   # index level where the limit is reached
    limit_share: float = 1.0    # maximum payout as a share of the sum insured
    min_payout_share: float = 0.05  # franchise: smaller payouts are not made
    hectares: float = 25000.0       # insured area across the book
    parcels: int = 12               # insured units (one per district here)

    @property
    def hectares_per_parcel(self) -> float:
        return self.hectares / max(self.parcels, 1)

    def payout_share(self, index) -> np.ndarray:
        """Share of the sum insured paid at a given index level."""
        idx = np.asarray(index, dtype="float64")
        span = max(self.attachment - self.exhaustion, 1e-9)  # exhaustion must sit below attachment
        share = np.clip((self.attachment - idx) / span, 0.0, 1.0) * self.limit_share
        return np.where(share < self.min_payout_share, 0.0, share)

    def payout_zar(self, index) -> np.ndarray:
        return self.payout_share(index) * self.sum_insured_ha


def apply_policy(df: pd.DataFrame, terms: PolicyTerms, index_col: str = "pred_yield_anomaly",
                 loss_col: str = "yield_anomaly") -> pd.DataFrame:
    """Adds the payout the policy makes and the loss the farmer actually suffered, per district-season."""
    out = df.copy()
    out["payout_share"] = terms.payout_share(out[index_col])
    out["payout_zar_ha"] = out["payout_share"] * terms.sum_insured_ha
    # What an ideal indemnity policy on the same ladder would have paid, given the reported yield
    out["loss_share"] = terms.payout_share(out[loss_col])
    out["loss_zar_ha"] = out["loss_share"] * terms.sum_insured_ha
    return out


def season_losses(scored: pd.DataFrame) -> pd.Series:
    """Portfolio loss ratio per season: the book is spread equally across the districts."""
    return scored.groupby("season")["payout_share"].mean()


def risk_stats(season_loss: pd.Series, bootstrap: int = 20000, seed: int = 42) -> dict:
    """Expected loss and tail measures.

    With ~26 observed seasons the 1-in-200 tail cannot be read off the data, so seasons are
    resampled with replacement to get a spread, and the number is reported as modelled, not measured.
    """
    values = season_loss.to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(bootstrap,), replace=True)
    return {
        "expected_loss": float(values.mean()),
        "std_dev": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "worst_observed": float(values.max()),
        "worst_season": int(season_loss.idxmax()) if len(season_loss) else None,
        "payout_seasons": int((values > 0).sum()),
        "seasons": int(len(values)),
        "var_995": float(np.quantile(draws, CAPITAL_QUANTILE)),
        "tvar_995": float(draws[draws >= np.quantile(draws, CAPITAL_QUANTILE)].mean()),
        "var_90": float(np.quantile(draws, 0.90)),
    }


def premium(stats: dict, terms: PolicyTerms, expense_ratio: float = 0.20,
            cost_of_capital: float = COST_OF_CAPITAL) -> dict:
    """Technical premium = expected loss + a charge for the capital the tail requires."""
    el = stats["expected_loss"] * terms.sum_insured_ha
    capital = max(stats["tvar_995"] - stats["expected_loss"], 0.0) * terms.sum_insured_ha
    capital_charge = capital * cost_of_capital
    technical = el + capital_charge
    gross = technical / (1 - expense_ratio) if expense_ratio < 1 else float("inf")
    return {
        "expected_loss_ha": el,
        "capital_ha": capital,
        "capital_charge_ha": capital_charge,
        "technical_premium_ha": technical,
        "gross_premium_ha": gross,
        "rate_on_line": gross / terms.sum_insured_ha if terms.sum_insured_ha else 0.0,
        "book_premium": gross * terms.hectares,
        "book_capital": capital * terms.hectares,
        "loss_ratio_at_expected": el / gross if gross else 0.0,
    }


def basis_metrics(scored: pd.DataFrame, loss_threshold: float = -0.15) -> dict:
    """How well the payout matches the loss, from the farmer's side."""
    real_loss = scored["loss_share"] > 0
    paid = scored["payout_share"] > 0
    covered = scored.loc[real_loss, "payout_share"] / scored.loc[real_loss, "loss_share"].replace(0, np.nan)
    severe = scored["yield_anomaly"] <= loss_threshold
    return {
        "loss_events": int(real_loss.sum()),
        "paid_when_loss": float((paid & real_loss).sum() / max(real_loss.sum(), 1) * 100),
        "coverage_ratio": float(covered.clip(upper=3).median()) if real_loss.any() else 0.0,
        "under_half_covered": float((covered < 0.5).mean() * 100) if real_loss.any() else 0.0,
        "paid_without_loss": float((paid & ~real_loss).sum() / max(paid.sum(), 1) * 100),
        "payout_loss_corr": float(scored["payout_share"].corr(scored["loss_share"])),
        "severe_paid": float((paid & severe).sum() / max(severe.sum(), 1) * 100),
    }


def district_correlation(scored: pd.DataFrame) -> float:
    """Mean pairwise correlation of district payouts: how little the book diversifies."""
    wide = scored.pivot_table(index="season", columns="district", values="payout_share")
    corr = wide.corr().to_numpy()
    off = corr[~np.eye(len(corr), dtype=bool)]
    return float(np.nanmean(off))


def reinsurance_split(season_loss: pd.Series, attachment: float, exhaustion: float) -> pd.DataFrame:
    """Splits each season's portfolio loss into retained and ceded under an aggregate stop-loss."""
    ceded = (season_loss.clip(lower=attachment) - attachment).clip(upper=max(exhaustion - attachment, 0))
    return pd.DataFrame({"season": season_loss.index, "portfolio_loss": season_loss.to_numpy(),
                         "ceded": ceded.to_numpy(), "retained": (season_loss - ceded).to_numpy()})
