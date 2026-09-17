"""Underwriting briefings written by Claude from the engine's own numbers.

The model never sees raw data or free text from outside the system: every request carries a compact
fact sheet the engine computed, and the prompt forbids inventing figures. Without API credentials the
same fact sheet renders through a deterministic template, so the page always works.
"""
import json
import os

import pandas as pd

from .pricing import PolicyTerms, apply_policy, basis_metrics, season_losses

MODEL = "claude-opus-5"
SYSTEM = (
    "You are an agricultural underwriting analyst for msIndex, a parametric maize insurance engine "
    "covering South Africa's maize triangle. You will be given a JSON fact sheet computed by the engine.\n\n"
    "Rules:\n"
    "- Use only the numbers in the fact sheet. Never invent or estimate figures, and never use outside knowledge "
    "about specific seasons.\n"
    "- If the fact sheet cannot answer something, say so plainly.\n"
    "- Be direct and concrete. An underwriter reads this to decide whether to pay, reserve or re-rate.\n"
    "- Flag basis risk honestly: the index can pay when there was no loss, and miss losses that happened.\n"
    "- Never promise a payout; decisions follow the signed ledger and policy wording."
)
BRIEF_PROMPT = (
    "Write the season briefing as four short sections with these headings: **Conditions**, **Model view**, "
    "**Exposure**, **Watch items**. Six sentences at most in total. Name the districts that matter rather than "
    "describing the region in general."
)


def credentials_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def build_facts(df: pd.DataFrame, season: int, terms: PolicyTerms, context: dict) -> dict:
    """The only thing the model sees: figures the engine already computed."""
    season_df = apply_policy(df[df["season"] == season], terms)
    prev = df[df["season"] == season - 1]
    districts = [{
        "district": r["district"],
        "province": r["province"],
        "rain_oct_mar_mm": round(float(r["rain_mm"])),
        "water_balance_mm": round(float(r["water_balance_mm"])),
        "dry_spell_days": int(r["dry_spell_days"]),
        "heat_stress_days": int(r["heat_stress_days"]),
        "greenness_anomaly": round(float(r["ndvi_anomaly"]), 3),
        "predicted_yield_anomaly_pct": round(float(r["pred_yield_anomaly"]) * 100, 1),
        "risk_band": r["Risk band"],
        "payout_share_of_cover_pct": round(float(r["payout_share"]) * 100, 1),
    } for _, r in season_df.sort_values("pred_yield_anomaly").iterrows()]

    provinces = [{
        "province": p,
        "predicted_yield_t_ha": round(float(g["pred_yield_tha"].mean()), 2),
        "reported_yield_t_ha": None if pd.isna(g["yield_tha"].iloc[0]) else round(float(g["yield_tha"].iloc[0]), 2),
        "trend_yield_t_ha": round(float(g["yield_trend"].iloc[0]), 2),
        "yield_status": g["yield_status"].iloc[0] if isinstance(g["yield_status"].iloc[0], str) else "not yet reported",
    } for p, g in season_df.groupby("province")]

    return {
        "season": f"{season}/{str(season + 1)[-2:]}",
        "data_split": season_df["Data split"].iloc[0],
        "region_summary": {
            "mean_rain_oct_mar_mm": round(float(season_df["rain_mm"].mean())),
            "change_vs_previous_season_mm": None if prev.empty else round(float(season_df["rain_mm"].mean() - prev["rain_mm"].mean())),
            "mean_greenness_anomaly": round(float(season_df["ndvi_anomaly"].mean()), 3),
            "districts_with_predicted_loss": int((season_df["pred_yield_anomaly"] <= -0.15).sum()),
            "districts_total": len(season_df),
            "mean_payout_share_of_cover_pct": round(float(season_df["payout_share"].mean()) * 100, 1),
        },
        "districts": districts,
        "provinces": provinces,
        "policy_terms": {
            "sum_insured_zar_per_ha": terms.sum_insured_ha,
            "attachment_yield_shortfall_pct": round(terms.attachment * 100, 1),
            "exhaustion_yield_shortfall_pct": round(terms.exhaustion * 100, 1),
            "insured_area_ha": terms.hectares,
            "season_payout_zar": round(float(season_df["payout_share"].mean()) * terms.sum_insured_ha * terms.hectares),
        },
        "track_record": context.get("track_record", {}),
        "model": context.get("model", {}),
        "caveats": [
            "Yields are published per province, so the four districts of a province share one reported yield.",
            "The index is a model estimate, not a measured farm yield; basis risk is real.",
            "The current season's reported yield is a preliminary CEC forecast until the February final figures.",
        ],
    }


def track_record(df: pd.DataFrame, test_df: pd.DataFrame, terms: PolicyTerms) -> dict:
    scored = apply_policy(test_df, terms)
    bm = basis_metrics(scored)
    losses = season_losses(scored)
    return {
        "holdout_seasons": f"{int(test_df['season'].min())}-{int(test_df['season'].max())}",
        "loss_events_paid_pct": round(bm["paid_when_loss"], 1),
        "median_cover_of_loss_pct": round(bm["coverage_ratio"] * 100, 1),
        "losses_under_half_covered_pct": round(bm["under_half_covered"], 1),
        "mean_seasonal_payout_share_pct": round(float(losses.mean()) * 100, 1),
    }


def fallback_briefing(facts: dict) -> str:
    """Deterministic briefing used when no Claude credentials are configured."""
    r = facts["region_summary"]
    worst = facts["districts"][0] if facts["districts"] else None
    change = r["change_vs_previous_season_mm"]
    change_txt = "" if change is None else f" ({change:+d} mm against last season)"
    lines = [
        f"**Conditions.** {facts['season']} averaged {r['mean_rain_oct_mar_mm']} mm of rain from October to March"
        f"{change_txt}, with a mean greenness anomaly of {r['mean_greenness_anomaly']:+.3f}.",
        f"**Model view.** {r['districts_with_predicted_loss']} of {r['districts_total']} districts are predicted to "
        f"fall at least 15% below trend."
        + (f" The weakest is {worst['district']} ({worst['province']}) at "
           f"{worst['predicted_yield_anomaly_pct']:+.1f}%." if worst else ""),
        f"**Exposure.** The book pays {r['mean_payout_share_of_cover_pct']:.1f}% of cover on average, about "
        f"R {facts['policy_terms']['season_payout_zar']:,} across {facts['policy_terms']['insured_area_ha']:,.0f} ha.",
        "**Watch items.** " + " ".join(facts["caveats"][:2]),
    ]
    return "\n\n".join(lines)


def claude_briefing(facts: dict, question: str | None = None, max_tokens: int = 1200) -> str:
    """Asks Claude to write the briefing, or to answer a question strictly from the fact sheet."""
    import anthropic

    client = anthropic.Anthropic()
    task = BRIEF_PROMPT if not question else (
        "Answer the underwriter's question using only the fact sheet. Keep it under four sentences. "
        f"If the fact sheet does not contain the answer, say exactly what is missing.\n\nQuestion: {question}"
    )
    request = dict(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM,
        output_config={"effort": "low"},  # a short briefing from supplied numbers needs little deliberation
        messages=[{"role": "user", "content": f"<fact_sheet>\n{json.dumps(facts, indent=2)}\n</fact_sheet>\n\n{task}"}],
    )
    try:
        # Server-side fallback keeps the page working if a request is declined
        response = client.beta.messages.create(
            **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    except Exception:
        response = client.messages.create(**request)
    if getattr(response, "stop_reason", None) == "refusal":
        raise RuntimeError("Claude declined to answer this request.")
    return "\n".join(block.text for block in response.content if block.type == "text").strip()


def write_briefing(facts: dict, question: str | None = None) -> tuple[str, str]:
    """Returns (text, source). Falls back to the template if Claude is unavailable."""
    if not credentials_available():
        if question:
            return ("Claude is not configured, so questions cannot be answered. Set ANTHROPIC_API_KEY to enable this.",
                    "unavailable")
        return fallback_briefing(facts), "template"
    try:
        return claude_briefing(facts, question), f"claude ({MODEL})"
    except Exception as exc:  # network, credentials, quota - the page still has to render
        if question:
            return f"Claude could not answer just now ({type(exc).__name__}).", "unavailable"
        return fallback_briefing(facts), f"template (Claude unavailable: {type(exc).__name__})"
