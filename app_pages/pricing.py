import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_data import download_row, get_context, policy_sidebar, season_label
from src.pricing import (COST_OF_CAPITAL, apply_policy, basis_metrics, district_correlation, premium, reinsurance_split,
                         risk_stats, season_losses)
from src.termsheet import build_term_sheet
from src.trigger import TEST_START_SEASON

ctx = get_context(TEST_START_SEASON, os.path.getmtime("data/district_seasons.csv"))
terms = policy_sidebar()

st.subheader("Parametric book pricing", anchor=False)

c1, c2, c3 = st.columns(3)
expense_ratio = c1.slider("Expense ratio (% of gross premium)", 5, 40, 20, key="expenses") / 100
cost_of_capital = c2.slider("Cost of capital (% a year)", 5, 30, int(COST_OF_CAPITAL * 100), key="coc") / 100
basis = c3.segmented_control("Experience used", options=["Holdout seasons", "All seasons"],
                             default="Holdout seasons", required=True, key="basis",
                             help="Holdout seasons are out-of-sample; all seasons include in-sample predictions.")

experience = ctx.test if basis == "Holdout seasons" else ctx.labelled
scored = apply_policy(experience, terms)
losses = season_losses(scored)
stats = risk_stats(losses)
price = premium(stats, terms, expense_ratio=expense_ratio, cost_of_capital=cost_of_capital)

with st.container(horizontal=True):
    st.metric("Expected loss / ha", f"R {price['expected_loss_ha']:,.0f}", border=True,
              help="Average payout per hectare per season over the experience period.")
    st.metric("Capital charge / ha", f"R {price['capital_charge_ha']:,.0f}", border=True,
              help=f"{cost_of_capital:.0%} of the capital held for the 1-in-200 tail.")
    st.metric("Gross premium / ha", f"R {price['gross_premium_ha']:,.0f}", border=True,
              help="Expected loss plus capital charge, grossed up for expenses.")
    st.metric("Rate on line", f"{price['rate_on_line']:.1%}", border=True,
              help="Gross premium as a share of the sum insured.")

with st.container(horizontal=True):
    st.metric("Book premium", f"R {price['book_premium']:,.0f}", border=True)
    st.metric("Capital required", f"R {price['book_capital']:,.0f}", border=True)
    st.metric("Worst season observed", f"{stats['worst_observed']:.0%} of cover", border=True,
              help=f"Season {season_label(stats['worst_season'])} — the heaviest loss in the experience period.")
    st.metric("Modelled 1-in-200 (TVaR)", f"{stats['tvar_995']:.0%} of cover", border=True,
              help="Bootstrap of the observed seasons. With so few seasons this is indicative, not measured.")

left, right = st.columns(2)

with left, st.container(border=True):
    st.markdown("**Payout ladder**")
    idx = np.linspace(-0.6, 0.1, 200)
    ladder = pd.DataFrame({"Yield shortfall": idx, "Payout": terms.payout_share(idx) * terms.sum_insured_ha})
    fig = px.line(ladder, x="Yield shortfall", y="Payout", height=280,
                  labels={"Payout": "Payout (ZAR / ha)"})
    fig.add_vline(x=terms.attachment, line_dash="dash", line_color="grey", annotation_text="attachment")
    fig.add_vline(x=terms.exhaustion, line_dash="dash", line_color="grey", annotation_text="exhaustion")
    fig.update_xaxes(tickformat="+.0%")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, key="ladder_chart")
    st.caption(f"Nothing is paid above {terms.attachment:+.0%}; cover rises proportionally to the full limit at "
               f"{terms.exhaustion:+.0%}. Payouts below {terms.min_payout_share:.0%} of cover are not made.")

with right, st.container(border=True):
    st.markdown("**Portfolio loss by season**")
    loss_df = losses.reset_index().rename(columns={"payout_share": "loss"})
    bar = px.bar(loss_df, x="season", y="loss", height=280,
                 labels={"season": "Season (planting year)", "loss": "Payout (share of cover)"})
    bar.add_hline(y=stats["expected_loss"], line_dash="dash", line_color="#2C7BB6",
                  annotation_text="expected loss", annotation_position="top left")
    bar.add_hline(y=stats["tvar_995"], line_dash="dot", line_color="#D7191C",
                  annotation_text="modelled 1-in-200", annotation_position="bottom right")
    bar.update_yaxes(tickformat=".0%")
    bar.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(bar, key="season_losses")
    st.caption(f"{stats['payout_seasons']} of {stats['seasons']} seasons produced a payout. "
               "Drought hits the whole book at once, so these are lumpy rather than smooth.")

corr = district_correlation(scored)
st.info(f"**Diversification is limited.** The average correlation between district payouts is **{corr:.2f}**: "
        "the districts sit in one drought region, so a bad season hits nearly all of them together. That is why "
        "the capital charge, not the expected loss, dominates the premium.", icon=":material/hub:")

with st.container(border=True):
    st.markdown("**Aggregate stop-loss reinsurance**")
    r1, r2 = st.columns(2)
    attach = r1.slider("Attachment (portfolio loss)", 0.0, 1.0, 0.25, step=0.05, key="re_attach")
    exhaust = r2.slider("Exhaustion (portfolio loss)", 0.0, 1.0, 0.75, step=0.05, key="re_exhaust")
    split = reinsurance_split(losses, attach, max(exhaust, attach))
    stacked = go.Figure()
    stacked.add_bar(x=split["season"], y=split["retained"], name="Retained", marker_color="#2C7BB6")
    stacked.add_bar(x=split["season"], y=split["ceded"], name="Ceded to reinsurer", marker_color="#FDAE61")
    stacked.update_layout(barmode="stack", height=280, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(tickformat=".0%", title="Share of cover"),
                          xaxis=dict(title="Season (planting year)"),
                          legend=dict(orientation="h", y=1.1, title=None))
    st.plotly_chart(stacked, key="reinsurance")
    ceded_mean = split["ceded"].mean()
    st.caption(f"The layer takes {ceded_mean:.1%} of cover per season on average "
               f"({ceded_mean / max(stats['expected_loss'], 1e-9):.0%} of the expected loss), capping the retained "
               f"loss at {attach:.0%} of cover.")

with st.expander("Assumptions behind these numbers", icon=":material/info:"):
    st.markdown(
        f"- Experience: **{basis.lower()}** ({losses.index.min()}–{losses.index.max()}, {stats['seasons']} seasons), "
        "one row per district-season, the book spread equally across the 12 districts.\n"
        f"- Capital is held to the bootstrapped **{1 - 0.995:.1%} tail (TVaR)** of the seasonal loss distribution, "
        f"charged at **{cost_of_capital:.0%}**. With {stats['seasons']} seasons of experience the tail is indicative.\n"
        "- The index is the predicted yield shortfall against the provincial trend, not a measured farm yield.\n"
        "- No reinstatement, no multi-year terms, and the sum insured is a fixed rand amount rather than tracking "
        "the maize price."
    )
    download_row(pd.DataFrame([{**stats, **price, "expense_ratio": expense_ratio, "cost_of_capital": cost_of_capital,
                                "attachment": terms.attachment, "exhaustion": terms.exhaustion,
                                "sum_insured_ha": terms.sum_insured_ha, "hectares": terms.hectares}]),
                 "msindex_pricing.csv", "Download pricing basis (CSV)")

st.download_button(
    "Download term sheet (PDF)",
    build_term_sheet(terms, stats, price, basis_metrics(scored), {
        "basis": basis,
        "worst_season_label": season_label(stats["worst_season"]) if stats["worst_season"] else "—",
        "data_note": (f"CHIRPS rainfall, ERA5 temperature and evapotranspiration, MODIS NDVI and CEC/HarvestStat "
                      f"yields, {season_label(ctx.seasons[0])}–{season_label(ctx.seasons[-1])}"),
    }),
    file_name=f"msindex_term_sheet_{ctx.current_season}.pdf", mime="application/pdf",
    icon=":material/description:", type="primary",
)
