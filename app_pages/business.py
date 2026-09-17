import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_data import get_context, policy_sidebar
from src.pricing import apply_policy, premium, risk_stats, season_losses
from src.trigger import TEST_START_SEASON

ctx = get_context(TEST_START_SEASON, os.path.getmtime("data/district_seasons.csv"))
terms = policy_sidebar()

scored = apply_policy(ctx.test, terms)
stats = risk_stats(season_losses(scored))
price = premium(stats, terms)

# National commercial maize area from the CEC release the pipeline parsed (2026 season, all provinces)
NATIONAL_MAIZE_HA = 2_724_200
MAIZE_TRIANGLE_SHARE = 0.86  # Free State, Mpumalanga, North West and Gauteng share of planted area

st.subheader("Business model", anchor=False)
st.markdown(
    """
msIndex is **underwriting infrastructure**, not an insurer. The engine turns public Earth-observation data into a
priced, auditable trigger, and licenses that capability to the parties who already carry the risk: insurers,
reinsurers, agri-lenders and grain co-operatives. That keeps regulatory capital where it belongs and makes the
business a software and data margin rather than a risk margin.
    """
)

st.markdown("#### Who pays, and for what")
REVENUE = [
    {"Line": "Index-as-a-service", "Customer": "Insurers, MGAs",
     "Basis": "Annual fee per insured hectare",
     "Why they buy": "A priced, back-tested trigger and a settlement file, without building an EO team"},
    {"Line": "Portfolio analytics", "Customer": "Reinsurers, brokers",
     "Basis": "Subscription per book reviewed",
     "Why they buy": "Independent loss distribution and correlation view at treaty renewal"},
    {"Line": "Lender drought screening", "Customer": "Agri-lenders, input financiers",
     "Basis": "Per-loan or per-portfolio scan",
     "Why they buy": "Early warning on a season before repayment falls due"},
    {"Line": "Audit & compliance", "Customer": "Insurers, regulators",
     "Basis": "Included, priced into the fee",
     "Why they buy": "Evidence a payout decision was not altered after the fact"},
]
st.dataframe(pd.DataFrame(REVENUE), hide_index=True,
             column_config={"Why they buy": st.column_config.TextColumn(width="large")})

st.markdown("#### Unit economics, from the book priced in the sidebar")
fee_per_ha = st.slider("Index-as-a-service fee (ZAR per insured hectare per season)", 1.0, 25.0, 6.0, step=0.5,
                       key="fee_ha")
book_premium = price["book_premium"]
fee_revenue = fee_per_ha * terms.hectares
share_of_premium = fee_revenue / book_premium if book_premium else 0

with st.container(horizontal=True):
    st.metric("Book premium priced", f"R {book_premium:,.0f}", border=True,
              help="From the Pricing page: expected loss plus capital charge, grossed up for expenses.")
    st.metric("msIndex revenue", f"R {fee_revenue:,.0f}", border=True,
              help=f"{terms.hectares:,.0f} ha at R {fee_per_ha:,.2f}/ha.")
    st.metric("Share of premium", f"{share_of_premium:.1%}", border=True,
              help="What the fee costs the insurer relative to the premium they collect.")
    st.metric("Insurer keeps", f"R {book_premium - fee_revenue:,.0f}", border=True)

st.caption("The fee is a fraction of premium because msIndex carries no risk: the insurer keeps the underwriting "
           "result, and pays for the trigger, the evidence and the settlement file.")

st.markdown("#### Market")
sizes = pd.DataFrame({
    "Segment": ["Commercial maize, national", "Maize triangle (this engine's region)", "Serviceable at 10% adoption"],
    "Hectares": [NATIONAL_MAIZE_HA, NATIONAL_MAIZE_HA * MAIZE_TRIANGLE_SHARE, NATIONAL_MAIZE_HA * MAIZE_TRIANGLE_SHARE * 0.10],
})
sizes["Fee revenue at this rate"] = sizes["Hectares"] * fee_per_ha
c1, c2 = st.columns([3, 2])
with c1, st.container(border=True):
    fig = px.bar(sizes, x="Hectares", y="Segment", orientation="h", height=240, text_auto=".2s")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(title=None, autorange="reversed"))
    st.plotly_chart(fig, key="market")
with c2, st.container(border=True):
    st.dataframe(sizes, hide_index=True, column_config={
        "Hectares": st.column_config.NumberColumn(format="%,.0f"),
        "Fee revenue at this rate": st.column_config.NumberColumn(format="R %,.0f")})
st.caption(f"Commercial maize area is {NATIONAL_MAIZE_HA:,} ha for the 2026 season, from the Crop Estimates "
           "Committee release this app already parses. The maize-triangle share is an estimate from the provincial "
           "area split in the same release; adoption is an assumption, not a forecast.")

st.markdown("#### Why this wins the account")
c1, c2, c3 = st.columns(3)
with c1, st.container(border=True):
    st.markdown("""
**Against a rainfall index**

Same payout budget, more real losses paid. Fewer wrong payouts means a lower loss cost for the same protection,
which is a direct underwriting result the insurer can see on the Basis risk page.
""")
with c2, st.container(border=True):
    st.markdown("""
**Against a loss adjuster**

No farm visit, no claim form. Settlement is a signed file generated from the ledger once the season closes,
which is what makes cover economical on land where adjusting costs more than the crop.
""")
with c3, st.container(border=True):
    st.markdown("""
**Against building it in-house**

The pipeline already reconciles six public datasets, follows the CEC release calendar, and ships a validated
26-season history. That is the part that takes a year, and it is done.
""")

st.markdown("#### Cost to run")
st.markdown(
    """
The entire data stack is public and free: CHIRPS, ERA5 via Open-Meteo, MODIS via Planetary Computer, ESA
WorldCover, CEC releases and HarvestStat. A full rebuild is about an hour of compute, and an in-season refresh is
minutes, so marginal cost per additional insured hectare is effectively zero. Cost scales with regions covered,
not with hectares insured — which is why the fee is per hectare and the margin grows with adoption.
    """
)

st.markdown("#### Getting to revenue")
st.markdown(
    """
| Stage | What has to be true | Where it stands |
|---|---|---|
| Pilot with one insurer | Trigger back-tested, terms priced, settlement path demonstrated | Done in this engine |
| Regulatory comfort | Decisions auditable and reproducible; POPIA handled | Ledger, fingerprints, pseudonymisation in place |
| Farm-level evidence | Basis risk measured against farm yields, not provincial ones | Needs an insurer's own yield data |
| Scale beyond maize | Same pipeline, different crop calendar and label source | Windows and thresholds are configuration |
"""
)
