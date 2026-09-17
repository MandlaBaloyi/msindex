import os

import streamlit as st

from src.app_data import get_context, policy_sidebar, season_label
from src.pricing import apply_policy, basis_metrics, premium, risk_stats, season_losses
from src.trigger import LOSS_THRESHOLD, TEST_START_SEASON, evaluate_triggers

ctx = get_context(TEST_START_SEASON, os.path.getmtime("data/district_seasons.csv"))
terms = policy_sidebar()

_, metrics = evaluate_triggers(ctx.test, rainfall_cutoff_mm=ctx.default_cutoff)
scored = apply_policy(ctx.test, terms)
stats = risk_stats(season_losses(scored))
price = premium(stats, terms)
bm = basis_metrics(scored)

st.subheader("A drought that pays out in days, not months", anchor=False)
st.markdown(
    """
South African maize farmers carry drought risk almost alone: traditional multi-peril cover needs a loss
adjuster on every farm, which costs more than the crop is worth on smallholder and mid-sized land. The usual
answer is a **rainfall index** — cheap to run, but it pays the wrong farms. Rain that falls at the wrong time,
or heat that cooks a crop that got its rain, is invisible to it.

**msIndex reads the crop itself.** It combines rainfall, evaporative demand, heat and the satellite greenness of
each district's cropland into one index of yield shortfall, and seals every payout decision in a cryptographic
ledger an auditor or regulator can verify.
    """
)

with st.container(horizontal=True):
    st.metric("Real losses paid", f"{metrics['msindex']['hit_rate']:.0f}%",
              delta=f"{metrics['msindex']['hit_rate'] - metrics['baseline']['hit_rate']:+.0f} pp vs rainfall index",
              border=True, help="Out-of-sample, at an identical payout budget.")
    st.metric("Seasons of evidence", f"{ctx.df['season'].nunique()}", border=True,
              help=f"{season_label(ctx.seasons[0])} to {season_label(ctx.seasons[-1])}, rebuilt from source each run.")
    st.metric("Gross premium", f"R {price['gross_premium_ha']:,.0f}/ha",
              delta=f"{price['rate_on_line']:.1%} rate on line", delta_color="off", border=True)
    st.metric("Decisions sealed", f"{len(ctx.current)} districts", border=True,
              help="Every decision this season carries an HMAC signature and the model fingerprint that made it.")

st.subheader("What is genuinely new here", anchor=False)
c1, c2, c3 = st.columns(3)
with c1, st.container(border=True):
    st.markdown(f"""
**A dual-source index, proven against the rainfall standard**

At a matched payout budget on {season_label(int(ctx.test['season'].min()))}–{season_label(int(ctx.test['season'].max()))}
seasons the engine never saw, msIndex pays **{metrics['msindex']['hit_rate']:.0f}%** of genuine losses against
**{metrics['baseline']['hit_rate']:.0f}%** for rainfall alone, and is ahead or level at every cutoff tested.
""")
with c2, st.container(border=True):
    st.markdown("""
**Decisions that survive a dispute**

Each payout decision is sealed with its features, the cutoff, and a fingerprint of the exact model and dataset
that produced it, chained by HMAC-SHA256. Editing one historical decision breaks every record after it, and the
settlement file will not generate from a broken ledger.
""")
with c3, st.container(border=True):
    st.markdown("""
**A live index, not a one-off study**

One command rebuilds the dataset from source — satellite imagery, rainfall, evapotranspiration and the latest
Crop Estimates Committee release — so the index moves with the season instead of ageing in a spreadsheet.
""")

st.subheader("How it works", anchor=False)
st.markdown(
    f"""
1. **Observe.** For each of 12 districts and every season since {ctx.seasons[0]}, the pipeline pulls CHIRPS
   rainfall, ERA5 temperature and evapotranspiration, and MODIS greenness masked to ESA WorldCover cropland.
2. **Learn.** A regularised linear model maps those signals to the provincial yield shortfall against trend,
   trained only on seasons before {TEST_START_SEASON}, then rescaled so the index reads in real yield units.
3. **Trigger.** A payout ladder turns the index into money: nothing above {terms.attachment:+.0%}, rising to the
   full limit at {terms.exhaustion:+.0%}.
4. **Price.** Expected loss plus the capital the tail demands, because drought hits the whole book at once.
5. **Seal and settle.** Decisions enter the hash-chained ledger; a signed payment batch is generated from it.
    """
)

st.subheader("What we are not claiming", anchor=False)
st.warning(
    f"""
Yields are published per province, so all four districts of a province share one reported yield — this is a
provincial index, not a farm-level one. The holdout contains **{bm['loss_events']}** loss events, concentrated in
the 2023/24 drought, so the comparison rests on few independent events. The model understates the deepest
droughts, and **{bm['under_half_covered']:.0f}%** of losses in the holdout are covered less than half. Every one of
those numbers is on the Basis risk page rather than buried here.
    """,
    icon=":material/balance:",
)

st.caption("Use the sidebar to set policy terms; every page prices and settles against them. "
           "Start with Portfolio risk for the map, then Basis risk for the evidence.")
