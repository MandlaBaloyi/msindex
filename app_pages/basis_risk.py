import os

import numpy as np
import plotly.express as px
import streamlit as st

from src.app_data import download_row, get_context, largest_gap, policy_sidebar, season_label, trigger_sweep
from src.pricing import apply_policy, basis_metrics
from src.trigger import LOSS_THRESHOLD, TEST_START_SEASON, evaluate_triggers

ctx = get_context(TEST_START_SEASON, os.path.getmtime("data/district_seasons.csv"))
terms = policy_sidebar()
test_first, test_last = int(ctx.test["season"].min()), int(ctx.test["season"].max())

st.subheader("Trigger performance: dual-source index vs rainfall-only", anchor=False)
st.markdown(f"Evaluated on the **{season_label(test_first)}–{season_label(test_last)} temporal holdout** "
            f"({len(ctx.test)} district-seasons scored by a model that never saw them). "
            f"A genuine loss is a reported provincial yield at least {abs(LOSS_THRESHOLD):.0%} below trend.")

rain_lo = float(np.floor(ctx.test["rain_mm"].min() / 10) * 10)
rain_hi = float(np.ceil(ctx.test["rain_mm"].max() / 10) * 10)
rain_threshold = st.slider("Rainfall-only trigger cutoff (Oct–Mar, mm)", rain_lo, rain_hi,
                           float(np.clip(ctx.default_cutoff, rain_lo, rain_hi)), step=10.0,
                           key="cutoff", bind="query-params")
_, metrics = evaluate_triggers(ctx.test, rainfall_cutoff_mm=rain_threshold)
b_met, m_met = metrics["baseline"], metrics["msindex"]

n_losses = b_met["tp"] + b_met["fn"]
st.caption(f"Matched payout budget: both triggers pay exactly **{b_met['payouts']}** district-seasons "
           f"({b_met['payout_rate']:.1%}). The holdout contains **{n_losses}** genuine loss events.")

gap_lo, gap_hi = largest_gap(ctx.test["rain_mm"])
if gap_lo <= rain_threshold <= gap_hi and gap_hi - gap_lo >= 30:
    st.caption(f":material/info: No holdout district-season had rainfall between {gap_lo:.0f} and {gap_hi:.0f} mm, "
               "so every cutoff in that range gives identical results.")

with st.container(horizontal=True):
    st.metric("Hit rate (real losses paid)", f"{m_met['hit_rate']:.1f}%",
              delta=f"{m_met['hit_rate'] - b_met['hit_rate']:+.1f} pp vs rainfall-only", border=True)
    st.metric("False payouts (share of payouts)", f"{m_met['false_payout_rate']:.1f}%",
              delta=f"{m_met['false_payout_rate'] - b_met['false_payout_rate']:+.1f} pp vs rainfall-only",
              delta_color="inverse", border=True)
    st.metric("Loss events missed (district-seasons)", f"{m_met['missed_losses']}",
              delta=f"{m_met['missed_losses'] - b_met['missed_losses']:+d} vs rainfall-only",
              delta_color="inverse", border=True)

diff = m_met["hit_rate"] - b_met["hit_rate"]
if diff > 0:
    st.success(f"At the same payout budget, msIndex pays **{m_met['hit_rate']:.1f}%** of genuine losses vs "
               f"**{b_met['hit_rate']:.1f}%** for rainfall-only, cutting missed losses from "
               f"**{b_met['missed_losses']} to {m_met['missed_losses']}**.", icon=":material/trending_up:")
elif diff < 0:
    st.warning(f"At this cutoff the rainfall-only trigger does better: **{b_met['hit_rate']:.1f}%** vs "
               f"**{m_met['hit_rate']:.1f}%** of genuine losses paid.", icon=":material/trending_down:")
else:
    st.info(f"At this cutoff both triggers pay the same share of genuine losses (**{m_met['hit_rate']:.1f}%**). "
            "The chart below shows whether any cutoff separates the two.", icon=":material/drag_handle:")

chart_col1, chart_col2 = st.columns(2)
sweep = trigger_sweep(ctx.test, tuple(float(c) for c in np.arange(rain_lo, rain_hi + 10, 10)))
with chart_col1, st.container(border=True):
    st.markdown("**Hit rate across rainfall cutoffs** (matched payout budget at each point)")
    st.line_chart(sweep, x="Rainfall cutoff (mm)", y=["Rainfall only", "msIndex"],
                  x_label="Rainfall-only cutoff (mm)", y_label="Real losses paid (%)", height=300)
    better = int((sweep["msIndex"] > sweep["Rainfall only"]).sum())
    worse = int((sweep["msIndex"] < sweep["Rainfall only"]).sum())
    st.caption(f"msIndex is ahead at {better} cutoffs, behind at {worse}, and level at {len(sweep) - better - worse}. "
               "Ties happen where the budget is generous enough that both triggers catch every loss.")

with chart_col2, st.container(border=True):
    st.markdown("**Model check: provincial yield anomaly, predicted vs reported**")
    prov_series = (ctx.df.groupby(["province", "season"])
                   .agg(Reported=("yield_anomaly", "first"), Predicted=("pred_yield_anomaly", "mean"))
                   .reset_index().melt(id_vars=["province", "season"], var_name="Series", value_name="Yield anomaly"))
    check_fig = px.line(prov_series, x="season", y="Yield anomaly", color="Series", facet_row="province",
                        height=300, markers=True, labels={"season": "Season (planting year)"},
                        color_discrete_map={"Reported": "#555", "Predicted": "#2C7BB6"})
    check_fig.add_vrect(x0=TEST_START_SEASON - 0.5, x1=test_last + 0.5, fillcolor="grey", opacity=0.1, line_width=0)
    check_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    check_fig.update_yaxes(tickformat="+.0%", title_text="")
    check_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.08, title=None))
    st.plotly_chart(check_fig, key="model_check")
    st.caption("Shaded: test seasons (out-of-sample). Predicted = mean of the province's four districts.")

# How much of each loss the graded policy actually covers, which a hit rate cannot show
st.subheader("Cover quality under the policy ladder", anchor=False)
scored_test = apply_policy(ctx.test, terms)
bm = basis_metrics(scored_test)
with st.container(horizontal=True):
    st.metric("Loss events paid", f"{bm['paid_when_loss']:.0f}%", border=True,
              help="Share of district-seasons with a genuine shortfall that received any payout.")
    st.metric("Median cover of the loss", f"{bm['coverage_ratio']:.0%}", border=True,
              help="Payout divided by what an indemnity policy on the same ladder would have paid.")
    st.metric("Losses under half covered", f"{bm['under_half_covered']:.0f}%", delta_color="inverse", border=True,
              help="Downside basis risk: the farmer suffers a loss and the index pays less than half of it.")
    st.metric("Payout vs loss correlation", f"{bm['payout_loss_corr']:.2f}", border=True)

scatter = px.scatter(scored_test, x="loss_share", y="payout_share", color="province",
                     hover_name="district", hover_data={"season": True},
                     labels={"loss_share": "Loss the farmer suffered (share of cover)",
                             "payout_share": "Payout made (share of cover)"}, height=340)
scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
scatter.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.12, title=None))
with st.container(border=True):
    st.plotly_chart(scatter, key="basis_scatter")
    st.caption("Each point is a holdout district-season. On the dashed line the payout matches the loss exactly; "
               "below it the farmer is short-changed, above it the insurer overpays.")
    download_row(scored_test[["district", "province", "season", "yield_anomaly", "pred_yield_anomaly",
                              "loss_share", "payout_share"]], "msindex_basis_risk.csv")
