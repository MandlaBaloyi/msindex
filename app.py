"""msIndex risk engine — entry point.

Pages live in app_pages/ and share data, policy terms and helpers through src/app_data.py.
The dataset is built separately with `python -m src.pipeline`.
"""
import os

import streamlit as st

from src.app_data import DATA_PATH, get_context, season_label
from src.trigger import TEST_START_SEASON

st.set_page_config(page_title="msIndex | Parametric Risk Engine", page_icon=":material/agriculture:", layout="wide")

st.title("msIndex risk engine", anchor=False)
st.caption("Contract-grade parametric maize underwriting: Earth observation & cryptographic audit trails")

pages = st.navigation({
    "Start here": [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/lightbulb:", default=True),
    ],
    "The engine": [
        st.Page("app_pages/portfolio.py", title="Portfolio risk", icon=":material/map:"),
        st.Page("app_pages/basis_risk.py", title="Basis risk", icon=":material/balance:"),
        st.Page("app_pages/pricing.py", title="Pricing & capital", icon=":material/payments:"),
        st.Page("app_pages/audit.py", title="Audit & settlement", icon=":material/verified_user:"),
        st.Page("app_pages/briefing.py", title="Underwriting briefing", icon=":material/psychology:"),
    ],
    "Behind it": [
        st.Page("app_pages/data_methods.py", title="Data & methods", icon=":material/database:"),
        st.Page("app_pages/security.py", title="Security", icon=":material/shield:"),
        st.Page("app_pages/business.py", title="Business model", icon=":material/trending_up:"),
    ],
})

if os.path.exists(DATA_PATH):
    ctx = get_context(TEST_START_SEASON, os.path.getmtime(DATA_PATH))
    with st.sidebar:
        st.caption(f"Seasons {season_label(ctx.seasons[0])}–{season_label(ctx.seasons[-1])} · "
                   f"current season {season_label(ctx.current_season)}")
        if ctx.metadata:
            st.caption(f"Data built {ctx.metadata['built_at'][:10]} · refresh with `python -m src.pipeline`")

pages.run()
