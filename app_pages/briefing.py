import json
import os

import streamlit as st

from src.app_data import get_context, policy_sidebar, season_label
from src.briefing import CLAUDE_MODEL, active_provider, build_facts, track_record, write_briefing
from src.trigger import TEST_START_SEASON

ctx = get_context(TEST_START_SEASON, os.path.getmtime("data/district_seasons.csv"))
terms = policy_sidebar()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_briefing(facts_json: str, question: str | None) -> tuple[str, str]:
    """One API call per (season, terms, question): a demo can click around without re-billing or re-waiting."""
    return write_briefing(json.loads(facts_json), question)

st.subheader("Underwriting briefing", anchor=False)
st.markdown("A language model reads the engine's own numbers and writes the season note an underwriter would "
            "otherwise assemble by hand — and answers questions about the season strictly from those numbers.")

season = st.select_slider("Season", options=ctx.seasons, value=ctx.current_season, format_func=season_label,
                          key="briefing_season")
facts = build_facts(ctx.df, season, terms, {
    "track_record": track_record(ctx.df, ctx.test, terms),
    "model": {"type": "RidgeCV on 10 satellite and weather features", "index_scale": float(ctx.df["index_scale"].iloc[0]),
              "trained_on": f"{season_label(ctx.seasons[0])}–{season_label(TEST_START_SEASON - 1)}"},
})

provider = active_provider()
if provider == "anthropic":
    st.caption(f":material/bolt: Connected to Claude ({CLAUDE_MODEL}). Briefings are written from the fact sheet below.")
elif provider == "gemini":
    st.caption(":material/bolt: Connected to Gemini (Google AI Studio free tier). Briefings are written from the "
               "fact sheet below.")
else:
    st.info("No model key is set, so briefings fall back to a deterministic template. Add `GEMINI_API_KEY` "
            "(free at aistudio.google.com) or `ANTHROPIC_API_KEY` to `.env` and restart.", icon=":material/key_off:")

col1, col2 = st.columns([3, 2])

with col1:
    with st.container(border=True):
        st.markdown(f"**Season note — {season_label(season)}**")
        with st.spinner("Writing the briefing…"):
            text, source = cached_briefing(json.dumps(facts, sort_keys=True), None)
        st.markdown(text)
        st.caption(f"Source: {source}. Figures come from the engine, not the model — the prompt forbids inventing "
                   "numbers or using outside knowledge about a season.")

    with st.container(border=True):
        st.markdown("**Ask about this season**")
        question = st.text_input("Question", placeholder="e.g. Which districts drive the payout, and why?",
                                 label_visibility="collapsed", key="briefing_q")
        if question:
            with st.spinner("Thinking…"):
                answer, answer_source = cached_briefing(json.dumps(facts, sort_keys=True), question)
            st.markdown(answer)
            st.caption(f"Source: {answer_source}")
        else:
            st.caption("Answers are grounded in the fact sheet only; anything outside it comes back as "
                       "\"not in the data\" rather than a guess.")

with col2:
    with st.container(border=True):
        st.markdown("**The fact sheet Claude receives**")
        st.caption("No raw records and no user text reach the model — only these computed figures.")
        st.json(facts, expanded=False)
        st.download_button("Download fact sheet (JSON)", json.dumps(facts, indent=2).encode("utf-8"),
                           file_name=f"msindex_facts_{season}.json", mime="application/json",
                           icon=":material/download:")

with st.expander("How AI is used here", icon=":material/psychology:"):
    st.markdown(
        """
- **Two models, different jobs.** A ridge regression turns satellite and weather features into the yield index that
  drives payouts — small, auditable, and testable. The language model never touches a payout decision; it explains
  them, and the engine runs either Claude or Gemini depending on which key is configured.
- **Grounding.** Every request carries a JSON fact sheet the engine computed. The system prompt forbids outside
  knowledge and invented figures, so the briefing cannot drift from the ledger.
- **Degrades safely.** Without credentials, or if a request fails, the same fact sheet renders through a
  deterministic template. The page never blocks an underwriter.
- **No personal data leaves the system.** Parcels are pseudonymised before they reach any record, and the fact
  sheet carries district aggregates only.
        """
    )
