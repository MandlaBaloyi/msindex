import os

import pandas as pd
import streamlit as st

from src.app_data import DATA_PATH, get_context, policy_sidebar, season_label
from src.ledger import batch_manifest, load, model_version, payment_batch, save, season_root
from src.model import FEATURE_COLS, make_model
from src.trigger import TEST_START_SEASON, USING_FALLBACK_SECRETS, AuditChain

TAMPER_INDEX = 1

ctx = get_context(TEST_START_SEASON, os.path.getmtime(DATA_PATH))
terms = policy_sidebar()


@st.cache_resource
def fitted_version(data_mtime: float) -> str:
    """Fingerprint of the model that scores the current season, bound into every record it signs."""
    labelled = ctx.labelled
    model = make_model().fit(labelled[FEATURE_COLS], labelled["yield_anomaly"])
    return model_version(model, FEATURE_COLS, DATA_PATH)


def build_chain() -> AuditChain:
    chain = AuditChain()
    version = fitted_version(os.path.getmtime(DATA_PATH))
    for _, r in ctx.current.sort_values("district").iterrows():
        chain.append_decision(
            parcel_id=f"PARCEL-ZA-{r['province'][:2].upper()}-{r['district'][:4].upper()}-{int(r['season'])}",
            season=int(r["season"]),
            features={col: r[col] for col in FEATURE_COLS},
            prediction=r["pred_yield_anomaly"],
            cutoff=ctx.index_cutoff,
            decision=int(terms.payout_share(r["pred_yield_anomaly"]) > 0),
            model_version=version,
        )
    save(chain, ctx.current_season)
    return chain


if "audit_chain" not in st.session_state:
    restored = load(ctx.current_season)
    st.session_state.audit_chain = restored or build_chain()
    st.session_state.tampered = restored is not None and not restored.verify()[0]

st.subheader("Cryptographic decision log (HMAC-SHA256)", anchor=False)
st.markdown(f"This season's ({season_label(ctx.current_season)}) payout decisions for every district, each sealed "
            "with the model fingerprint, the dataset fingerprint, the features and the previous record's hash.")

if USING_FALLBACK_SECRETS:
    st.warning("MSINDEX_HMAC_KEY / MSINDEX_PARCEL_SALT are not set. Using random ephemeral secrets; signatures will "
               "not verify after a restart.", icon=":material/key_off:")

chain = st.session_state.audit_chain
is_valid, broken_idx = chain.verify()

if is_valid:
    st.success("Decision ledger verified: all hash links and HMAC signatures intact.", icon=":material/verified:")
else:
    st.error(f"Tampering detected at record #{broken_idx}. The cryptographic hash chain is broken.",
             icon=":material/gpp_bad:")

with st.container(horizontal=True):
    st.metric("Season root hash", season_root(chain)[:18] + "…", border=True,
              help="Publishing this one hash fixes every record beneath it.")
    st.metric("Records", f"{len(chain.chain)}", border=True)
    st.metric("Model version", (chain.chain[0]["payload"]["model_version"][:22] + "…") if chain.chain else "—",
              border=True, help="Model coefficients and dataset fingerprint, so a record names the model that made it.")

with st.container(horizontal=True):
    if st.button(f"Simulate insider tampering (flip record #{TAMPER_INDEX})", icon=":material/edit_note:",
                 disabled=st.session_state.tampered or len(chain.chain) <= TAMPER_INDEX):
        chain.tamper_decision(TAMPER_INDEX)
        save(chain, ctx.current_season)
        st.session_state.tampered = True
        st.rerun()
    if st.button("Rebuild ledger from the engine", icon=":material/restart_alt:"):
        st.session_state.audit_chain = build_chain()
        st.session_state.tampered = False
        st.rerun()

audit_rows = [{
    "Index": r["index"],
    "Pseudonym": r["payload"]["parcel_pseudonym"],
    "Season": season_label(r["payload"]["season"]),
    "Predicted anomaly": f"{r['payload']['prediction']:+.2%}",
    "Payout": f"{terms.payout_share(r['payload']['prediction']):.0%}",
    "Decision": "PAYOUT" if r["payload"]["decision"] == 1 else "NO PAY",
    "Previous hash": r["prev_hash"][:14] + "…",
    "Record hash": r["record_hash"][:14] + "…",
} for r in chain.chain]
st.dataframe(pd.DataFrame(audit_rows), hide_index=True)

st.subheader("Settlement", anchor=False)
if not is_valid:
    st.warning("A payment batch cannot be produced while the ledger is broken — rebuild it first.",
               icon=":material/block:")
else:
    try:
        csv_text, signature, summary = payment_batch(chain, terms, ctx.current_season)
    except ValueError as exc:
        st.error(str(exc), icon=":material/error:")
    else:
        with st.container(horizontal=True):
            st.metric("Payments in batch", f"{summary['payments']}", border=True)
            st.metric("Total to settle", f"R {summary['total_zar']:,.0f}", border=True)
            st.metric("Area per parcel", f"{terms.hectares_per_parcel:,.0f} ha", border=True)
        st.caption("The batch is generated only from a ledger that verifies, and is signed with the same key as the "
                   "decisions, so a treasury system can confirm it came from this engine unedited.")
        b1, b2 = st.columns(2)
        b1.download_button("Download payment batch (CSV)", csv_text.encode("utf-8"),
                           file_name=f"msindex_payments_{ctx.current_season}.csv", mime="text/csv",
                           icon=":material/download:", disabled=summary["payments"] == 0)
        b2.download_button("Download signed manifest (JSON)", batch_manifest(summary).encode("utf-8"),
                           file_name=f"msindex_payments_{ctx.current_season}_manifest.json", mime="application/json",
                           icon=":material/receipt_long:")
        if summary["payments"] == 0:
            st.caption(f"No district triggers a payout in {season_label(ctx.current_season)}, so the batch is empty.")
        else:
            st.dataframe(pd.read_csv(pd.io.common.StringIO(csv_text)), hide_index=True)
