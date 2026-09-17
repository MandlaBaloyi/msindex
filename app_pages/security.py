import hashlib
import os

import pandas as pd
import streamlit as st

from src.app_data import DATA_PATH, get_context, policy_sidebar
from src.ledger import dataset_fingerprint, ledger_path, load, season_root
from src.trigger import TEST_START_SEASON, USING_FALLBACK_SECRETS, AuditChain, pseudonymise_parcel

ctx = get_context(TEST_START_SEASON, os.path.getmtime(DATA_PATH))
policy_sidebar()

st.subheader("Security & data protection", anchor=False)
st.markdown(
    "A payout engine is a financial control: it decides who gets money. The threats that matter are an insider "
    "quietly editing a decision, a disputed payout with no evidence, personal data leaking into a shared log, and "
    "a settlement file altered in transit. Each is addressed below, with what is deliberately still open."
)

chain = load(ctx.current_season) or AuditChain()
valid, broken = chain.verify() if chain.chain else (True, -1)

with st.container(horizontal=True):
    st.metric("Ledger integrity", "Verified" if valid else f"Broken at #{broken}", border=True,
              help="Every record recomputed from its payload, key and predecessor.")
    st.metric("Sealed decisions", f"{len(chain.chain)}", border=True)
    st.metric("Dataset fingerprint", dataset_fingerprint(DATA_PATH), border=True,
              help="SHA-256 of the dataset the decisions were made from.")
    st.metric("Secrets", "Ephemeral" if USING_FALLBACK_SECRETS else "Configured", border=True,
              delta="set MSINDEX_HMAC_KEY" if USING_FALLBACK_SECRETS else None, delta_color="inverse")

st.markdown("#### Controls in place")
CONTROLS = [
    ("Tamper-evident decisions", "HMAC-SHA256 hash chain",
     "Each record seals its payload plus the previous record's hash. Editing a historical decision invalidates that "
     "record and every one after it, and `verify()` names the first broken index."),
    ("Provenance of every decision", "Model and dataset fingerprints",
     "Records carry a hash of the fitted model coefficients and of the dataset file, so a decision names the exact "
     "model that made it. A retrained model cannot silently reuse a version string."),
    ("Personal data minimisation (POPIA)", "Salted SHA-256 pseudonyms",
     "Parcel identifiers are salted and hashed before they enter any record, log or payment file. The ledger can be "
     "audited without carrying identifying references."),
    ("Settlement integrity", "Signed payment batches",
     "A payment file is generated only from a ledger that verifies, and is signed with the same key, so a treasury "
     "system can confirm it came from this engine unedited."),
    ("Secret handling", "Environment variables, never the repo",
     "Keys come from `.env`, which is git-ignored. If they are missing the engine uses a random per-process value "
     "and warns loudly rather than falling back to a known constant."),
    ("Supply chain", "Pinned dependencies, no account-bound data",
     "Every dependency is pinned in `requirements.txt`. Data comes from public scientific archives over HTTPS; no "
     "credentials or third-party SaaS sit in the data path."),
    ("Input validation", "Build-time schema and range checks",
     "Duplicate keys, missing district-seasons and out-of-range values fail the build, so a malformed source release "
     "cannot silently reach a payout decision."),
    ("Blast radius", "Read-only app, offline data",
     "The app reads one CSV and never writes to a source system or calls a network service; the pipeline runs "
     "separately and under human control."),
]
st.dataframe(pd.DataFrame(CONTROLS, columns=["Risk addressed", "Control", "How it works"]), hide_index=True,
             column_config={"How it works": st.column_config.TextColumn(width="large")})

st.markdown("#### Prove it: the chain refuses to lie")
st.markdown(
    "The Audit & settlement page has a working demonstration — flip one historical decision and the ledger names "
    "the broken record, the settlement file refuses to generate, and rebuilding restores a verified chain. The same "
    "behaviour is asserted in the test suite, not just in the demo."
)
demo = AuditChain()
for i in range(3):
    demo.append_decision(f"PARCEL-DEMO-{i}", ctx.current_season, {"rain_mm": 400.0 + i}, -0.2 - i / 100, -0.15, i % 2)
before_root = season_root(demo)
demo.tamper_decision(1)
after_valid, after_idx = demo.verify()
st.code(
    f"chain of {len(demo.chain)} records\n"
    f"root before tampering : {before_root[:32]}…\n"
    f"flip decision #1      : verify() -> ({after_valid}, {after_idx})\n"
    f"payment batch          : refused — 'ledger does not verify at record #{after_idx}'",
    language="text")
st.caption(f"Pseudonymisation in the same run: `PARCEL-DEMO-0` → `{pseudonymise_parcel('PARCEL-DEMO-0')}` "
           "(salted, truncated; the original never reaches a record).")

st.markdown("#### Deliberately still open")
st.markdown(
    """
- **No sign-in.** Anyone who can reach the app can see it. Streamlit's OIDC login (`st.login`) would separate
  underwriter from auditor; it needs an identity provider, so it is configuration rather than code.
- **The ledger is local.** It survives restarts as append-only JSONL, but a production deployment would write to
  append-only storage (WORM or object-lock) and publish each season's root hash externally, so a tampered *file*
  is as detectable as a tampered record.
- **One signing key, no rotation schedule.** Key rotation needs a key id per record and a re-verification path.
- **No encryption at rest** beyond the host filesystem, and no access logging of who viewed what.
- **Not yet independently reviewed.** The cryptography is standard library HMAC-SHA256, but the design has had no
  external security review.
    """
)

if USING_FALLBACK_SECRETS:
    st.error("This instance is running with ephemeral secrets: signatures will not verify after a restart. "
             "Set `MSINDEX_HMAC_KEY` and `MSINDEX_PARCEL_SALT` in `.env` before any real use.",
             icon=":material/key_off:")
else:
    st.success(f"Signing key and salt are configured. Current season ledger: `{ledger_path(ctx.current_season).name}`, "
               f"root `{season_root(chain)[:24]}…`", icon=":material/lock:")
