"""Shared data loading, policy controls and small helpers for the Streamlit pages."""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from .model import SPLIT_CURRENT, SPLIT_TEST, SPLIT_TRAIN, score
from .pricing import PolicyTerms
from .trigger import classify_risk, evaluate_triggers

DATA_PATH = "data/district_seasons.csv"
METADATA_PATH = "data/metadata.json"
OVERLAY_DIR = "static/overlays"        # written by the pipeline
OVERLAY_URL = "app/static/overlays"    # the same files as the browser sees them
LEGEND_FILE = f"{OVERLAY_DIR}/legend.png"

# ColorBrewer RdYlBu: readable with red-green colour blindness, unlike the usual red-to-green ramp
RISK_COLORS = {"Catastrophic": "#D7191C", "Severe": "#FDAE61", "Moderate": "#ABD9E9", "Low": "#2C7BB6"}
RISK_RGB = {"Catastrophic": [215, 25, 28], "Severe": [253, 174, 97], "Moderate": [171, 217, 233], "Low": [44, 123, 182]}
GREENNESS_SCALE = "BrBG"  # brown = below normal (drought), green = above normal
DEFAULT_PAYOUT_QUANTILE = 0.2  # default rainfall cutoff pays roughly the driest fifth of training seasons


def season_label(season: int) -> str:
    return f"{season}/{str(season + 1)[-2:]}"


@st.cache_data
def _load_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def _load_json(path: str, mtime: float) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def _score(path: str, mtime: float, test_start: int) -> pd.DataFrame:
    df = score(_load_csv(path, mtime), test_start)
    df["Risk band"] = df["pred_yield_anomaly"].apply(classify_risk)
    return df


@st.cache_data
def trigger_sweep(test_df: pd.DataFrame, cutoffs: tuple) -> pd.DataFrame:
    rows = []
    for c in cutoffs:
        _, m = evaluate_triggers(test_df, rainfall_cutoff_mm=c)
        rows.append({"Rainfall cutoff (mm)": c, "Rainfall only": m["baseline"]["hit_rate"], "msIndex": m["msindex"]["hit_rate"]})
    return pd.DataFrame(rows)


def largest_gap(values: pd.Series) -> tuple[float, float]:
    v = np.sort(values.to_numpy())
    i = int(np.argmax(np.diff(v)))
    return float(v[i]), float(v[i + 1])


class Context:
    """Everything the pages share: the scored dataset, its splits and the default calibration."""

    def __init__(self, test_start: int):
        if not os.path.exists(DATA_PATH):
            st.error(f"Dataset not found at '{DATA_PATH}'. Build it with `python -m src.pipeline`.", icon=":material/error:")
            st.stop()
        self.df = _score(DATA_PATH, os.path.getmtime(DATA_PATH), test_start)
        self.metadata = _load_json(METADATA_PATH, os.path.getmtime(METADATA_PATH)) if os.path.exists(METADATA_PATH) else {}
        overlays = f"{OVERLAY_DIR}/overlays.json"
        self.overlay_meta = _load_json(overlays, os.path.getmtime(overlays)) if os.path.exists(overlays) else {}
        self.train = self.df[self.df["Data split"] == SPLIT_TRAIN]
        self.test = self.df[self.df["Data split"] == SPLIT_TEST].copy()
        self.labelled = self.df[self.df["Data split"] != SPLIT_CURRENT]
        self.current_season = int(self.df["season"].max())
        self.current = self.df[self.df["season"] == self.current_season]
        self.seasons = sorted(self.df["season"].unique().tolist())
        self.default_cutoff = float(np.round(self.train["rain_mm"].quantile(DEFAULT_PAYOUT_QUANTILE), -1))
        _, self.default_metrics = evaluate_triggers(self.test, rainfall_cutoff_mm=self.default_cutoff)
        self.index_cutoff = self.default_metrics["pred_cutoff"]


@st.cache_resource
def get_context(test_start: int, data_mtime: float) -> Context:
    return Context(test_start)


def policy_sidebar(defaults: PolicyTerms | None = None) -> PolicyTerms:
    """Contract terms live in the sidebar because pricing, basis risk and settlement all use them."""
    d = defaults or PolicyTerms()
    with st.sidebar:
        st.subheader("Policy terms", anchor=False)
        sum_insured = st.number_input("Sum insured (ZAR / ha)", value=int(d.sum_insured_ha), step=1000, min_value=0,
                                      key="sum_insured", help="Cover per hectare when the index reaches exhaustion.")
        hectares = st.number_input("Insured area (ha)", value=int(d.hectares), step=5000, min_value=0, key="hectares")
        # A range slider hands back (low, high), so exhaustion (the deeper shortfall) comes first
        exhaustion, attachment = st.slider(
            "Payout ladder (yield shortfall)", -0.60, 0.0, (d.exhaustion, d.attachment), step=0.01, key="ladder",
            format="%+.0f%%",
            help="Cover starts at the right-hand value (attachment) and reaches the full limit at the left-hand one "
                 "(exhaustion).",
        )
        limit = st.slider("Limit (share of sum insured)", 0.25, 1.0, d.limit_share, step=0.05, key="limit")
        franchise = st.slider("Minimum payout (share)", 0.0, 0.25, d.min_payout_share, step=0.01, key="franchise",
                              help="Payouts below this share are not made, to avoid uneconomic settlements.")
    return PolicyTerms(sum_insured_ha=float(sum_insured), attachment=float(attachment), exhaustion=float(exhaustion),
                       limit_share=float(limit), min_payout_share=float(franchise), hectares=float(hectares),
                       parcels=12)


def download_row(df: pd.DataFrame, filename: str, label: str = "Download this table (CSV)") -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv",
                       icon=":material/download:")
