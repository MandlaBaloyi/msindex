import os

import pandas as pd
import streamlit as st

from src.app_data import DATA_PATH, download_row, get_context, policy_sidebar, season_label
from src.model import FEATURE_COLS
from src.trigger import TEST_START_SEASON

ctx = get_context(TEST_START_SEASON, os.path.getmtime(DATA_PATH))
policy_sidebar()

SOURCES = [
    {
        "Dataset": "CHIRPS v2.0 daily rainfall",
        "Publisher": "Climate Hazards Center, UC Santa Barbara",
        "Used for": "Seasonal and flowering rainfall, dry spells",
        "Resolution": "0.05° (~5 km), daily",
        "Access": "data.chc.ucsb.edu — cloud-optimised GeoTIFFs, no account",
        "Citation": "Funk et al. (2015), Scientific Data 2:150066",
    },
    {
        "Dataset": "ERA5 reanalysis",
        "Publisher": "ECMWF / Copernicus, via Open-Meteo archive",
        "Used for": "Daily maximum temperature, FAO-56 reference evapotranspiration",
        "Resolution": "~9–31 km, daily",
        "Access": "archive-api.open-meteo.com, no account",
        "Citation": "Hersbach et al. (2020), QJRMS 146:1999–2049",
    },
    {
        "Dataset": "MODIS MOD13Q1 + MYD13Q1 v6.1 NDVI",
        "Publisher": "NASA LP DAAC, via Microsoft Planetary Computer",
        "Used for": "Cropland greenness: peak, integral, anomaly, green-up date",
        "Resolution": "250 m, 16-day (8-day effective with Terra + Aqua)",
        "Access": "planetarycomputer.microsoft.com STAC, anonymous token",
        "Citation": "Didan (2021), NASA EOSDIS LP DAAC, doi:10.5067/MODIS/MOD13Q1.061",
    },
    {
        "Dataset": "ESA WorldCover 2021 v200",
        "Publisher": "European Space Agency",
        "Used for": "Cropland mask — only farmland pixels contribute",
        "Resolution": "10 m, single epoch",
        "Access": "Planetary Computer STAC",
        "Citation": "Zanaga et al. (2022), doi:10.5281/zenodo.7254221",
    },
    {
        "Dataset": "Crop Estimates Committee releases",
        "Publisher": "Department of Agriculture (DALRRD), published via SAGIS",
        "Used for": "Provincial maize area and production: current and previous season",
        "Resolution": "Province, per season",
        "Access": "sagis.org.za — monthly PDF releases, parsed automatically",
        "Citation": "CEC release of 26 August 2026 (and successors)",
    },
    {
        "Dataset": "HarvestStat Africa v1.2",
        "Publisher": "Lee et al., from FEWS NET and CEC statistics",
        "Used for": "Provincial maize area and production, 1986–2024 plantings",
        "Resolution": "Admin-1 (province), per season",
        "Access": "github.com/HarvestStat/HarvestStat-Africa",
        "Citation": "Lee et al. (2025), Scientific Data, doi:10.1038/s41597-025-05001-z",
    },
    {
        "Dataset": "Abstract of Agricultural Statistics 2018",
        "Publisher": "DAFF (supplied by the user, Table 9)",
        "Used for": "Independent cross-check of provincial production",
        "Resolution": "Province, per season",
        "Access": "Local workbook",
        "Citation": "DAFF (2018), Abstract of Agricultural Statistics",
    },
    {
        "Dataset": "Esri World Imagery; AWS terrain tiles",
        "Publisher": "Esri; Mapzen/AWS Open Data",
        "Used for": "Satellite basemap and 3D elevation on the map",
        "Resolution": "Sub-metre imagery; ~30 m elevation",
        "Access": "Tile services, fetched by the browser",
        "Citation": "Esri World Imagery; Terrarium elevation tiles",
    },
]

st.subheader("Data & methods", anchor=False)
st.markdown(
    f"Every figure in this app traces to one of eight public datasets. Nothing is hand-entered: "
    f"`python -m src.pipeline` rebuilds **{len(ctx.df):,} district-seasons** "
    f"({season_label(ctx.seasons[0])}–{season_label(ctx.seasons[-1])}) from source, and refuses to write the "
    "dataset if any validation check fails."
)

if ctx.metadata:
    with st.container(horizontal=True):
        st.metric("District-seasons", f"{len(ctx.df):,}", border=True)
        st.metric("Seasons", f"{ctx.df['season'].nunique()}", border=True)
        st.metric("Daily observations", "≈167,000", border=True,
                  help="56,856 CHIRPS district-days and 111,000 ERA5 district-days behind the features.")
        st.metric("Satellite composites", "568", border=True, help="MODIS NDVI scenes, cropland-masked per district.")
        st.metric("Built", ctx.metadata["built_at"][:10], border=True)

st.markdown("#### Sources, with citations")
st.dataframe(pd.DataFrame(SOURCES), hide_index=True, column_config={
    "Access": st.column_config.TextColumn(width="medium"),
    "Citation": st.column_config.TextColumn(width="medium")})

st.markdown("#### How the raw observations become features")
st.markdown(
    """
| Step | What happens |
|---|---|
| Sample | Rainfall from a 3×3 CHIRPS window (~15 km) per district; NDVI from a 20 km window, masked to WorldCover cropland and MODIS quality flags |
| Phenology | Green-up is detected per district-season (half the seasonal NDVI amplitude), so the flowering window follows the actual planting date rather than the calendar |
| Aggregate | Season windows defined once in `src/pipeline/config.py` — rainfall Oct–Mar, flowering green-up +30 to +65 days, NDVI Dec–Apr |
| Anomalies | Greenness is measured against each district's fixed 2000–2014 norm, which sits inside every training window, so no test-year information leaks into an input |
| Label | Provincial yield (white + yellow maize, production ÷ area) against a per-province linear trend fitted on training seasons only |
| Validate | Duplicate keys, missing district-seasons, and out-of-range values fail the build |
"""
)

st.markdown("#### Cross-checks")
st.markdown(
    """
- **Yields against an independent publication.** HarvestStat provincial production matches the DAFF Abstract
  exactly for 96 province-seasons (median difference 0.0%). The one disagreement, North West 2017/18 at 17%, is
  the Abstract's own preliminary figure against the later final one.
- **Features against known events.** Heat-stress days, dry spells and greenness all peak in the recorded drought
  seasons (2006/07, 2015/16, 2023/24), and the derived green-up dates reproduce the real east-to-west planting
  order: Mpumalanga late December, Free State and North West mid-January.
- **The previous dataset failed these checks**, which is why it was replaced: it marked 2005/06 as a yield
  drought when reported yields that season were normal.
"""
)

if ctx.metadata:
    with st.expander("Field definitions (from the build)", icon=":material/description:"):
        st.table(pd.DataFrame(list(ctx.metadata["definitions"].items()),
                              columns=["Field", "Definition"]).set_index("Field"))

st.markdown("#### Take the data with you")
c1, c2 = st.columns(2)
with c1:
    download_row(ctx.df, "msindex_district_seasons.csv", "Download the full dataset (CSV)")
    st.caption(f"{len(ctx.df):,} rows · {len(FEATURE_COLS)} model features · yields, splits and provenance included.")
with c2:
    st.caption("Rebuild it yourself:")
    st.code("python -m src.pipeline", language="bash")
    st.caption("Incremental: only new seasons, composites and CEC releases are fetched.")
