import os

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from src.app_data import (GREENNESS_SCALE, LEGEND_FILE, OVERLAY_DIR, OVERLAY_URL, RISK_COLORS, RISK_RGB,
                          download_row, get_context, policy_sidebar, season_label)
from src.model import SPLIT_CURRENT, SPLIT_TEST, SPLIT_TRAIN
from src.pricing import apply_policy
from src.trigger import LOSS_THRESHOLD, TEST_START_SEASON

# Free basemap tiles: satellite imagery (Esri) and terrain heights (AWS terrarium encoding)
SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TERRAIN_TILES = "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"
TERRARIUM_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}
COLUMN_RADIUS_M = 6000
SEVERITY_SCALE = 150000  # metres of column height per unit of predicted yield shortfall

ctx = get_context(TEST_START_SEASON, os.path.getmtime("data/district_seasons.csv"))
terms = policy_sidebar()


def build_deck(season_df: pd.DataFrame, season: int, layer: str, terrain_3d: bool,
               overlay_opacity: float, tilt: int) -> pdk.Deck:
    """Satellite basemap (optionally draped over real terrain), this season's greenness image, and a
    3D column per district whose height is the predicted yield shortfall."""
    layers = []
    if terrain_3d:
        # TerrainLayer textures the elevation mesh, so the imagery follows the hills
        layers.append(pdk.Layer("TerrainLayer", elevation_decoder=TERRARIUM_DECODER,
                                elevation_data=TERRAIN_TILES, texture=SATELLITE_TILES, max_zoom=12))
    else:
        layers.append(pdk.Layer("TileLayer", data=SATELLITE_TILES, min_zoom=0, max_zoom=19, tile_size=256))
        if layer == "Vegetation greenness" and ctx.overlay_meta and os.path.exists(f"{OVERLAY_DIR}/greenness_{season}.png"):
            bitmap = pdk.Layer("BitmapLayer", data=None, bounds=ctx.overlay_meta["bounds"], opacity=overlay_opacity)
            # Assigned after construction so pydeck keeps the served URL instead of base64-embedding the file
            bitmap.image = f"{OVERLAY_URL}/greenness_{season}.png"
            layers.append(bitmap)

    cols = season_df.assign(
        elevation=lambda d: (-d["pred_yield_anomaly"]).clip(lower=0) * SEVERITY_SCALE + 3000,
        color=lambda d: d["Risk band"].map(lambda b: RISK_RGB[b] + [210]),
        pred_pct=lambda d: (d["pred_yield_anomaly"] * 100).round(1),
        payout_pct=lambda d: (d["payout_share"] * 100).round(1),
        ndvi_txt=lambda d: d["ndvi_anomaly"].round(3),
        rain_txt=lambda d: d["rain_mm"].round(0),
    )
    layers.append(pdk.Layer("ColumnLayer", data=cols.to_dict("records"), get_position=["lon", "lat"],
                            get_elevation="elevation", get_fill_color="color", radius=COLUMN_RADIUS_M,
                            extruded=True, pickable=True, auto_highlight=True, elevation_scale=1))
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=-26.85, longitude=27.5, zoom=6.0, pitch=tilt, bearing=0),
        map_provider=None, map_style=None,  # the satellite tiles above are the basemap: no Mapbox/Carto key needed
        tooltip={"html": "<b>{district}</b> ({province})<br/>Predicted yield: {pred_pct}%<br/>"
                         "Payout: {payout_pct}% of sum insured<br/>Greenness anomaly: {ndvi_txt}<br/>"
                         "Rain Oct–Mar: {rain_txt} mm<br/>Risk band: {Risk band}",
                 "style": {"backgroundColor": "rgba(30,30,30,0.9)", "color": "white", "fontSize": "12px"}},
    )


def render_season_view(slider_season, animate, map_layer, terrain_3d, overlay_opacity, tilt):
    """Everything that depends on the selected season, run as a fragment so "Play all seasons"
    can advance without rerunning the rest of the page."""
    if animate:
        st.session_state.anim_index = (st.session_state.get("anim_index", ctx.seasons.index(slider_season)) + 1) % len(ctx.seasons)
        season = ctx.seasons[st.session_state.anim_index]
    else:
        st.session_state.anim_index = ctx.seasons.index(slider_season)
        season = slider_season

    season_df = apply_policy(ctx.df[ctx.df["season"] == season], terms)
    prev_df = ctx.df[ctx.df["season"] == season - 1]
    split = season_df["Data split"].iloc[0]

    def delta(cur, prev, fmt):
        return None if prev is None else format(cur - prev, fmt)

    has_prev = not prev_df.empty
    cur_ndvi, cur_rain = season_df["ndvi_anomaly"].mean(), season_df["rain_mm"].mean()
    cur_loss = int((season_df["pred_yield_anomaly"] <= LOSS_THRESHOLD).sum())
    prev_ndvi = prev_df["ndvi_anomaly"].mean() if has_prev else None
    prev_rain = prev_df["rain_mm"].mean() if has_prev else None
    prev_loss = int((prev_df["pred_yield_anomaly"] <= LOSS_THRESHOLD).sum()) if has_prev else None

    if animate:
        st.caption(f":material/play_circle: Playing — {season_label(season)}")

    with st.container(horizontal=True):
        st.metric("Mean greenness anomaly (NDVI)", f"{cur_ndvi:+.3f}", delta(cur_ndvi, prev_ndvi, "+.3f"), border=True,
                  help="Peak cropland NDVI against each district's 2000–2014 norm. Negative = browner than usual.")
        st.metric("Mean Oct–Mar rainfall", f"{cur_rain:,.0f} mm", delta(cur_rain, prev_rain, "+,.0f"), border=True)
        st.metric("Districts with predicted loss", f"{cur_loss} / {len(season_df)}",
                  None if prev_loss is None else f"{cur_loss - prev_loss:+d}", delta_color="inverse", border=True,
                  help=f"Predicted yield at least {abs(LOSS_THRESHOLD):.0%} below the provincial trend.")
        st.metric("Payout this season", f"{season_df['payout_share'].mean():.1%} of cover", border=True,
                  help="Average payout across districts under the policy terms in the sidebar.")

    if split == SPLIT_TEST:
        st.badge("Test season: out-of-sample predictions", icon=":material/check_circle:", color="green")
    elif split == SPLIT_TRAIN:
        st.badge("Training season: in-sample predictions, accuracy is overstated", icon=":material/warning:", color="orange")
    else:
        st.badge("Current season: forecast from a model trained on all final seasons; reported yield is preliminary",
                 icon=":material/schedule:", color="blue")

    col1, col2 = st.columns([3, 2])
    with col1:
        with st.container(border=True):
            st.pydeck_chart(build_deck(season_df, season, map_layer, terrain_3d, overlay_opacity, tilt),
                            height=560, key=f"risk_map_{season}_{map_layer}_{terrain_3d}")
            legend_col, text_col = st.columns([1, 3], vertical_alignment="center")
            if map_layer == "Vegetation greenness" and not terrain_3d and ctx.overlay_meta:
                if os.path.exists(LEGEND_FILE):
                    legend_col.image(LEGEND_FILE, width="stretch")
                    legend_col.caption(f"−{ctx.overlay_meta['anomaly_limit']:.0%}  ·  normal  ·  +{ctx.overlay_meta['anomaly_limit']:.0%}")
                text_col.caption(
                    f"Cropland greenness in {season_label(season)} against the "
                    f"{ctx.overlay_meta['baseline_seasons'][0]}–{ctx.overlay_meta['baseline_seasons'][1]} normal, from "
                    "MODIS imagery: brown = drought-stressed, green = healthier than usual. Only cropland is painted. "
                    "Columns are districts, height = predicted loss severity."
                )
            else:
                text_col.caption("Columns are districts: height = predicted loss severity, colour = risk band. "
                                 "Drag to pan, Ctrl-drag to rotate and tilt, scroll to zoom.")

    with col2:
        with st.container(border=True):
            st.markdown("**District risk summary**")
            table = season_df[["district", "province", "rain_mm", "water_balance_mm", "ndvi_anomaly",
                               "pred_yield_anomaly", "payout_share", "Risk band"]].sort_values("pred_yield_anomaly")
            st.dataframe(
                table, hide_index=True, height=250,
                column_config={
                    "district": "District", "province": "Province",
                    "rain_mm": st.column_config.NumberColumn("Rain (mm)", format="%.0f"),
                    "water_balance_mm": st.column_config.NumberColumn("Water bal. (mm)", format="%.0f"),
                    "ndvi_anomaly": st.column_config.NumberColumn("Greenness", format="%+.3f"),
                    "pred_yield_anomaly": st.column_config.NumberColumn("Pred. yield", format="percent"),
                    "payout_share": st.column_config.NumberColumn("Payout", format="percent"),
                },
            )
            download_row(table, f"msindex_districts_{season}.csv")
        with st.container(border=True):
            st.markdown("**Provincial yield: predicted vs reported** (commercial maize)")
            prov = (season_df.groupby("province")
                    .agg(pred=("pred_yield_tha", "mean"), reported=("yield_tha", "first"),
                         trend=("yield_trend", "first"), status=("yield_status", "first")).reset_index())
            st.dataframe(prov, hide_index=True, column_config={
                "province": "Province",
                "pred": st.column_config.NumberColumn("Predicted t/ha", format="%.2f"),
                "reported": st.column_config.NumberColumn("Reported t/ha", format="%.2f"),
                "trend": st.column_config.NumberColumn("Trend t/ha", format="%.2f"),
                "status": "Yield status"})

    # Regional greenness through time, so the selected season is seen in context
    ndvi_lim = float(ctx.df["ndvi_anomaly"].abs().quantile(0.98))
    trend_df = ctx.df.groupby("season", as_index=False)[["ndvi_anomaly", "yield_anomaly"]].mean()
    trend_fig = px.bar(trend_df, x="season", y="ndvi_anomaly", color="ndvi_anomaly",
                       color_continuous_scale=GREENNESS_SCALE, range_color=[-ndvi_lim, ndvi_lim],
                       labels={"season": "Season (planting year)", "ndvi_anomaly": "Mean greenness anomaly",
                               "yield_anomaly": "Reported yield anomaly"},
                       hover_data={"yield_anomaly": ":+.1%"}, height=260)
    trend_fig.update_traces(marker_line_color=["#111" if s == season else "rgba(0,0,0,0)" for s in trend_df["season"]],
                            marker_line_width=2.5)
    trend_fig.add_vline(x=TEST_START_SEASON - 0.5, line_dash="dash", line_color="grey",
                        annotation_text="Test period →", annotation_position="top right")
    trend_fig.add_vline(x=ctx.current_season - 0.5, line_dash="dot", line_color="grey",
                        annotation_text="Current →", annotation_position="bottom right")
    trend_fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False, xaxis=dict(dtick=2))
    with st.container(border=True):
        st.markdown("**Regional greenness by season** (selected season outlined)")
        st.plotly_chart(trend_fig, key=f"greenness_trend_{season}")


st.subheader("Maize triangle vegetation & production risk", anchor=False)

ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 1], vertical_alignment="bottom")
slider_season = ctrl1.select_slider("Agricultural season", options=ctx.seasons, value=ctx.current_season,
                                    format_func=season_label, key="season", bind="query-params")
map_layer = ctrl2.segmented_control("Map layer", options=["Vegetation greenness", "Predicted yield risk"],
                                    default="Vegetation greenness", required=True, key="layer", bind="query-params")
animate = ctrl3.toggle("Play all seasons",
                       help="Step through every season so droughts are visible as brown spreading across the region.")
with st.expander("Map options", icon=":material/tune:"):
    o1, o2, o3 = st.columns(3)
    terrain_3d = o1.toggle("3D terrain", value=False,
                           help="Drape the satellite imagery over real elevation. The greenness image is only drawn "
                                "on the flat map, so risk-band columns carry the colour in 3D.")
    overlay_opacity = o2.slider("Greenness opacity", 0.0, 1.0, 0.85, step=0.05, disabled=terrain_3d)
    tilt = o3.slider("Camera tilt", 0, 60, 40, step=5, help="0 = straight down.")

# Auto-advancing rerun of just this section while "Play all seasons" is on
st.fragment(run_every=2.0 if animate else None)(render_season_view)(
    slider_season, animate, map_layer, terrain_3d, overlay_opacity, tilt)

if ctx.metadata:
    with st.expander("Data sources & definitions", icon=":material/database:"):
        st.caption(f"Dataset built {ctx.metadata['built_at']} · seasons {season_label(ctx.metadata['seasons'][0])}–"
                   f"{season_label(ctx.metadata['seasons'][1])} · refresh with `python -m src.pipeline`")
        st.table(pd.DataFrame([(k.replace("_", " ").capitalize(), v) for k, v in ctx.metadata["sources"].items()],
                              columns=["Input", "Source"]).set_index("Input"))
        st.table(pd.DataFrame(list(ctx.metadata["definitions"].items()),
                              columns=["Field", "Definition"]).set_index("Field"))
