"""Turns cached daily/16-day observations and provincial yields into data/district_seasons.csv."""
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from .config import (DISTRICTS, DRY_DAY_MM, DRY_SPELL_WINDOW, FLOWERING_DAYS_AFTER_GREENUP, FLOWERING_WINDOW,
                     GREENUP_AMPLITUDE_FRACTION, GREENUP_WINDOW, HEAT_THRESHOLD_C, HEAT_WINDOW, METADATA_JSON,
                     NDVI_BASELINE_SEASONS, NDVI_WINDOW, OUTPUT_CSV, RAIN_WINDOW, window_dates)

SOURCES = {
    "rainfall": "CHIRPS v2.0 daily, 0.05 deg (Funk et al. 2015), UCSB Climate Hazards Center",
    "temperature": "ERA5 daily maximum 2 m temperature and FAO-56 reference evapotranspiration "
                   "(ECMWF/Copernicus) via Open-Meteo archive API",
    "vegetation": "MODIS MOD13Q1/MYD13Q1 v6.1 250 m 16-day NDVI (NASA LP DAAC) via Microsoft Planetary Computer",
    "cropland_mask": "ESA WorldCover 2021 v200, 10 m (class 40 cropland)",
    "yields": "Crop Estimates Committee (SAGIS) and HarvestStat Africa v1.2; commercial white + yellow maize by province",
    "yield_check": "DAFF Abstract of Agricultural Statistics 2018, Table 9",
}


def _window(series: pd.Series, season: int, window) -> pd.Series:
    start, end = window_dates(season, window)
    return series[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def greenup_date(ndvi: pd.Series, season: int) -> pd.Timestamp | None:
    """Emergence: the first composite where cropland NDVI passes half of the season's amplitude.

    Green-up moves with planting, which differs by several weeks between the eastern and western
    districts, so the flowering window is anchored to it rather than to a fixed calendar date.
    """
    v = _window(ndvi, season, GREENUP_WINDOW).dropna()
    full = _window(ndvi, season, NDVI_WINDOW).dropna()
    if len(v) < 5 or full.empty:
        return None
    smooth = v.rolling(3, center=True, min_periods=1).median()
    low, high = float(smooth.min()), float(full.max())
    if high - low < 0.05:  # no discernible green-up
        return None
    threshold = low + GREENUP_AMPLITUDE_FRACTION * (high - low)
    crossed = smooth[smooth >= threshold]
    return crossed.index[0] if len(crossed) else None


def season_features(rain: pd.Series, tmax: pd.Series, et0: pd.Series, ndvi: pd.Series, season: int) -> dict | None:
    r = _window(rain, season, RAIN_WINDOW)
    expected = (window_dates(season, RAIN_WINDOW)[1] - window_dates(season, RAIN_WINDOW)[0]).days + 1
    t = _window(tmax, season, HEAT_WINDOW).dropna()
    e = _window(et0, season, RAIN_WINDOW).dropna()
    v = _window(ndvi, season, NDVI_WINDOW).dropna()
    if len(r.dropna()) < 0.97 * expected or len(t) < 100 or len(e) < 0.9 * expected or len(v) < 8:
        return None
    dry = _window(rain, season, DRY_SPELL_WINDOW)
    smooth = v.rolling(3, center=True, min_periods=1).median()  # suppress single cloudy composites

    greenup = greenup_date(ndvi, season)
    if greenup is not None:
        lo, hi = FLOWERING_DAYS_AFTER_GREENUP
        flower_start, flower_end = greenup + pd.Timedelta(days=lo), greenup + pd.Timedelta(days=hi)
        flowering_rain = rain[(rain.index >= flower_start) & (rain.index <= flower_end)]
        flowering_et0 = et0[(et0.index >= flower_start) & (et0.index <= flower_end)]
        greenup_doy = int(greenup.dayofyear if greenup.month < 7 else greenup.dayofyear - 365)
    else:
        flowering_rain = _window(rain, season, FLOWERING_WINDOW)
        flowering_et0 = _window(et0, season, FLOWERING_WINDOW)
        greenup_doy = np.nan

    return {
        "rain_mm": round(float(r.sum()), 1),
        "rain_flowering_mm": round(float(flowering_rain.sum()), 1),
        "water_balance_mm": round(float(r.sum() - e.sum()), 1),
        "water_balance_flowering_mm": round(float(flowering_rain.sum() - flowering_et0.sum()), 1),
        "dry_spell_days": _longest_run((dry.fillna(0) < DRY_DAY_MM).to_numpy()),
        "heat_stress_days": int((t >= HEAT_THRESHOLD_C).sum()),
        "greenup_doy": greenup_doy,
        "ndvi_peak": round(float(smooth.max()), 4),
        "ndvi_integral": round(float(v.mean()), 4),
        "n_ndvi_composites": int(len(v)),
    }


RANGE_CHECKS = {           # plausible bounds for the maize triangle; outside these the inputs are wrong
    "rain_mm": (50, 1600),
    "rain_flowering_mm": (0, 700),
    "water_balance_mm": (-1500, 500),
    "dry_spell_days": (0, 120),
    "heat_stress_days": (0, 130),
    "ndvi_peak": (0.1, 1.0),
    "ndvi_integral": (0.05, 1.0),
    "ndvi_anomaly": (-0.6, 0.6),
    "yield_tha": (0.3, 12.0),
}


def validate(df: pd.DataFrame, seasons: list[int]) -> None:
    """Fails the build rather than shipping a dataset that is quietly wrong."""
    problems = []
    if df.duplicated(["district", "season"]).any():
        problems.append("duplicate district-season rows")
    missing = {(d, s) for d, _, _, _ in DISTRICTS for s in seasons} - set(zip(df["district"], df["season"]))
    if missing:
        problems.append(f"{len(missing)} district-seasons missing, e.g. {sorted(missing)[:3]}")
    for column, (lo, hi) in RANGE_CHECKS.items():
        values = df[column].dropna()
        outside = values[(values < lo) | (values > hi)]
        if len(outside):
            problems.append(f"{column}: {len(outside)} values outside [{lo}, {hi}] (worst {outside.abs().max():.1f})")
    if df["yield_tha"].isna().all():
        problems.append("no yields joined: check the provincial yield sources")
    unlabelled = df.loc[df["yield_tha"].isna(), "season"].unique()
    if len(unlabelled) > 2:
        problems.append(f"{len(unlabelled)} seasons without any yield: {sorted(unlabelled)}")
    if problems:
        raise ValueError("dataset validation failed:\n  - " + "\n  - ".join(problems))
    print(f"  validation passed: {len(df)} rows, {df['season'].nunique()} seasons, "
          f"{int(df['yield_tha'].notna().sum())} rows with yields")


def build(seasons: list[int], chirps: pd.DataFrame, daily: pd.DataFrame, ndvi: pd.DataFrame,
          yields: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, province, lat, lon in DISTRICTS:
        rain = chirps[chirps["district"] == name].set_index(pd.to_datetime(chirps.loc[chirps["district"] == name, "date"]))["precip_mm"]
        day = daily[daily["district"] == name].set_index(pd.to_datetime(daily.loc[daily["district"] == name, "date"]))
        nd = ndvi[ndvi["district"] == name].groupby("date")["ndvi"].mean()  # Terra and Aqua can share a date
        nd.index = pd.to_datetime(nd.index)
        for s in seasons:
            feats = season_features(rain.sort_index(), day["tmax_c"].sort_index(), day["et0_mm"].sort_index(),
                                    nd.sort_index(), s)
            if feats is None:
                print(f"  skipping {name} {s}: incomplete observations")
                continue
            rows.append({"district": name, "province": province, "lat": lat, "lon": lon, "season": s, **feats})
    df = pd.DataFrame(rows)

    # Greenness anomaly against a fixed per-district climatology that sits inside every training window
    base = df[df["season"].isin(NDVI_BASELINE_SEASONS)].groupby("district")["ndvi_peak"].mean()
    df["ndvi_anomaly"] = (df["ndvi_peak"] - df["district"].map(base)).round(4)

    y = yields.rename(columns={"source": "yield_source", "status": "yield_status"})
    df = df.merge(y[["province", "season", "yield_tha", "yield_status", "yield_source"]], on=["province", "season"], how="left")
    df["yield_tha"] = df["yield_tha"].round(3)
    df = df.sort_values(["season", "district"]).reset_index(drop=True)
    validate(df, seasons)
    df.to_csv(OUTPUT_CSV, index=False)

    METADATA_JSON.write_text(json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons": [int(df["season"].min()), int(df["season"].max())],
        "rows": len(df),
        "sources": SOURCES,
        "definitions": {
            "season": "planting year Y = the Y/Y+1 summer crop",
            "rain_mm": f"CHIRPS total {RAIN_WINDOW}",
            "rain_flowering_mm": f"CHIRPS total over this district-season's flowering window: green-up "
                                 f"+{FLOWERING_DAYS_AFTER_GREENUP[0]} to +{FLOWERING_DAYS_AFTER_GREENUP[1]} days "
                                 f"(fallback {FLOWERING_WINDOW})",
            "water_balance_mm": f"CHIRPS rainfall minus FAO-56 reference evapotranspiration, {RAIN_WINDOW}",
            "water_balance_flowering_mm": "same, over the flowering window",
            "greenup_doy": f"day of year when cropland NDVI first passes {GREENUP_AMPLITUDE_FRACTION:.0%} of the "
                           f"season's amplitude, searched over {GREENUP_WINDOW} (negative = before 1 January)",
            "dry_spell_days": f"longest run of days < {DRY_DAY_MM} mm within {DRY_SPELL_WINDOW}",
            "heat_stress_days": f"days with Tmax >= {HEAT_THRESHOLD_C} C within {HEAT_WINDOW}",
            "ndvi_peak": f"max of 3-composite running median of cropland NDVI within {NDVI_WINDOW}",
            "ndvi_integral": f"mean cropland NDVI within {NDVI_WINDOW}",
            "ndvi_anomaly": f"ndvi_peak minus district mean over seasons {NDVI_BASELINE_SEASONS.start}-{NDVI_BASELINE_SEASONS.stop - 1}",
            "yield_tha": "provincial commercial maize (white + yellow) production / area planted",
        },
        "latest_yield": {
            "season": int(df.loc[df["yield_tha"].notna(), "season"].max()),
            "status": df.loc[df["yield_tha"].notna()].sort_values("season")["yield_status"].iloc[-1],
        },
    }, indent=2))
    return df
