"""Daily rainfall (CHIRPS v2.0, 0.05 deg) and maximum temperature (ERA5-Land via Open-Meteo) per district."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.windows import from_bounds

from .config import CHIRPS_RADIUS_PX, DISTRICTS, RAIN_WINDOW, HEAT_WINDOW, RAW_DIR, window_dates

CHIRPS_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/cogs/p05/{d.year}/chirps-v2.0.{d.year}.{d.month:02d}.{d.day:02d}.cog"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
CHIRPS_CACHE = RAW_DIR / "chirps_daily.csv"
TMAX_CACHE = RAW_DIR / "era5land_tmax_daily.csv"

_lats = [lat for _, _, lat, _ in DISTRICTS]
_lons = [lon for _, _, _, lon in DISTRICTS]
BBOX = (min(_lons) - 0.3, min(_lats) - 0.3, max(_lons) + 0.3, max(_lats) + 0.3)
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".cog,.tif")


def _days(seasons, window):
    for s in seasons:
        start, end = window_dates(s, window)
        yield from (start + timedelta(n) for n in range((end - start).days + 1))


def _read_chirps_day(day: date) -> list[tuple]:
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(CHIRPS_URL.format(d=day)) as src:
            win = from_bounds(*BBOX, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win).astype("float32")
            arr[arr < 0] = np.nan
            wt = src.window_transform(win)
    rows = []
    for name, _, lat, lon in DISTRICTS:
        r, c = rasterio.transform.rowcol(wt, lon, lat)
        k = CHIRPS_RADIUS_PX
        rows.append((day.isoformat(), name, float(np.nanmean(arr[r - k:r + k + 1, c - k:c + k + 1]))))
    return rows


def update_chirps(seasons: list[int], workers: int = 12) -> pd.DataFrame:
    cached = pd.read_csv(CHIRPS_CACHE) if CHIRPS_CACHE.exists() else pd.DataFrame(columns=["date", "district", "precip_mm"])
    have = set(cached["date"])
    todo = sorted({d for d in _days(seasons, RAIN_WINDOW) if d.isoformat() not in have})
    print(f"  CHIRPS: {len(have)} days cached, {len(todo)} to download")
    rows, failed = [], []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_read_chirps_day, d): d for d in todo}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                rows.extend(f.result())
            except Exception as e:  # keep going; missing days are retried on the next run
                failed.append((futures[f], e))
            if i % 250 == 0:
                print(f"    {i}/{len(todo)} days")
                _save(cached, rows, CHIRPS_CACHE)
    if failed:
        print(f"  CHIRPS: {len(failed)} days failed (first: {failed[0][0]} {failed[0][1]})")
    return _save(cached, rows, CHIRPS_CACHE)


def _save(cached, rows, path):
    new = pd.DataFrame(rows, columns=cached.columns)
    out = pd.concat([cached, new]).drop_duplicates(["date", "district"]).sort_values(["district", "date"])
    out.to_csv(path, index=False)
    return out


def update_daily(seasons: list[int]) -> pd.DataFrame:
    """Daily maximum temperature and FAO-56 reference evapotranspiration (the crop's evaporative demand)."""
    start = window_dates(min(seasons), RAIN_WINDOW)[0]
    end = window_dates(max(seasons), RAIN_WINDOW)[1]
    columns = ["date", "district", "tmax_c", "et0_mm"]
    cached = pd.read_csv(TMAX_CACHE) if TMAX_CACHE.exists() else pd.DataFrame(columns=columns)
    if "et0_mm" not in cached.columns:
        cached = pd.DataFrame(columns=columns)  # older cache without ET0: refetch
    frames = [cached]
    for name, _, lat, lon in DISTRICTS:
        have = cached.loc[cached["district"] == name, "date"]
        if len(have) and have.min() <= start.isoformat() and have.max() >= end.isoformat():
            continue
        params = dict(latitude=lat, longitude=lon, start_date=start.isoformat(), end_date=end.isoformat(),
                      daily="temperature_2m_max,et0_fao_evapotranspiration", timezone="Africa/Johannesburg")
        for attempt in range(5):
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=180)
            if resp.status_code == 200:
                break
            time.sleep(10 * (attempt + 1))
        resp.raise_for_status()
        daily = resp.json()["daily"]
        frames.append(pd.DataFrame({"date": daily["time"], "district": name, "tmax_c": daily["temperature_2m_max"],
                                    "et0_mm": daily["et0_fao_evapotranspiration"]}))
        print(f"  ERA5: {name} {len(daily['time'])} days")
        time.sleep(1)
    out = pd.concat(frames).drop_duplicates(["date", "district"], keep="last").sort_values(["district", "date"])
    out.to_csv(TMAX_CACHE, index=False)
    return out
