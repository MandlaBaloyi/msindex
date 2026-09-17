"""Cropland-masked MODIS NDVI (MOD13Q1 + MYD13Q1 v6.1, 250 m, 16-day) per district.

Imagery is read from Microsoft Planetary Computer's cloud-optimised GeoTIFFs (anonymous SAS tokens).
Only pixels that ESA WorldCover 2021 (10 m) classifies as mostly cropland, and that MODIS flags as
good or marginal quality, contribute to each district's NDVI.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import Resampling, reproject, transform as warp_transform
from rasterio.windows import Window, from_bounds

from .config import (CROPLAND_MIN_FRACTION, DISTRICTS, MODIS_RADIUS_KM, NDVI_MIN_VALID_SHARE,
                     NDVI_WINDOW, RAW_DIR, window_dates)
from .weather import BBOX, GDAL_ENV

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
NDVI_CACHE = RAW_DIR / "modis_ndvi.csv"
MASK_CACHE = RAW_DIR / "cropland_masks.npz"
TILE = "h20v11"
WORLDCOVER_CROPLAND = 40

_tokens: dict[str, str] = {}


def _signed(href: str) -> str:
    u = urlparse(href)
    key = f"{u.netloc.split('.')[0]}/{u.path.split('/')[1]}"
    if key not in _tokens:
        _tokens[key] = requests.get(f"{SAS}/{key}", timeout=60).json()["token"]
    return f"{href}?{_tokens[key]}"


def _search(collection: str, datetime_range: str | None = None) -> list[dict]:
    body = {"collections": [collection], "bbox": list(BBOX), "limit": 500}
    if datetime_range:
        body["datetime"] = datetime_range
    items, url = [], f"{STAC}/search"
    while url:
        page = requests.post(url, json=body, timeout=120).json()
        items += page["features"]
        nxt = next((l for l in page.get("links", []) if l["rel"] == "next"), None)
        url, body = (nxt["href"], nxt.get("body", body)) if nxt else (None, None)
    return items


def _windows(href: str) -> dict[str, tuple[Window, rasterio.Affine]]:
    with rasterio.Env(**GDAL_ENV), rasterio.open(_signed(href)) as src:
        px = int(round(MODIS_RADIUS_KM * 1000 / src.res[0]))
        out = {}
        for name, _, lat, lon in DISTRICTS:
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            r, c = src.index(xs[0], ys[0])
            if not (px <= r < src.height - px and px <= c < src.width - px):
                raise ValueError(f"{name} is too close to the edge of MODIS tile {TILE}")
            win = Window(c - px, r - px, 2 * px + 1, 2 * px + 1)
            out[name] = (win, src.window_transform(win))
        return out, src.crs


def build_masks(template_href: str) -> dict[str, np.ndarray]:
    if MASK_CACHE.exists():
        return dict(np.load(MASK_CACHE))
    windows, modis_crs = _windows(template_href)
    wc_items = [i for i in _search("esa-worldcover") if "2021" in i["id"]]
    masks = {}
    for name, _, lat, lon in DISTRICTS:
        win, wt = windows[name]
        frac = np.zeros((int(win.height), int(win.width)), dtype="float32")
        weight = np.zeros_like(frac)
        pad = MODIS_RADIUS_KM / 100 + 0.05
        for item in wc_items:
            w, s, e, n = item["bbox"]
            if not (w <= lon + pad and e >= lon - pad and s <= lat + pad and n >= lat - pad):
                continue
            with rasterio.Env(**GDAL_ENV), rasterio.open(_signed(item["assets"]["map"]["href"])) as src:
                rwin = from_bounds(lon - pad, lat - pad, lon + pad, lat + pad, src.transform).intersection(
                    Window(0, 0, src.width, src.height)).round_offsets().round_lengths()
                # 40 m (4x decimated, served from overviews) is ample for 231 m MODIS pixels
                shape = (int(rwin.height) // 4, int(rwin.width) // 4)
                cls = src.read(1, window=rwin, out_shape=shape, resampling=Resampling.nearest)
                read_transform = src.window_transform(rwin) * rasterio.Affine.scale(rwin.width / shape[1], rwin.height / shape[0])
                crop = (cls == WORLDCOVER_CROPLAND).astype("float32")
                valid = (cls > 0).astype("float32")
                for arr, dst in ((crop * valid, frac), (valid, weight)):
                    tmp = np.zeros_like(frac)
                    reproject(arr, tmp, src_transform=read_transform, src_crs=src.crs,
                              dst_transform=wt, dst_crs=modis_crs, resampling=Resampling.average)
                    dst += tmp
        share = np.divide(frac, weight, out=np.zeros_like(frac), where=weight > 0)
        mask = share >= CROPLAND_MIN_FRACTION
        if mask.sum() < 50:  # sparse cropland: fall back to the most cropland-dominated quarter
            mask = share >= np.quantile(share, 0.75)
        masks[name] = mask
        print(f"  cropland mask {name}: {mask.mean():.0%} of {mask.size} MODIS pixels")
    np.savez_compressed(MASK_CACHE, **masks)
    return masks


def _read_item(item: dict, windows: dict, masks: dict) -> list[tuple]:
    rows = []
    start = datetime.fromisoformat(item["properties"]["start_datetime"].replace("Z", "+00:00"))
    composite_date = (start + timedelta(days=8)).date().isoformat()
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(_signed(item["assets"]["250m_16_days_NDVI"]["href"])) as ndvi_src, \
             rasterio.open(_signed(item["assets"]["250m_16_days_pixel_reliability"]["href"])) as rel_src:
            for name, (win, _) in windows.items():
                ndvi = ndvi_src.read(1, window=win).astype("float32") / 10000.0
                rel = rel_src.read(1, window=win)
                mask = masks[name]
                good = mask & np.isin(rel, (0, 1)) & (ndvi > -0.2)
                share = good.sum() / max(mask.sum(), 1)
                value = float(ndvi[good].mean()) if share >= NDVI_MIN_VALID_SHARE else np.nan
                rows.append((item["id"], composite_date, name, value, round(float(share), 3)))
    return rows


def update_ndvi(seasons: list[int], workers: int = 8) -> pd.DataFrame:
    cols = ["item_id", "date", "district", "ndvi", "valid_share"]
    cached = pd.read_csv(NDVI_CACHE) if NDVI_CACHE.exists() else pd.DataFrame(columns=cols)
    have = set(cached["item_id"])
    items = []
    for s in seasons:
        start, end = window_dates(s, NDVI_WINDOW)
        # Composites starting up to 15 days before the window still overlap it
        rng = f"{(start - timedelta(days=15)).isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z"
        items += [i for i in _search("modis-13Q1-061", rng) if f".{TILE}." in i["id"]]
    items = list({i["id"]: i for i in items}.values())
    todo = [i for i in items if i["id"] not in have]
    print(f"  MODIS: {len(items)} composites in season windows, {len(todo)} to download")
    if not todo:
        return cached

    windows, _ = _windows(todo[0]["assets"]["250m_16_days_NDVI"]["href"])
    masks = build_masks(todo[0]["assets"]["250m_16_days_NDVI"]["href"])
    rows, failed = [], []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_read_item, it, windows, masks): it["id"] for it in todo}
        for n, f in enumerate(as_completed(futures), 1):
            try:
                rows.extend(f.result())
            except Exception as e:
                failed.append((futures[f], e))
            if n % 50 == 0:
                print(f"    {n}/{len(todo)} composites")
                _save(cached, rows, cols)
    if failed:
        print(f"  MODIS: {len(failed)} composites failed (first: {failed[0][0]} {failed[0][1]})")
    return _save(cached, rows, cols)


def _save(cached, rows, cols):
    out = pd.concat([cached, pd.DataFrame(rows, columns=cols)])
    out = out.drop_duplicates(["item_id", "district"]).sort_values(["district", "date"])
    out.to_csv(NDVI_CACHE, index=False)
    return out
