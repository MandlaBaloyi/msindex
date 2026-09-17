"""Renders one greenness-anomaly image per season for the whole maize triangle.

For each season the per-pixel seasonal maximum NDVI is composited from every MODIS composite in the
NDVI window (max-value compositing suppresses cloud contamination), reprojected to WGS84, and compared
with the 2000-2014 per-pixel mean. The result is written as a transparent PNG that the app drapes over
satellite imagery, so drought reads as brown spreading across the landscape rather than a dot changing colour.

Only cropland (ESA WorldCover) is painted; everything else stays transparent so the basemap shows through.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from .config import DATA_DIR, NDVI_BASELINE_SEASONS, NDVI_WINDOW, RAW_DIR, window_dates
from .modis import TILE, _search, _signed
from .weather import GDAL_ENV

OVERLAY_DIR = DATA_DIR.parent / "static" / "overlays"  # served by Streamlit at /app/static/overlays/
SEASON_CACHE = RAW_DIR / "overlay_seasons"
# Region drawn: the maize triangle plus a margin, and the output pixel size in degrees
BOUNDS = (24.9, -28.45, 30.05, -25.35)  # west, south, east, north
PIXEL_DEG = 0.004  # ~440 m
ANOMALY_LIMIT = 0.25  # colour scale saturates here, matching the app's legend
# BrBG-style ramp: brown (below normal) through pale (normal) to green (above normal)
RAMP = np.array([
    (84, 48, 5), (140, 81, 10), (191, 129, 45), (223, 194, 125), (246, 232, 195),
    (245, 245, 245),
    (199, 234, 229), (128, 205, 193), (53, 151, 143), (1, 102, 94), (0, 60, 48),
], dtype="float32")


def _grid():
    west, south, east, north = BOUNDS
    width = int(round((east - west) / PIXEL_DEG))
    height = int(round((north - south) / PIXEL_DEG))
    return from_origin(west, north, PIXEL_DEG, PIXEL_DEG), width, height


def _season_max(season: int, workers: int = 6) -> np.ndarray:
    """Per-pixel maximum NDVI over the season, reprojected to the output grid."""
    start, end = window_dates(season, NDVI_WINDOW)
    rng = f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z"
    items = [i for i in _search("modis-13Q1-061", rng) if f".{TILE}." in i["id"]]
    transform, width, height = _grid()
    acc = None

    def read(item):
        with rasterio.Env(**GDAL_ENV), rasterio.open(_signed(item["assets"]["250m_16_days_NDVI"]["href"])) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, *BOUNDS)
            win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win).astype("float32") / 10000.0
            return arr, src.window_transform(win), src.crs

    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(read, it) for it in items]):
            arr, win_transform, crs = future.result()
            arr[arr < -0.2] = np.nan
            acc = arr if acc is None else np.fmax(acc, arr)
            src_transform, src_crs = win_transform, crs

    out = np.full((height, width), np.nan, dtype="float32")
    reproject(acc, out, src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
              dst_transform=transform, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=Resampling.average)
    print(f"  season {season}: {len(items)} composites -> {np.isfinite(out).mean():.0%} of grid covered")
    return out


def _cropland_alpha() -> np.ndarray:
    """Cropland share on the output grid, read from WorldCover at ~440 m."""
    path = SEASON_CACHE / "cropland_alpha.npy"
    if path.exists():
        return np.load(path)
    transform, width, height = _grid()
    frac = np.zeros((height, width), dtype="float32")
    for item in [i for i in _search("esa-worldcover") if "2021" in i["id"]]:
        with rasterio.Env(**GDAL_ENV), rasterio.open(_signed(item["assets"]["map"]["href"])) as src:
            win = from_bounds(*BOUNDS, transform=src.transform)
            win = win.intersection(Window(0, 0, src.width, src.height)).round_offsets().round_lengths()
            if win.width < 2 or win.height < 2:
                continue
            shape = (max(int(win.height) // 16, 1), max(int(win.width) // 16, 1))  # ~160 m
            cls = src.read(1, window=win, out_shape=shape, resampling=Resampling.nearest)
            read_transform = src.window_transform(win) * rasterio.Affine.scale(win.width / shape[1], win.height / shape[0])
            tmp = np.zeros_like(frac)
            reproject((cls == 40).astype("float32"), tmp, src_transform=read_transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.average)
            frac = np.fmax(frac, tmp)
    SEASON_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, frac)
    return frac


def _colorise(anomaly: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    valid = np.isfinite(anomaly)
    pos = np.clip((np.where(valid, anomaly, 0.0) + ANOMALY_LIMIT) / (2 * ANOMALY_LIMIT), 0, 1) * (len(RAMP) - 1)
    lo = np.clip(np.floor(pos), 0, len(RAMP) - 2).astype(int)
    w = (pos - lo)[..., None]
    rgb = RAMP[lo] * (1 - w) + RAMP[lo + 1] * w
    a = np.clip(alpha, 0, 1) * 255
    a[~valid] = 0  # nothing is painted where the season has no usable observation
    return np.dstack([rgb, a]).astype("uint8")


def _smooth(a: np.ndarray, radius: int = 1) -> np.ndarray:
    """NaN-aware box mean: knits field-level speckle into readable patches."""
    filled = np.nan_to_num(a, nan=0.0)
    weight = np.isfinite(a).astype("float32")
    total = np.zeros_like(filled)
    count = np.zeros_like(weight)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            total += np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
            count += np.roll(np.roll(weight, dy, axis=0), dx, axis=1)
    out = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return np.where(np.isfinite(a) | (count > 2), out, np.nan)


def _write_png(rgba: np.ndarray, path, colors: int = 96):
    from PIL import Image
    img = Image.fromarray(rgba, mode="RGBA")
    # A 96-colour palette keeps the ramp smooth while cutting each file to a fraction of the RGBA size
    img.quantize(colors=colors, method=Image.Quantize.FASTOCTREE).save(path, optimize=True)


def build(seasons: list[int], refresh: bool = False) -> dict:
    SEASON_CACHE.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for s in seasons:
        cache = SEASON_CACHE / f"ndvi_max_{s}.npy"
        if cache.exists() and not refresh:
            arrays[s] = np.load(cache)
            continue
        arrays[s] = _season_max(s)
        np.save(cache, arrays[s].astype("float32"))

    baseline_seasons = [s for s in seasons if s in NDVI_BASELINE_SEASONS]
    baseline = np.nanmean(np.dstack([arrays[s] for s in baseline_seasons]), axis=2)
    alpha = _smooth(_cropland_alpha(), radius=2)
    alpha = np.where(alpha >= 0.12, np.clip(alpha * 2.0, 0, 0.9), 0.0)

    for s in seasons:
        _write_png(_colorise(_smooth(arrays[s] - baseline), alpha), OVERLAY_DIR / f"greenness_{s}.png")

    strip = np.linspace(-ANOMALY_LIMIT, ANOMALY_LIMIT, 256)[None, :].repeat(18, axis=0)
    _write_png(_colorise(strip, np.ones_like(strip)), OVERLAY_DIR / "legend.png", colors=64)

    meta = {
        "bounds": list(BOUNDS),
        "pixel_deg": PIXEL_DEG,
        "anomaly_limit": ANOMALY_LIMIT,
        "baseline_seasons": [min(baseline_seasons), max(baseline_seasons)],
        "seasons": seasons,
        "definition": "per-pixel seasonal maximum NDVI minus the per-pixel 2000-2014 mean, cropland only",
    }
    (OVERLAY_DIR / "overlays.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {len(seasons)} overlays to {OVERLAY_DIR}")
    return meta
