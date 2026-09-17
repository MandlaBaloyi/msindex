"""Refresh all msIndex data: `python -m src.pipeline`.

Downloads are cached in data/raw/, so re-running only fetches seasons and composites that are new.
"""
import argparse
import time

from . import build, modis, overlays, weather, yields
from .config import OUTPUT_CSV, complete_seasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="planting-year range (default: every complete season since 2000)")
    args = parser.parse_args()
    seasons = list(range(args.seasons[0], args.seasons[1] + 1)) if args.seasons else complete_seasons()
    print(f"Seasons {seasons[0]}/{seasons[0] + 1} to {seasons[-1]}/{seasons[-1] + 1}")

    t0 = time.time()
    print("[1/6] Provincial yields")
    y = yields.build()
    print("[2/6] ERA5 temperature & evapotranspiration")
    daily = weather.update_daily(seasons)
    print("[3/6] CHIRPS rainfall")
    rain = weather.update_chirps(seasons)
    print("[4/6] MODIS NDVI")
    ndvi = modis.update_ndvi(seasons)
    print("[5/6] Building district-season table")
    df = build.build(seasons, rain, daily, ndvi, y)
    print("[6/6] Map overlays")
    overlays.build(seasons)
    print(f"Wrote {len(df)} rows ({df['season'].min()}-{df['season'].max()}) to {OUTPUT_CSV} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
