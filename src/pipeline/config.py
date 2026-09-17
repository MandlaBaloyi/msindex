"""Shared settings for the msIndex data pipeline.

A "season" is identified by its planting year Y and runs from spring of Y into autumn of Y+1
(e.g. season 2015 = the 2015/16 crop, which the Crop Estimates Committee calls "2016").
"""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_CSV = DATA_DIR / "district_seasons.csv"
METADATA_JSON = DATA_DIR / "metadata.json"
ABSTRACT_XLSX = ROOT / "cfafrica-_-storylab-_-ancir-_-media24-_-data-_-agricultural-statistics-_-2018.xlsx"

# (district, province, lat, lon): towns at the centre of each maize-triangle district
DISTRICTS = [
    ("Bothaville", "Free State", -27.38, 26.62),
    ("Kroonstad", "Free State", -27.65, 27.23),
    ("Sasolburg", "Free State", -26.81, 27.83),
    ("Welkom", "Free State", -27.98, 26.73),
    ("Bethal", "Mpumalanga", -26.45, 29.46),
    ("Delmas", "Mpumalanga", -26.15, 28.68),
    ("Middelburg", "Mpumalanga", -25.77, 29.46),
    ("Standerton", "Mpumalanga", -26.95, 29.24),
    ("Klerksdorp", "North West", -26.86, 26.66),
    ("Lichtenburg", "North West", -26.15, 26.16),
    ("Schweizer-Reneke", "North West", -27.18, 25.33),
    ("Ventersdorp", "North West", -26.32, 26.82),
]
PROVINCES = sorted({p for _, p, _, _ in DISTRICTS})

FIRST_SEASON = 2000  # MODIS Terra starts Feb 2000, so 2000/01 is the first complete season

# Season windows as ((month, day), (month, day)). Months Jul-Dec fall in the planting year, Jan-Jun in the next.
RAIN_WINDOW = ((10, 1), (3, 31))        # seasonal rainfall total and water balance
FLOWERING_WINDOW = ((1, 15), (2, 15))   # fallback silking window when green-up cannot be detected
DRY_SPELL_WINDOW = ((12, 1), (3, 31))   # longest run of dry days during crop growth
HEAT_WINDOW = ((12, 1), (3, 31))        # days above the heat-stress threshold
NDVI_WINDOW = ((12, 1), (4, 30))        # canopy development to maturity
GREENUP_WINDOW = ((10, 1), (2, 28))     # where emergence is looked for in the NDVI series

# Phenology: green-up is when cropland NDVI first passes half its seasonal amplitude, and maize
# silks roughly 30-65 days later. Flowering rainfall is measured over that district-season's own window.
GREENUP_AMPLITUDE_FRACTION = 0.5
FLOWERING_DAYS_AFTER_GREENUP = (30, 65)

DRY_DAY_MM = 1.0           # a day with less rain than this counts as dry
HEAT_THRESHOLD_C = 30.0    # maize heat-stress threshold for daily maximum temperature
NDVI_BASELINE_SEASONS = range(2000, 2015)  # fixed climatology for ndvi_anomaly, inside every training window

# Pixel sampling around each town
CHIRPS_RADIUS_PX = 1        # 3x3 CHIRPS cells (~15 km)
MODIS_RADIUS_KM = 10        # MODIS window half-width
CROPLAND_MIN_FRACTION = 0.5 # MODIS pixel counts as cropland if >= this share is WorldCover cropland
NDVI_MIN_VALID_SHARE = 0.3  # composites with fewer valid cropland pixels are dropped

DATA_LATENCY_DAYS = 20  # wait this long after a window closes before treating a season as complete


def window_dates(season: int, window) -> tuple[date, date]:
    (m0, d0), (m1, d1) = window
    start = date(season if m0 >= 7 else season + 1, m0, d0)
    end = date(season if m1 >= 7 else season + 1, m1, d1)
    return start, end


def season_end(season: int) -> date:
    return max(window_dates(season, w)[1] for w in (RAIN_WINDOW, FLOWERING_WINDOW, DRY_SPELL_WINDOW, HEAT_WINDOW, NDVI_WINDOW))


def complete_seasons(today: date | None = None) -> list[int]:
    today = today or date.today()
    last = FIRST_SEASON
    while (today - season_end(last + 1)).days >= DATA_LATENCY_DAYS:
        last += 1
    return list(range(FIRST_SEASON, last + 1))
