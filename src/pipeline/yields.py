"""Provincial commercial maize area, production and yield (white + yellow combined).

Sources, in order of precedence:
1. Latest Crop Estimates Committee (CEC) release on SAGIS: current season (preliminary forecast)
   and the previous season (final crop).
2. HarvestStat Africa v1.2 (Lee et al. 2025, Scientific Data), which harmonises CEC provincial
   statistics from FEWS NET for 1986-2024 plantings.
The DAFF Abstract of Agricultural Statistics 2018 (Table 9) is used as an independent check of production.
"""
import io
import re
import tempfile
from pathlib import Path

import pandas as pd
import requests

from .config import ABSTRACT_XLSX, PROVINCES, RAW_DIR

HARVESTSTAT_URL = "https://raw.githubusercontent.com/HarvestStat/HarvestStat-Africa/main/public/hvstat_africa_data_v1.2.csv"
SAGIS_CEC_PAGE = "https://www.sagis.org.za/crop-estimates-committee-2/"
CACHE = RAW_DIR / "provincial_yields.csv"
CEC_DIR = RAW_DIR / "cec"


def fetch_harveststat() -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(requests.get(HARVESTSTAT_URL, timeout=300).text), low_memory=False)
    maize = raw[
        (raw["country_code"] == "ZA")
        & raw["product"].isin(["Maize", "Maize (Yellow)"])  # "Maize" is white maize for South Africa
        & (raw["crop_production_system"] == "Commercial (PS)")
        & raw["admin_1"].isin(PROVINCES)
    ]
    out = (
        maize.groupby(["admin_1", "planting_year"])
        .agg(area_ha=("area", "sum"), production_t=("production", "sum"), n=("product", "nunique"))
        .reset_index()
    )
    out = out[out["n"] == 2].drop(columns="n")  # require both white and yellow
    out = out.rename(columns={"admin_1": "province", "planting_year": "season"})
    out["status"] = "final"
    out["source"] = "HarvestStat Africa v1.2 (CEC via FEWS NET)"
    return out


def _split_numbers(tokens: list[str], n_numbers: int, check) -> list[int] | None:
    """Split space-separated digit groups ("845 000 370 000 ...") into n_numbers integers.

    CEC tables use spaces as thousands separators, so the split is ambiguous; `check` validates a
    candidate (white + yellow == total) and backtracking finds the consistent reading.
    """
    def rec(i, acc):
        if len(acc) == n_numbers:
            return acc if i == len(tokens) and check(acc) else None
        if i >= len(tokens) or len(tokens[i]) > 3:
            return None
        for j in range(i + 1, len(tokens) + 1):
            if j > i + 1 and len(tokens[j - 1]) != 3:
                break
            if tokens[i] == "0" and j > i + 1:
                break
            res = rec(j, acc + [int("".join(tokens[i:j]))])
            if res:
                return res
        return None
    return rec(0, [])


def parse_cec_pdf(path: Path, released: str) -> pd.DataFrame:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        page = next((p.extract_text() or "" for p in pdf.pages
                     if "WHITE AND YELLOW MAIZE" in (p.extract_text() or "").upper()), None)
    if page is None:
        return pd.DataFrame()

    title = next(line for line in page.splitlines() if "WHITE AND YELLOW MAIZE" in line.upper())
    years = re.findall(r"\b(20\d{2})\b", page.split("Western Cape")[0])
    cur_year, prev_year = int(years[-4]), int(years[-2])  # header row reads "2026 2026 2025 2025"
    stage = re.search(r"AREA PLANTED AND (.+?) FORECAST", title.upper())
    cur_status = f"preliminary (CEC {stage.group(1).lower()} forecast, {released})" if stage else f"CEC estimate, {released}"

    def triple_ok(v):
        return all(v[k] + v[k + 1] == v[k + 2] for k in range(0, 12, 3))

    rows = []
    for province in PROVINCES:
        line = next((ln for ln in page.splitlines() if ln.startswith(province)), None)
        if line is None:
            continue
        tokens = re.findall(r"\d+", line[len(province):].split("/", 1)[-1] if "/" in line else line[len(province):])
        values = _split_numbers(tokens, 12, triple_ok)
        if values is None:
            raise ValueError(f"Could not parse CEC row for {province}: {line}")
        rows.append((province, cur_year - 1, values[2], values[5], cur_status))
        rows.append((province, prev_year - 1, values[8], values[11], f"final (CEC, {released})"))

    out = pd.DataFrame(rows, columns=["province", "season", "area_ha", "production_t", "status"])
    out["source"] = f"Crop Estimates Committee release {released} ({path.name})"
    return out


def fetch_latest_cec(max_tries: int = 6) -> pd.DataFrame:
    html = requests.get(SAGIS_CEC_PAGE, timeout=60).text
    links = set(re.findall(r'href="([^"]*CEC_(\d{4})-(\d{2})(?:-(\d{2}))?\.pdf)"', html))
    links = sorted(links, key=lambda t: (t[1], t[2], t[3] or "00"), reverse=True)
    CEC_DIR.mkdir(parents=True, exist_ok=True)
    for url, y, m, d in links[:max_tries]:
        released = f"{y}-{m}-{d}" if d else f"{y}-{m}"
        path = CEC_DIR / url.rsplit("/", 1)[-1]
        if not path.exists():
            path.write_bytes(requests.get(url, timeout=120).content)
        df = parse_cec_pdf(path, released)
        if len(df):
            return df
    return pd.DataFrame()


def validate_against_abstract(yields: pd.DataFrame) -> pd.DataFrame | None:
    """Compares production with DAFF Abstract 2018 Table 9 (1000 t, per province)."""
    if not ABSTRACT_XLSX.exists():
        return None
    sheet = pd.read_excel(ABSTRACT_XLSX, sheet_name="Field Crops (7-33)", header=None)
    start = sheet.index[sheet[1].astype(str).str.startswith("Table 9")][0]
    block = sheet.iloc[start:start + 90]
    header_row = block.index[block[2].astype(str).str.contains("Western", na=False)][0]
    cols = {"Free State": 5, "Mpumalanga": 8, "North West": 10}
    rows = []
    for _, r in block.loc[header_row:].iterrows():
        m = re.match(r"^(\d{4})/\d{2}", str(r[1]))
        if m:
            for prov, c in cols.items():
                rows.append((prov, int(m.group(1)), float(r[c]) * 1000))
    ab = pd.DataFrame(rows, columns=["province", "season", "abstract_production_t"])
    merged = yields.merge(ab, on=["province", "season"])
    merged["diff_pct"] = (merged["production_t"] / merged["abstract_production_t"] - 1) * 100
    return merged


def build(refresh: bool = True) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE)
    hs = fetch_harveststat()
    cec = fetch_latest_cec()
    combined = pd.concat([cec, hs]).drop_duplicates(["province", "season"], keep="first")
    combined["yield_tha"] = combined["production_t"] / combined["area_ha"]
    combined = combined.sort_values(["province", "season"]).reset_index(drop=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(CACHE, index=False)

    check = validate_against_abstract(combined)
    if check is not None and len(check):
        worst = check.loc[check["diff_pct"].abs().idxmax()]
        print(f"  Abstract 2018 cross-check: {len(check)} province-seasons, median |diff| "
              f"{check['diff_pct'].abs().median():.1f}%, worst {worst['province']} {int(worst['season'])} {worst['diff_pct']:+.1f}%")
    return combined
