from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.model import SPLIT_CURRENT, SPLIT_TEST, SPLIT_TRAIN, add_yield_anomaly, assign_split, fill_features
from src.pipeline.build import _longest_run, greenup_date, season_features
from src.pipeline.config import complete_seasons, season_end, window_dates
from src.pipeline.yields import _split_numbers


def test_season_windows_cross_the_new_year():
    assert window_dates(2015, ((10, 1), (3, 31))) == (date(2015, 10, 1), date(2016, 3, 31))
    assert window_dates(2015, ((1, 15), (2, 15))) == (date(2016, 1, 15), date(2016, 2, 15))
    assert season_end(2015) == date(2016, 4, 30)


def test_a_season_only_counts_once_its_windows_have_closed():
    assert complete_seasons(date(2026, 9, 17))[-1] == 2025   # 2025/26 finished in April 2026
    assert complete_seasons(date(2026, 4, 1))[-1] == 2024    # 2025/26 not yet complete
    assert complete_seasons(date(2026, 5, 10))[-1] == 2024   # still inside the latency window


def test_longest_dry_run():
    assert _longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)) == 3
    assert _longest_run(np.zeros(5, dtype=bool)) == 0
    assert _longest_run(np.ones(5, dtype=bool)) == 5


# --- CEC PDF number parsing: "845 000 370 000" must not be read as one number ---

def triple_check(v):
    return all(v[k] + v[k + 1] == v[k + 2] for k in range(0, len(v), 3))


def test_cec_row_splits_on_the_totals_constraint():
    tokens = "845 000 370 000 1 215 000 4 816 500 2 460 500 7 277 000".split()
    assert _split_numbers(tokens, 6, triple_check) == [845000, 370000, 1215000, 4816500, 2460500, 7277000]


def test_cec_row_with_small_provinces():
    tokens = "200 4 000 4 200 1 840 38 000 39 840".split()
    assert _split_numbers(tokens, 6, triple_check) == [200, 4000, 4200, 1840, 38000, 39840]


def test_cec_row_that_cannot_be_reconciled_returns_none():
    tokens = "845 000 370 000 999 999".split()
    assert _split_numbers(tokens, 6, triple_check) is None


# --- season features ---

def synthetic_series(season=2015, rain_mm=3.0, tmax=28.0, et0=5.0, peak=0.75):
    days = pd.date_range(f"{season}-09-01", f"{season + 1}-06-30", freq="D")
    rain = pd.Series(rain_mm, index=days)
    tmaxs = pd.Series(tmax, index=days)
    et0s = pd.Series(et0, index=days)
    composites = pd.date_range(f"{season}-10-01", f"{season + 1}-05-15", freq="16D")
    curve = np.concatenate([np.linspace(0.2, peak, len(composites) // 2),
                            np.linspace(peak, 0.25, len(composites) - len(composites) // 2)])
    return rain, tmaxs, et0s, pd.Series(curve, index=composites)


def test_features_are_computed_over_the_right_windows():
    rain, tmax, et0, ndvi = synthetic_series()
    feats = season_features(rain, tmax, et0, ndvi, 2015)
    assert feats["rain_mm"] == pytest.approx(3.0 * 183, abs=0.1)        # 1 Oct - 31 Mar
    assert feats["water_balance_mm"] == pytest.approx((3.0 - 5.0) * 183, abs=0.1)
    assert feats["heat_stress_days"] == 0                               # 28C is below the 30C threshold
    assert feats["ndvi_peak"] == pytest.approx(0.75, abs=0.05)
    assert feats["dry_spell_days"] == 0                                 # every day has 3 mm


def test_incomplete_observations_return_none():
    rain, tmax, et0, ndvi = synthetic_series()
    assert season_features(rain.iloc[:50], tmax, et0, ndvi, 2015) is None
    assert season_features(rain, tmax, et0, ndvi.iloc[:3], 2015) is None


def test_greenup_tracks_a_later_planting():
    _, _, _, early = synthetic_series()
    late = early.copy()
    late.index = late.index + pd.Timedelta(days=30)
    assert greenup_date(late, 2015) > greenup_date(early, 2015)


def test_greenup_is_none_without_a_curve():
    flat = pd.Series(0.4, index=pd.date_range("2015-10-01", "2016-04-30", freq="16D"))
    assert greenup_date(flat, 2015) is None


# --- labels and splits ---

def label_frame():
    rows = []
    for season in range(2000, 2026):
        for district, province in [("A", "P1"), ("B", "P1")]:
            rows.append({"season": season, "district": district, "province": province,
                         "yield_tha": 2.0 + 0.1 * (season - 2000),
                         "yield_status": "final" if season <= 2024 else "preliminary (CEC)",
                         "greenup_doy": 10.0})
    return pd.DataFrame(rows)


def test_trend_removes_the_technology_slope():
    df = add_yield_anomaly(label_frame(), 2018)
    assert df["yield_anomaly"].abs().max() < 1e-6  # a perfectly linear history has no anomaly


def test_trend_is_fitted_on_training_seasons_only():
    df = label_frame()
    df.loc[df["season"] >= 2018, "yield_tha"] *= 0.5  # a collapse after the split
    out = add_yield_anomaly(df, 2018)
    assert out.loc[out["season"] < 2018, "yield_anomaly"].abs().max() < 1e-6
    assert out.loc[out["season"] >= 2018, "yield_anomaly"].max() < -0.4  # shows up as a loss, not absorbed


def test_splits_label_current_season_as_forecast():
    split = assign_split(label_frame(), 2018)
    frame = label_frame().assign(split=split)
    assert set(frame.loc[frame.season < 2018, "split"]) == {SPLIT_TRAIN}
    assert set(frame.loc[frame.season.between(2018, 2024), "split"]) == {SPLIT_TEST}
    assert set(frame.loc[frame.season == 2025, "split"]) == {SPLIT_CURRENT}


def test_missing_greenup_is_filled_per_district():
    df = label_frame()
    df.loc[df["district"] == "A", "greenup_doy"] = [np.nan] + [20.0] * (len(df[df.district == "A"]) - 1)
    filled = fill_features(df)
    assert filled["greenup_doy"].isna().sum() == 0
    assert filled.loc[df["district"] == "A", "greenup_doy"].iloc[0] == 20.0
