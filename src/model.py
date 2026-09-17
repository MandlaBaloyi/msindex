"""Yield-anomaly labels, data splits and the msIndex regressor."""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "rain_mm", "rain_flowering_mm", "water_balance_mm", "water_balance_flowering_mm",
    "dry_spell_days", "heat_stress_days", "greenup_doy",
    "ndvi_peak", "ndvi_integral", "ndvi_anomaly",
]
SPLIT_TRAIN = "Training (in-sample)"
SPLIT_TEST = "Test (out-of-sample)"
SPLIT_CURRENT = "Current season (forecast)"


def is_final(df: pd.DataFrame) -> pd.Series:
    return df["yield_tha"].notna() & df["yield_status"].astype(str).str.startswith("final")


def add_yield_anomaly(df: pd.DataFrame, test_start: int) -> pd.DataFrame:
    """Yield anomaly = reported yield vs a per-province linear technology trend.

    The trend is fitted on training seasons only, so test and current seasons carry no look-ahead.
    """
    df = df.copy()
    prov = df[is_final(df) & (df["season"] < test_start)].drop_duplicates(["province", "season"])
    coefs = {p: np.polyfit(g["season"], g["yield_tha"], 1) for p, g in prov.groupby("province")}
    df["yield_trend"] = [np.polyval(coefs[p], s) for p, s in zip(df["province"], df["season"])]
    df["yield_anomaly"] = (df["yield_tha"] / df["yield_trend"] - 1).round(4)
    return df


def assign_split(df: pd.DataFrame, test_start: int) -> pd.Series:
    final = is_final(df)
    return pd.Series(
        np.select([final & (df["season"] < test_start), final], [SPLIT_TRAIN, SPLIT_TEST], SPLIT_CURRENT),
        index=df.index,
    )


def make_model():
    # With ~200 training rows (labels shared within a province), a regularised linear model beat gradient
    # boosting and random forests in leave-one-season-out CV on 2000-2017 (RMSE 0.137 vs 0.151-0.155).
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


def fill_features(df: pd.DataFrame) -> pd.DataFrame:
    """Green-up is occasionally undetectable (persistent cloud); fall back to the district's usual date."""
    df = df.copy()
    by_district = df.groupby("district")["greenup_doy"].transform("median")
    df["greenup_doy"] = df["greenup_doy"].fillna(by_district).fillna(df["greenup_doy"].median())
    return df


def score(df: pd.DataFrame, test_start: int) -> pd.DataFrame:
    """Adds pred_yield_anomaly.

    Training and test seasons are scored by a model fitted on training seasons only (honest evaluation).
    The current season is scored by a model refitted on every season with a final yield.
    """
    df = add_yield_anomaly(fill_features(df), test_start)
    df["Data split"] = assign_split(df, test_start)
    train = df[df["Data split"] == SPLIT_TRAIN]
    labelled = df[df["Data split"] != SPLIT_CURRENT]

    eval_model = make_model().fit(train[FEATURE_COLS], train["yield_anomaly"])
    prod_model = make_model().fit(labelled[FEATURE_COLS], labelled["yield_anomaly"])
    current = df["Data split"] == SPLIT_CURRENT
    pred = np.where(current, prod_model.predict(df[FEATURE_COLS]), eval_model.predict(df[FEATURE_COLS]))

    # A regularised model shrinks its predictions toward the mean, which is right for accuracy but wrong
    # for an index: a payout ladder reading in yield-shortfall units would almost never reach its lower
    # steps. Rescaling to the training spread keeps the ranking and makes the index mean what it says.
    train_pred = eval_model.predict(train[FEATURE_COLS])
    scale = train["yield_anomaly"].std() / train_pred.std() if train_pred.std() > 0 else 1.0
    df["pred_yield_anomaly"] = np.round(train["yield_anomaly"].mean() + (pred - train_pred.mean()) * scale, 4)
    df["index_scale"] = round(float(scale), 3)
    df["pred_yield_tha"] = (df["yield_trend"] * (1 + df["pred_yield_anomaly"])).round(2)
    return df
