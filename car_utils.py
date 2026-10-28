"""Small helpers shared by the notebook and the app, so both prepare a car in exactly the same way."""

import numpy as np
import pandas as pd

NUM_FEATURES = ["age", "engine_cc", "mileage_km", "mileage_reported"]
CAT_FEATURES = ["brand", "model", "gear", "fuel", "condition"]
FEATURES = NUM_FEATURES + CAT_FEATURES


def fit_fill_stats(train: pd.DataFrame) -> dict:
    """Learn the values used to fill missing data, from the TRAINING set only."""
    reported = train[(train["mileage_reported"] == 1) & train["mileage_km"].notna() & (train["age"] > 0)]
    return {
        # a missing engine size gets the median engine size of the same model (e.g. Aqua -> 1500 cc)
        "engine_by_model": train.groupby("model")["engine_cc"].median().to_dict(),
        "engine_overall": float(train["engine_cc"].median()),
        # missing mileage = age x typical km driven per year
        "km_per_year": float((reported["mileage_km"] / reported["age"]).median()),
        # categories the model has seen; anything else becomes "OTHER"
        "categories": {c: sorted(train[c].unique()) for c in CAT_FEATURES},
    }


def fill_missing(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Impute missing values with the training statistics."""
    df = df[FEATURES].copy()
    for c in CAT_FEATURES:
        df[c] = df[c].where(df[c].isin(stats["categories"][c]), "OTHER")
    df["engine_cc"] = df["engine_cc"].fillna(df["model"].map(stats["engine_by_model"])).fillna(stats["engine_overall"])
    df["mileage_km"] = df["mileage_km"].fillna(df["age"].clip(lower=0.5) * stats["km_per_year"])
    df[NUM_FEATURES] = df[NUM_FEATURES].astype(float)
    return df


def encode(df: pd.DataFrame, stats: dict) -> np.ndarray:
    """Turn the text categories into integer codes (tree models split on these codes; no scaling is needed).
    The code list comes from the training set; anything unseen becomes -1."""
    df = df[FEATURES].copy()
    for c in CAT_FEATURES:
        df[c] = df[c].map({v: i for i, v in enumerate(stats["categories"][c])}).fillna(-1)
    return df.astype(float).values


def make_input_row(brand, model, year, engine_cc, gear, fuel, condition, mileage_km=None, listing_year=2026):
    """Turn what a user types into one row with the same columns the model was trained on."""
    reported = mileage_km is not None and mileage_km > 0
    return pd.DataFrame([{
        "age": max(listing_year - int(year), 0),
        "engine_cc": float(engine_cc),
        "mileage_km": float(mileage_km) if reported else np.nan,
        "mileage_reported": int(reported or condition == "NEW"),
        "brand": brand, "model": model, "gear": gear, "fuel": fuel, "condition": condition,
    }])[FEATURES]


def load_and_clean(path: str = "cars.csv") -> pd.DataFrame:
    """The same cleaning as section 4 of the notebook (used by train.py and drift_check.py)."""
    df = pd.read_csv(path).rename(columns={
        "Brand": "brand", "Model": "model_full", "Engine (cc)": "engine_cc", "Gear": "gear", "Fuel Type": "fuel",
        "Millage(KM)": "mileage_km", "Condition": "condition", "Price": "price", "Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "date"])
    filled_price = ((df["price"] * 100 % 1).abs() > 1e-6) & (df.groupby("price")["price"].transform("size") >= 3)
    df = df[~filled_price]
    df = df[(df["YOM"] >= 1990) & df["price"].between(3, 1500)].copy()
    df.loc[df["engine_cc"] < 600, "engine_cc"] = np.nan
    mode = df.groupby("YOM")["mileage_km"].transform(lambda s: s.mode().iloc[0])
    share = df.groupby("YOM")["mileage_km"].transform(lambda s: (s == s.mode().iloc[0]).mean())
    fake = (df["mileage_km"] == mode) & (share >= 0.30) & (df["condition"] == "USED")
    df["mileage_reported"] = (~fake).astype(int)
    df.loc[fake, "mileage_km"] = np.nan
    df["age"] = (df["date"].dt.year - df["YOM"]).clip(lower=0)
    df["model"] = df["model_full"].str.split().str[0]
    for col, k in [("brand", 20), ("model", 15)]:
        counts = df[col].value_counts()
        df[col] = df[col].where(df[col].map(counts) >= k, "OTHER")
    return df.reset_index(drop=True)
