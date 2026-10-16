"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

All data wrangling lives here so the rest of the code just
calls these functions. The random seed is fixed everywhere (see SEED).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/bina_az_sale.csv"   # file name inside the Kaggle archive

# Columns that LEAK the target (derived from price). Dropped before training.
#   unit_price  = price / area ("3 450 AZN/m²")  -> price restated per m2
#   total_price = price restated (equal to price in 99.99% of rows)
LEAKAGE_COLUMNS: list[str] = [
    "unit_price",
    "total_price",
]

# Identifiers, personal data and scraping metadata: they do not describe the
# property, or they encode the listing itself, so they are dropped too.
IDENTIFIER_COLUMNS: list[str] = [
    "id_x", "id_y", "estate_id", "estate_details_id_x", "estate_details_id_y",
    "rel_url", "estate_rel_url_y", "estate_rel_url", "img_url",
    "owner_name", "owner_title", "shop_name", "shop_title", "address",
    "datetime_scrape_x", "datetime_scrape_y", "day_x", "hour_x", "day_y",
    "hour_y", "city_when", "updated", "views", "vip", "featured",
    "products_label",
]

# Azerbaijani column name -> English name, for the columns we keep.
# Not kept: city (99.7% Baku), Binanın növü (99.8% missing), the duplicated
# English flags (repair / bill_of_sale / mortgage repeat Təmir / Çıxarış /
# İpoteka) and the free text (description, attributes, extra_info).
RENAME = {
    "price": "price",
    "currency_x": "currency",
    "estate_rel_url_x": "listing_url",
    "location": "location",
    "lat": "lat",
    "lng": "lng",
    "Kateqoriya": "category",
    "Sahə": "area",
    "Otaq sayı": "rooms",
    "Mərtəbə": "floor",
    "Torpaq sahəsi": "land",
    "Təmir": "repair",
    "Çıxarış": "bill_of_sale",
    "İpoteka": "mortgage",
}

# We model residential property only (Kateqoriya values).
CATEGORIES = {
    "Yeni tikili": "new_build",          # flat in a new building
    "Köhnə tikili": "old_build",         # flat in an old (Soviet) building
    "Həyət evi/Bağ evi": "house",        # private house / villa
}

# Outlier rules (fixed domain limits, not statistics of the data).
PRICE_MIN, PRICE_MAX = 10_000, 5_000_000     # AZN
AREA_MIN, AREA_MAX = 15, 1_000               # m2
ROOMS_MAX = 12
LAND_MAX = 100                               # sot (1 sot = 100 m2)
LAT_MIN, LAT_MAX = 38.3, 42.0                # Azerbaijan bounding box
LNG_MIN, LNG_MAX = 44.7, 50.7

N_TOP_LOCATIONS = 20    # one-hot the 20 most common districts, rest = "other"

# Azerbaijani letters -> ASCII, so feature names print on any console.
AZ_TO_ASCII = {
    "ə": "e", "Ə": "E", "ı": "i", "İ": "I", "ö": "o", "Ö": "O",
    "ü": "u", "Ü": "U", "ş": "sh", "Ş": "Sh", "ç": "ch", "Ç": "Ch",
    "ğ": "g", "Ğ": "G",
}


def to_ascii(text: str) -> str:
    """Transliterate an Azerbaijani string to a simple ASCII name."""
    for az, en in AZ_TO_ASCII.items():
        text = text.replace(az, en)
    text = text.replace(".", "").strip().replace(" ", "_")
    return text


def load_raw(path: str = DATA_PATH):
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    return pd.read_csv(path, encoding="utf-8")


def clean(df, verbose: bool = True):
    """
    Clean the raw frame:
      - drop LEAKAGE_COLUMNS and identifiers, translate kept column names,
      - fix dtypes (areas / floors / land arrive as strings with units),
      - handle missing values,
      - drop duplicates,
      - handle outliers (fixed domain rules, see the constants above).
    Return the cleaned DataFrame.
    """
    def log(step, frame):
        if verbose:
            print(f"  {step:<45s} rows = {len(frame)}")

    log("raw dump", df)

    # 1) leakage + identifiers out, keep and translate the useful columns
    df = df.drop(columns=LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS)
    df = df[list(RENAME.keys())].rename(columns=RENAME)

    # 2) currency: every price must be in AZN (all rows are, but we check)
    df = df[df["currency"] == "AZN"].copy()
    log("price in AZN", df)

    # 3) duplicates: the scraper saw the same listing on several pages,
    #    and some flats are re-posted as new listings with identical content
    df = df.drop_duplicates(subset="listing_url")
    log("drop repeated listing URLs", df)
    df = df.drop_duplicates(
        subset=["price", "area", "rooms", "floor", "lat", "lng"])
    log("drop re-posted identical listings", df)

    # 4) residential categories only (land, offices, garages, commercial
    #    objects follow a different pricing logic; land area is in "sot")
    df = df[df["category"].isin(CATEGORIES.keys())].copy()
    df["category"] = df["category"].map(CATEGORIES)
    log("keep flats and houses", df)

    # 5) type coercion: "145 m²" -> 145.0, "1.3 sot" -> 1.3, "7 / 9" -> 7, 9
    df["area"] = pd.to_numeric(df["area"].str.replace("m²", "").str.replace(" ", ""),
                               errors="coerce")
    df["land"] = pd.to_numeric(df["land"].str.replace("sot", "").str.strip(),
                               errors="coerce")
    floors = df["floor"].str.split("/", expand=True)
    df["floor"] = pd.to_numeric(floors[0].str.strip(), errors="coerce")
    df["total_floors"] = pd.to_numeric(floors[1].str.strip(), errors="coerce")
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")

    # 6) missing values
    #    - floor / total_floors do not exist for houses   -> 0 (not applicable)
    #    - land area does not exist for flats              -> 0 (not applicable)
    #    - repair / mortgage / bill of sale: missing means "not stated" -> 0
    #    - rooms missing (a few houses): we drop those rows
    is_house = df["category"] == "house"
    df.loc[is_house, ["floor", "total_floors"]] = 0
    df.loc[~is_house, "land"] = 0
    df["repair"] = (df["repair"] == "var").astype(int)
    df["bill_of_sale"] = (df["bill_of_sale"] == "var").astype(int)
    df["mortgage"] = (df["mortgage"] == "var").astype(int)
    df = df.dropna(subset=["price", "area", "rooms", "floor", "total_floors",
                           "land", "lat", "lng"])
    log("drop rows with missing numeric values", df)

    # 7) outliers: fixed, documented domain limits
    keep = (
        df["price"].between(PRICE_MIN, PRICE_MAX)
        & df["area"].between(AREA_MIN, AREA_MAX)
        & df["rooms"].between(1, ROOMS_MAX)
        & (df["land"] <= LAND_MAX)
        & (df["floor"] <= df["total_floors"])
        & df["lat"].between(LAT_MIN, LAT_MAX)
        & df["lng"].between(LNG_MIN, LNG_MAX)
    )
    df = df[keep]
    log("remove outliers (price/area/rooms/geo rules)", df)

    df = df.drop(columns=["currency", "listing_url"]).reset_index(drop=True)
    return df


def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - numeric columns are used as they are,
      - category -> one-hot (3 values),
      - location -> one-hot for the N_TOP_LOCATIONS most common districts;
        all other districts share the all-zeros "other" group.
        (One-hot keeps both models honest: no target statistics are used.)
    Return (X, y, feature_names) where y is the price in AZN.
    """
    numeric = ["area", "rooms", "floor", "total_floors", "land", "lat", "lng",
               "repair", "bill_of_sale", "mortgage"]
    columns = [df[c].to_numpy(dtype=float) for c in numeric]
    names = list(numeric)

    for value in CATEGORIES.values():
        columns.append((df["category"] == value).to_numpy(dtype=float))
        names.append("cat_" + value)

    top_locations = df["location"].value_counts().index[:N_TOP_LOCATIONS]
    for loc in top_locations:
        columns.append((df["location"] == loc).to_numpy(dtype=float))
        names.append("loc_" + to_ascii(loc))

    X = np.column_stack(columns)
    y = df["price"].to_numpy(dtype=float)
    return X, y, names


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    The threshold is the MEDIAN of the TRAINING price.
      - premium = 1 (price >  threshold), standard = 0.
    Call it once on the training prices (threshold=None) and then reuse the
    returned threshold for validation / test, so they never set it.
    Return (y_tier, threshold).
    """
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    y_tier = (y_price > threshold).astype(int)
    return y_tier, threshold


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.

    The split is stratified on the price tier without letting val/test touch
    the threshold:
      1) shuffle and take the training part,
      2) compute the tier threshold on the training prices only,
      3) split the remaining rows into validation and test separately for
         each tier, so both keep the same class balance.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))

    n_train = int(round(len(y) * (1 - val_size - test_size)))
    train_idx = idx[:n_train]
    rest_idx = idx[n_train:]

    _, threshold = make_tier_label(y[train_idx])
    rest_tier, _ = make_tier_label(y[rest_idx], threshold)

    val_share = val_size / (val_size + test_size)
    val_idx, test_idx = [], []
    for tier in [0, 1]:
        tier_idx = rest_idx[rest_tier == tier]     # already shuffled
        n_val = int(round(len(tier_idx) * val_share))
        val_idx.extend(tier_idx[:n_val])
        test_idx.extend(tier_idx[n_val:])
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0           # constant column: avoid division by zero
    scaled = [(X_tr - mean) / std]
    for X in others:
        scaled.append((X - mean) / std)
    return tuple(scaled)
