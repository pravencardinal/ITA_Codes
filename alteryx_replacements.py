"""
alteryx_replacements.py

Pure transformation layer.
NO SQL.
NO BigQuery calls.
"""

import pandas as pd
import numpy as np
import pgeocode
from logger_utils import log_execution
from config import get_default_config

cfg = get_default_config()


# -------------------------------
# Simple pgeocode-based geocoder
# -------------------------------
def get_lat_long_offline(zipcodes: pd.Series, country: str = "US") -> pd.DataFrame:
    """
    Vectorized postal-code → lat/lon using pgeocode
    """
    nom = pgeocode.Nominatim(country)
    zipcodes = zipcodes.astype(str).str.strip()
    unique_zips = zipcodes.dropna().unique()

    lookup = {}
    for z in unique_zips:
        try:
            res = nom.query_postal_code(z)
            if res is not None and not pd.isna(res.latitude):
                lookup[z] = (res.latitude, res.longitude)
            else:
                lookup[z] = (np.nan, np.nan)
        except Exception:
            lookup[z] = (np.nan, np.nan)

    return pd.DataFrame({
        "lat": zipcodes.map(lambda z: lookup.get(z, (np.nan, np.nan))[0]),
        "lon": zipcodes.map(lambda z: lookup.get(z, (np.nan, np.nan))[1]),
    })


@log_execution(
    "alteryx",
    "fetch_w1_sales",
    "Transform ship-to raw data into inference-ready dat",
    cfg.LOG_FILE
)
def fetch_w1_sales(ship_to_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Final inference dat.csv
    """
    df = ship_to_raw.copy()
    df = df[~df["Entpr_ID_5"].astype(str).str.contains("Open", na=False)]
    df["Total_Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)

    dat = (
        df.groupby(["Ship_To_Customer", "SIG"], as_index=False)
        .agg({
            "Total_Sales": "sum",
            "Ship_To_Customer_Name": "first",
            "Ship_To_Street_Address": "first",
            "Ship_To_City": "first",
            "Ship_To_State": "first",
            "C_SHIP_TO_PSTLZ_POSTAL_CD": "first",
            "Entpr_ID_3": "first",
            "Entpr_ID_4": "first",
            "Entpr_ID_5": "first",
            "Level3_Employee": "first",
            "Level4_Employee": "first",
            "Level5_Employee":"first",
        })
    )

    geo = get_lat_long_offline(dat["Ship_To_Postal_Code"])
    dat["lat"] = geo["lat"]
    dat["lon"] = geo["lon"]

    return dat


@log_execution(
    "alteryx",
    "build_dat_zip_from_snapshot",
    "Build sales_rep_data_v2 from sales-rep raw data",
    cfg.LOG_FILE
)
def build_dat_zip_from_snapshot(sales_rep_raw: pd.DataFrame) -> pd.DataFrame:
    """
    sales_rep_data_v2
    """
    df = sales_rep_raw.copy()
    df = df[~df["Entpr_ID_5"].astype(str).str.contains("Open", na=False)]
    df = df.drop_duplicates(subset=["Entpr_ID_5"])

    entprs = sorted(df["Entpr_ID_5"].astype(str).unique())
    entpr_map = {e: i for i, e in enumerate(entprs)}
    df["SIG_lvl5_int"] = df["Entpr_ID_5"].astype(str).map(entpr_map)

    geo = get_lat_long_offline(df["HOMEPOSTALCODE"])
    df["lat"] = geo["lat"]
    df["lon"] = geo["lon"]

    return df[
        ["Entpr_ID_5", "SIG_lvl5_int", "lat", "lon"]
    ]


@log_execution(
    "alteryx",
    "fetch_w4_sig_withnames",
    "Derive SIG-with-names from inference dat",
    cfg.LOG_FILE
)
def fetch_w4_sig_withnames(dat: pd.DataFrame) -> pd.DataFrame:
    """
    Derived SCSE_withnames (no SQL)
    """
    return (
        dat[["Entpr_ID_3", "Entpr_ID_4", "Entpr_ID_5", "SIG"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
