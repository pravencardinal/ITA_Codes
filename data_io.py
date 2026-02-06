"""
data_io.py

STEP: Data Read + First-Level Structuring (ITA)

Responsibilities:
1. Fetch raw data from BigQuery (2 sources)
2. Apply required transformation functions from alteryx_replacements
3. Assemble the `raw` dictionary used downstream

NO business logic beyond this step.
"""

from typing import Dict, Optional
import pandas as pd
from google.cloud import bigquery
from logger_utils import log_execution
from config import get_default_config
from alteryx_replacements import (
    fetch_w1_sales,
    build_dat_zip_from_snapshot,
    fetch_w4_sig_withnames,
)

cfg = get_default_config()


def _get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=cfg.PROJECT)


# -------------------------------------------------
# Ship-To raw data
# -------------------------------------------------
@log_execution(
    "data_io",
    "fetch_ship_to_data",
    "Fetch ship-to level sales data from BigQuery",
    cfg.LOG_FILE
)
def fetch_ship_to_data(limit: Optional[int] = None) -> pd.DataFrame:
    client = _get_bq_client()
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
    SELECT
        Entpr_ID_3,
        Entpr_ID_4,
        Entpr_ID_5,
        Level3_Employee,
        Level4_Employee,
        Level5_Employee,
        Total_Sales,
        Ship_To_Customer,
        Ship_To_Customer_Name,
        Ship_To_Street_Address,
        Ship_To_City,
        Ship_To_State,
        C_SHIP_TO_PSTLZ_POSTAL_CD,
        SIG
    FROM `edna-data-pr-cah.VW_MED_MSADM_CMRL_CMBN_SRC.VW_SHIPTO_LEVEL_SALES_DATA`
    WHERE SIG IN ('SCSE')
    {limit_clause}
    """

    return client.query(query).result().to_dataframe(create_bqstorage_client=False)


# -------------------------------------------------
# Sales Rep / Workday raw data
# -------------------------------------------------
@log_execution(
    "data_io",
    "fetch_sales_rep_data",
    "Fetch sales rep data from Workday BigQuery table",
    cfg.LOG_FILE
)
def fetch_sales_rep_data(limit: Optional[int] = None) -> pd.DataFrame:
    client = _get_bq_client()
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
    SELECT DISTINCT
        Entpr_ID_5,
        HOMECITY,
        HOMEPOSTALCODE
    FROM `edna-data-edd-pr-cah.VW_MED_AI_ML_ITA.CORP_WRK_DAY_PUBLIC_HCM__CARDINAL_WORKDAY_DATA_CV`
    {limit_clause}
    """

    return client.query(query).result().to_dataframe(create_bqstorage_client=False)


# -------------------------------------------------
# Pipeline entry point for this step
# -------------------------------------------------
@log_execution(
    "data_io",
    "load_input_data",
    "Load raw inputs and derive dat, dat_zip, SIG_names",
    cfg.LOG_FILE
)
def load_input_data(cfg, limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:

    # 1. Fetch raw data
    ship_to_raw = fetch_ship_to_data(limit)
    sales_rep_raw = fetch_sales_rep_data(limit)

    # 2. Rename postal column BEFORE transformation
    ship_to_raw = ship_to_raw.rename(
        columns={"C_SHIP_TO_PSTLZ_POSTAL_CD": "Ship_To_Postal_Code"}
    )

    # 3. Customer-level aggregation (dat)
    dat = fetch_w1_sales(ship_to_raw)

    # 4. Sales-rep-level data (dat_zip)
    dat_zip = build_dat_zip_from_snapshot(sales_rep_raw)

    # 5. SIG names derived from dat
    sig_names = fetch_w4_sig_withnames(dat)

    return {
        "ship_to_raw": ship_to_raw,
        "sales_rep_raw": sales_rep_raw,
        "dat": dat,
        "dat_zip": dat_zip,
        "SIG_names": sig_names,
    }
