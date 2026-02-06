"""
data_io.py

PRODUCTION FINAL – ITA PROJECT

Responsibilities:
1. Read raw input data from BigQuery (2 finalized sources)
2. Invoke transformation functions from alteryx_replacements
3. Assemble the `raw` dictionary used downstream
4. Provide ALL persistence utilities required by the pipeline:
   - save_checkpoint
   - load_checkpoint
   - save_predictions_csv

IMPORTANT:
- NO business logic
- NO GA logic
- NO preprocessing logic
- SQL lives ONLY here
"""

from typing import Dict, Any, Optional
import os
import pickle
import pandas as pd
from google.cloud import bigquery

from logger_utils import log_execution
from config import get_default_config
from alteryx_replacements import (
    fetch_w1_sales,
    build_dat_zip_from_snapshot,
    fetch_w4_sig_withnames,
)

# ------------------------------------------------------------------
# Load configuration ONCE
# ------------------------------------------------------------------
cfg = get_default_config()


# ------------------------------------------------------------------
# BigQuery helper
# ------------------------------------------------------------------
def _get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=cfg.PROJECT)


# ------------------------------------------------------------------
# 1. Ship-To raw data (Input 1)
# ------------------------------------------------------------------
@log_execution(
    "data_io",
    "fetch_ship_to_data",
    "Fetch ship-to level sales data from BigQuery",
    cfg.LOG_FILE
)
def fetch_ship_to_data(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Source:
    edna-data-pr-cah.VW_MED_MSADM_CMRL_CMBN_SRC.VW_SHIPTO_LEVEL_SALES_DATA
    """
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

    return client.query(query).result().to_dataframe(
        create_bqstorage_client=False
    )


# ------------------------------------------------------------------
# 2. Sales Rep / Workday raw data (Input 2)
# ------------------------------------------------------------------
@log_execution(
    "data_io",
    "fetch_sales_rep_data",
    "Fetch sales rep data from Workday BigQuery table",
    cfg.LOG_FILE
)
def fetch_sales_rep_data(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Source:
    edna-data-edd-pr-cah.VW_MED_AI_ML_ITA.CORP_WRK_DAY_PUBLIC_HCM__CARDINAL_WORKDAY_DATA_CV
    """
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

    return client.query(query).result().to_dataframe(
        create_bqstorage_client=False
    )


# ------------------------------------------------------------------
# 3. Pipeline entry – build RAW dictionary
# ------------------------------------------------------------------
@log_execution(
    "data_io",
    "load_input_data",
    "Load raw inputs and derive dat, dat_zip, SIG_names",
    cfg.LOG_FILE
)
def load_input_data(cfg, limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    This is the ONLY function main.py should call.

    Output contract:
    raw = {
        ship_to_raw,
        sales_rep_raw,
        dat,
        dat_zip,
        SIG_names
    }
    """

    # --- Fetch raw data ---
    ship_to_raw = fetch_ship_to_data(limit)
    sales_rep_raw = fetch_sales_rep_data(limit)

    # --- Mandatory rename BEFORE transformation ---
    ship_to_raw = ship_to_raw.rename(
        columns={"C_SHIP_TO_PSTLZ_POSTAL_CD": "Ship_To_Postal_Code"}
    )

    # --- Transformations ---
    dat = fetch_w1_sales(ship_to_raw)
    dat_zip = build_dat_zip_from_snapshot(sales_rep_raw)
    sig_names = fetch_w4_sig_withnames(dat)

    return {
        "ship_to_raw": dat,
        "sales_rep_raw": dat_zip,
        "SIG_names": sig_names,
    }


# ------------------------------------------------------------------
# 4. Checkpoint utilities (REQUIRED by GAEngine)
# ------------------------------------------------------------------
@log_execution(
    "data_io",
    "save_checkpoint",
    "Persist GA checkpoint object to disk",
    cfg.LOG_FILE
)
def save_checkpoint(obj: Any, file_path: str) -> None:
    """
    Save any Python object as a pickle checkpoint.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


@log_execution(
    "data_io",
    "load_checkpoint",
    "Load GA checkpoint object from disk",
    cfg.LOG_FILE
)
def load_checkpoint(file_path: str) -> Any:
    """
    Load checkpoint object if present, else None.
    """
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------
# 5. Output persistence (REQUIRED by reporting)
# ------------------------------------------------------------------
@log_execution(
    "data_io",
    "save_predictions_csv",
    "Save predictions or final output to CSV",
    cfg.LOG_FILE
)
def save_predictions_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Persist final model outputs to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
