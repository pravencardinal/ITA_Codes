"""
config.py

Single responsibility:
- Load config.yaml
- Expose GAConfig with ALL attributes required by existing modules

DO NOT remove attributes without full pipeline audit.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Optional
from logger_utils import init_log

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


@dataclass
class GAConfig:
    # Core
    PROJECT: str
    USE_BIGQUERY: bool

    # Paths
    LOCAL_DATA_DIR: str
    OUTPUT_DIR: str
    GCS_BUCKET: str

    # Logging (REQUIRED)
    LOG_FILE: str
    LOG_LEVEL: str   # <-- CRITICAL ATTRIBUTE (RESTORED)

    # Outputs
    DAT_CSV: str
    SALES_REP_DATA_V2: str
    SIG_NAMES_CSV: str
    SIG_PICKLE: str

    # GA params
    POPULATION_SIZE: int
    MAX_GENERATIONS: int
    MUTATION_RATE: float
    CROSSOVER_RATE: float
    USE_DISTANCE_METRIC: bool
    SEED: int
    
    GEO_ENABLED: bool
    GEO_COUNTRY: str
    GEO_LOOKUP_FILE: str
    GEO_LOOKUP_DIR: str


def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.yaml not found at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_default_config(path: Optional[str] = None) -> GAConfig:
    """
    Load config.yaml and return GAConfig.
    This function is SAFE to call from any module.
    """
    cfg_path = path or os.getenv("CONFIG_YAML", DEFAULT_CONFIG_PATH)
    raw = _load_yaml(cfg_path)

    cfg = GAConfig(
        PROJECT=raw.get("PROJECT", "edd-trn-mdai-pr-cah"),
        USE_BIGQUERY=bool(raw.get("USE_BIGQUERY", True)),

        LOCAL_DATA_DIR=raw.get("LOCAL_DATA_DIR", "/home/jupyter/data"),
        OUTPUT_DIR=raw.get("OUTPUT_DIR", "./output"),
        GCS_BUCKET=raw.get("GCS_BUCKET", ""),

        LOG_FILE=raw.get("LOG_FILE", "log.txt"),
        LOG_LEVEL=raw.get("LOG_LEVEL", "INFO"),  # <-- DEFAULT SAFE VALUE

        DAT_CSV=raw.get("DAT_CSV", "./output/dat.csv"),
        SALES_REP_DATA_V2=raw.get("SALES_REP_DATA_V2", "./output/sales_rep_data_v2.csv"),
        SIG_NAMES_CSV=raw.get("SIG_NAMES_CSV", "./output/SCSE_withnames_V3.csv"),
        SIG_PICKLE=raw.get("SIG_PICKLE", "./output/lvl3tolvl5_dict_v4.pkl"),

        POPULATION_SIZE=int(raw.get("POPULATION_SIZE", 200)),
        MAX_GENERATIONS=int(raw.get("MAX_GENERATIONS", 200)),
        MUTATION_RATE=float(raw.get("MUTATION_RATE", 0.02)),
        CROSSOVER_RATE=float(raw.get("CROSSOVER_RATE", 0.5)),
        USE_DISTANCE_METRIC=bool(raw.get("USE_DISTANCE_METRIC", True)),
        
        GEO_ENABLED=bool(raw.get("GEO_ENABLED", True)),
        GEO_COUNTRY=raw.get("GEO_COUNTRY", "US"),
        GEO_LOOKUP_FILE=raw.get("GEO_LOOKUP_FILE", "postal_geo_lookup_us.csv"),
        GEO_LOOKUP_DIR=raw.get("GEO_LOOKUP_DIR", "geo"),
        SEED=int(raw.get("SEED", 42)),
    )

    # Initialize logger once config is fully valid
    init_log(cfg.LOG_FILE)

    return cfg

@property
def GEO_LOOKUP_PATH(self) -> str:
    return os.path.join(self.GEO_LOOKUP_DIR, self.GEO_LOOKUP_FILE)