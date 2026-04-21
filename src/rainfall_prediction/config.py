from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "kathmandu_dhm.xlsx"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "weather_processed.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BEST_MODEL_INFO_PATH = MODELS_DIR / "best_model_metadata.json"

TARGET_COLUMN = "precipitation"
DATE_COLUMN = "date"
FEATURE_COLUMNS = [
    "max_temp",
    "min_temp",
    "rel_humidity",
    "pressure",
    "wind_direction",
    "wind_speed",
]

COLUMN_MAPPING = {
    "Time": DATE_COLUMN,
    "Precipitation": TARGET_COLUMN,
    "Max Temp": "max_temp",
    "Min Temp": "min_temp",
    "Rel Humidity": "rel_humidity",
    "Pressure": "pressure",
    "Wind Direction": "wind_direction",
    "Wind Speed": "wind_speed",
}


@dataclass(frozen=True)
class SplitConfig:
    test_start_date: str = "2024-01-01"

