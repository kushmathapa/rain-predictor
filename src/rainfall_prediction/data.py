from __future__ import annotations

from pathlib import Path

import pandas as pd

from rainfall_prediction.config import (
    COLUMN_MAPPING,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

_BASE_WEATHER_FEATURES = [
    "max_temp",
    "min_temp",
    "rel_humidity",
    "pressure",
    "wind_direction",
    "wind_speed",
]


def load_weather_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns=COLUMN_MAPPING)
    missing_columns = {
        DATE_COLUMN,
        TARGET_COLUMN,
        *_BASE_WEATHER_FEATURES,
    } - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df[[DATE_COLUMN, TARGET_COLUMN, *_BASE_WEATHER_FEATURES]].copy()


def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    processed[DATE_COLUMN] = pd.to_datetime(processed[DATE_COLUMN], errors="coerce")

    numeric_columns = [TARGET_COLUMN, *_BASE_WEATHER_FEATURES]
    for column in numeric_columns:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")

    processed = processed.dropna(subset=[DATE_COLUMN, TARGET_COLUMN]).sort_values(DATE_COLUMN)

    # Seasonality features (no target leakage).
    processed["month"] = processed[DATE_COLUMN].dt.month.astype("Int64")
    processed["day_of_year"] = processed[DATE_COLUMN].dt.dayofyear.astype("Int64")

    # Lag features to capture temporal dependence (t-1, t-2, t-3).
    processed["precip_lag1"] = processed[TARGET_COLUMN].shift(1)
    processed["precip_lag2"] = processed[TARGET_COLUMN].shift(2)
    processed["precip_lag3"] = processed[TARGET_COLUMN].shift(3)

    # The first few rows cannot have lag features.
    processed = processed.dropna(subset=["precip_lag1", "precip_lag2", "precip_lag3"])

    # Keep the final modeling table in a consistent column order.
    processed = processed[[DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]].copy()

    return processed.reset_index(drop=True)


def save_processed_data(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
