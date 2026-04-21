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


def load_weather_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns=COLUMN_MAPPING)
    missing_columns = {
        DATE_COLUMN,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    } - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df[[DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]].copy()


def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    processed[DATE_COLUMN] = pd.to_datetime(processed[DATE_COLUMN], errors="coerce")

    numeric_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    for column in numeric_columns:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")

    processed = processed.dropna(subset=[DATE_COLUMN, TARGET_COLUMN]).sort_values(DATE_COLUMN)

    feature_means = processed[FEATURE_COLUMNS].mean(numeric_only=True)
    processed[FEATURE_COLUMNS] = processed[FEATURE_COLUMNS].fillna(feature_means)

    return processed.reset_index(drop=True)


def save_processed_data(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

