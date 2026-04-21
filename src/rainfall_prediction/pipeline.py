from __future__ import annotations

import pandas as pd

from rainfall_prediction.config import FEATURE_COLUMNS, SplitConfig
from rainfall_prediction.data import load_weather_data, preprocess_weather_data, save_processed_data
from rainfall_prediction.eda import generate_eda_reports
from rainfall_prediction.modeling import ModelingArtifacts, train_and_evaluate_models


def run_full_pipeline(generate_eda: bool = True) -> ModelingArtifacts:
    raw_df = load_weather_data()
    processed_df = preprocess_weather_data(raw_df)
    save_processed_data(processed_df)

    if generate_eda:
        generate_eda_reports(processed_df)

    split_config = SplitConfig()
    return train_and_evaluate_models(processed_df, test_start_date=split_config.test_start_date)


def build_prediction_frame(
    max_temp: float,
    min_temp: float,
    rel_humidity: float,
    pressure: float,
    wind_direction: float,
    wind_speed: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "max_temp": max_temp,
                "min_temp": min_temp,
                "rel_humidity": rel_humidity,
                "pressure": pressure,
                "wind_direction": wind_direction,
                "wind_speed": wind_speed,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
