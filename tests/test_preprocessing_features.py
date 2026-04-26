import unittest

import os
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rainfall_prediction.config import DATE_COLUMN, FEATURE_COLUMNS, TARGET_COLUMN
from rainfall_prediction.data import preprocess_weather_data


class TestPreprocessingFeatures(unittest.TestCase):
    def test_preprocess_adds_time_and_lag_features(self) -> None:
        df = pd.DataFrame(
            {
                DATE_COLUMN: pd.date_range("2024-01-01", periods=10, freq="D"),
                TARGET_COLUMN: [0, 1, 0, 2, 0, 0, 5, 0, 0, 1],
                "max_temp": [20] * 10,
                "min_temp": [10] * 10,
                "rel_humidity": [80] * 10,
                "pressure": [860] * 10,
                "wind_direction": [90] * 10,
                "wind_speed": [3] * 10,
            }
        )

        processed = preprocess_weather_data(df)

        # Lags require dropping the first 3 rows.
        self.assertEqual(len(processed), 7)

        for col in [DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]:
            self.assertIn(col, processed.columns)

        # Lag columns should be present and non-null after dropping the initial rows.
        self.assertFalse(processed["precip_lag1"].isna().any())
        self.assertFalse(processed["precip_lag2"].isna().any())
        self.assertFalse(processed["precip_lag3"].isna().any())

        # Seasonality columns should be present and non-null.
        self.assertFalse(processed["month"].isna().any())
        self.assertFalse(processed["day_of_year"].isna().any())
