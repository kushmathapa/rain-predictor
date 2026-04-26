import os
import unittest

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rainfall_prediction.config import DATE_COLUMN, FEATURE_COLUMNS, TARGET_COLUMN
from rainfall_prediction.modeling import train_and_evaluate_models


class TestModelingInvariants(unittest.TestCase):
    def _make_featured_df(self) -> pd.DataFrame:
        # Build a small, already-feature-engineered dataset so tests don't depend on Excel.
        dates = pd.date_range("2023-12-20", periods=20, freq="D")
        precipitation = pd.Series([0, 0, 1, 0, 2, 0, 0, 5, 0, 0, 1, 0, 0, 3, 0, 0, 2, 0, 0, 1])

        # Ensure train (< 2024-01-01) has max_temp ~= 0 and test (>= 2024-01-01) has max_temp ~= 100.
        max_temp = [0.0 if d < pd.Timestamp("2024-01-01") else 100.0 for d in dates]

        df = pd.DataFrame(
            {
                DATE_COLUMN: dates,
                TARGET_COLUMN: precipitation,
                # Base weather features:
                "max_temp": max_temp,
                "min_temp": [0.0] * 20,
                "rel_humidity": [0.0] * 20,
                "pressure": [0.0] * 20,
                "wind_direction": [0.0] * 20,
                "wind_speed": [0.0] * 20,
            }
        )

        # Add seasonality.
        df["month"] = df[DATE_COLUMN].dt.month.astype(int)
        df["day_of_year"] = df[DATE_COLUMN].dt.dayofyear.astype(int)

        # Add lag features from target.
        df["precip_lag1"] = df[TARGET_COLUMN].shift(1)
        df["precip_lag2"] = df[TARGET_COLUMN].shift(2)
        df["precip_lag3"] = df[TARGET_COLUMN].shift(3)
        df = df.dropna(subset=["precip_lag1", "precip_lag2", "precip_lag3"]).reset_index(drop=True)

        # Introduce a missing value in TRAIN only. If imputation leaked, the mean would
        # incorporate test values and shift away from 0.
        df.loc[0, "max_temp"] = None

        # Keep only the expected modeling columns.
        return df[[DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]].copy()

    def test_train_only_imputation_and_time_aware_tuning(self) -> None:
        df = self._make_featured_df()

        artifacts = train_and_evaluate_models(
            df,
            test_start_date="2024-01-01",
            model_names=["linear_regression"],  # keep tests fast and deterministic
        )

        # Baseline + tuned comparison should include MAE and cleaned params.
        self.assertIn("mae", artifacts.baseline_comparison.columns)
        self.assertIn("mae", artifacts.comparison.columns)

        tuned_row = artifacts.comparison.iloc[0].to_dict()
        self.assertNotIn("model__", str(tuned_row.get("best_params")))

        # Verify the imputer was fit on train only by checking the learned mean for max_temp.
        # Train period has max_temp ~= 0, test period has max_temp ~= 100. If leakage occurred,
        # the mean would be pulled upward.
        best = artifacts.best_model
        self.assertTrue(hasattr(best, "named_steps"))
        imputer = best.named_steps["imputer"]
        # Find max_temp index in FEATURE_COLUMNS order.
        max_temp_idx = FEATURE_COLUMNS.index("max_temp")
        self.assertLess(imputer.statistics_[max_temp_idx], 10.0)
