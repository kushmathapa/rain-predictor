# Rainfall Prediction With Gradient Boosting Models

This project implements a rainfall prediction workflow for Kathmandu weather data:

`Data -> Preprocessing -> EDA -> Train Models -> Hyperparameter Tuning -> Evaluation -> Prediction`

The pipeline uses three gradient boosting regressors:

- XGBoost
- LightGBM
- CatBoost

## Project Layout

```text
rain-predictor/
├── data/
│   ├── raw/kathmandu_dhm.xlsx
│   └── processed/
├── models/
├── reports/
│   └── figures/
├── src/rainfall_prediction/
├── notebooks/
├── pyproject.toml
└── requirements.txt
```

## Dataset

The raw workbook is stored at `data/raw/kathmandu_dhm.xlsx`.

Detected source columns:

- `Time`
- `Precipitation`
- `Max Temp`
- `Min Temp`
- `Rel Humidity`
- `Pressure`
- `Wind Direction`
- `Wind Speed`

The code standardizes these to snake case and uses:

- Target: `precipitation`
- Features: `max_temp`, `min_temp`, `rel_humidity`, `pressure`, `wind_direction`, `wind_speed`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Run The Full Workflow

From the project root:

```bash
rainfall-predictor run-all
```

This will:

1. Load the Excel dataset
2. Clean and preprocess it
3. Save the processed dataset to `data/processed/weather_processed.csv`
4. Generate EDA charts in `reports/figures/`
5. Train baseline XGBoost, LightGBM, and CatBoost models
6. Tune all three models with `GridSearchCV` search spaces
7. Evaluate them on the temporal split
8. Save before-tuning metrics to `reports/baseline_model_comparison.csv`
9. Save after-tuning metrics to `reports/model_comparison.csv`
10. Save the best model to `models/best_model.joblib`

## Prediction

After training, make a prediction with the saved best model:

```bash
rainfall-predictor predict \
  --max-temp 26 \
  --min-temp 14 \
  --rel-humidity 80 \
  --pressure 865 \
  --wind-direction 120 \
  --wind-speed 3
```

## Notes

- The code uses mean imputation for missing feature values.
- Rows with missing `date` or `precipitation` are dropped because they cannot be used for temporal modeling.
- Temporal split:
  - Training: dates before `2024-01-01`
  - Testing: dates on or after `2024-01-01`
- The implementation produces both before-tuning and after-tuning model comparisons.
- Hyperparameter tuning uses `GridSearchCV` with 3-fold cross-validation.
- The EDA output includes `time_series_all_variables.png` in addition to a rainfall-only time series.
- On macOS, the project installs a user-space `libomp` runtime and preloads it automatically before importing XGBoost or LightGBM. This avoids the common Apple Silicon vs Intel Homebrew mismatch.
