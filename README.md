# Rainfall Prediction With Gradient Boosting Models

This project implements the thesis workflow for Kathmandu rainfall prediction:

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
5. Tune and train XGBoost, LightGBM, and CatBoost
6. Evaluate them on a temporal test split
7. Save model metrics to `reports/model_comparison.csv`
8. Save the best model to `models/best_model.joblib`

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

- The code uses mean imputation for missing feature values, matching the thesis direction.
- Rows with missing `date` or `precipitation` are dropped because they cannot be used for temporal modeling.
- Temporal split:
  - Training: dates before `2024-01-01`
  - Testing: dates on or after `2024-01-01`
- Hyperparameter tuning uses `GridSearchCV` with `TimeSeriesSplit` to respect the time order within the training set.
- On macOS, both XGBoost and LightGBM may require an architecture-compatible `libomp` runtime. If either cannot be loaded, the pipeline will continue with the remaining models and mark the unavailable ones in the comparison report.
