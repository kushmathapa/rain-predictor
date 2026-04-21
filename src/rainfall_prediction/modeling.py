from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from rainfall_prediction.config import (
    BASELINE_MODEL_COMPARISON_PATH,
    BEST_MODEL_INFO_PATH,
    BEST_MODEL_PATH,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    MODEL_COMPARISON_PATH,
    TARGET_COLUMN,
)
from rainfall_prediction.runtime import bootstrap_openmp_runtime


matplotlib.use("Agg")
import matplotlib.pyplot as plt

bootstrap_openmp_runtime()


@dataclass
class ModelingArtifacts:
    baseline_comparison: pd.DataFrame
    comparison: pd.DataFrame
    best_model_name: str
    best_model: Any


def split_train_test(df: pd.DataFrame, test_start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df[DATE_COLUMN] < test_start_date].copy()
    test = df[df[DATE_COLUMN] >= test_start_date].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split failed. Check the dataset date range and test split date.")
    return train, test


def _build_model_specs() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    spaces: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "catboost": (
            CatBoostRegressor(random_state=42, verbose=0),
            {
                "iterations": [300, 500],
                "learning_rate": [0.01, 0.1],
                "depth": [4, 6],
            },
        ),
    }
    unavailable_models: dict[str, str] = {}

    try:
        from xgboost import XGBRegressor

        spaces["xgboost"] = (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.7, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        )
    except Exception as exc:
        unavailable_models["xgboost"] = str(exc)

    try:
        from lightgbm import LGBMRegressor

        spaces["lightgbm"] = (
            LGBMRegressor(random_state=42, verbosity=-1, n_jobs=1),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "num_leaves": [20, 31],
                "max_depth": [3, 5],
                "subsample": [0.7, 1.0],
            },
        )
    except Exception as exc:
        unavailable_models["lightgbm"] = str(exc)

    ordered_spaces = {name: spaces[name] for name in ["xgboost", "lightgbm", "catboost"] if name in spaces}
    ordered_unavailable = {
        name: unavailable_models[name]
        for name in ["xgboost", "lightgbm", "catboost"]
        if name in unavailable_models
    }

    return ordered_spaces, ordered_unavailable


def train_and_evaluate_models(
    df: pd.DataFrame,
    test_start_date: str,
    baseline_reports_path: Path = BASELINE_MODEL_COMPARISON_PATH,
    reports_path: Path = MODEL_COMPARISON_PATH,
) -> ModelingArtifacts:
    train_df, test_df = split_train_test(df, test_start_date=test_start_date)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    baseline_results: list[dict[str, Any]] = []
    tuned_results: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    baseline_predictions: dict[str, pd.Series] = {}
    tuned_predictions: dict[str, pd.Series] = {}
    model_specs, unavailable_models = _build_model_specs()

    for model_name, error_message in unavailable_models.items():
        baseline_results.append(
            {
                "model": model_name,
                "status": "unavailable",
                "mse": None,
                "r2": None,
                "best_params": error_message,
            }
        )
        tuned_results.append(
            {
                "model": model_name,
                "status": "unavailable",
                "mse": None,
                "r2": None,
                "best_params": error_message,
            }
        )

    for model_name, (estimator, param_grid) in model_specs.items():
        baseline_model = clone(estimator)
        baseline_model.fit(x_train, y_train)
        baseline_pred = pd.Series(baseline_model.predict(x_test))
        baseline_results.append(
            {
                "model": model_name,
                "status": "trained",
                "mse": mean_squared_error(y_test, baseline_pred),
                "r2": r2_score(y_test, baseline_pred),
                "best_params": "default_parameters",
            }
        )
        baseline_predictions[model_name] = baseline_pred

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=1,
            verbose=0,
        )
        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        predictions = best_model.predict(x_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        tuned_results.append(
            {
                "model": model_name,
                "status": "trained",
                "mse": mse,
                "r2": r2,
                "best_params": json.dumps(search.best_params_),
            }
        )
        fitted_models[model_name] = best_model
        tuned_predictions[model_name] = pd.Series(predictions)

        _save_prediction_plot(stage="after_tuning", model_name=model_name, y_true=y_test.reset_index(drop=True), y_pred=pd.Series(predictions))
        _save_feature_importance_plot(model_name=model_name, model=best_model)

    if not fitted_models:
        raise RuntimeError("No model could be trained in the current environment.")

    baseline_comparison = _sort_comparison_rows(pd.DataFrame(baseline_results))
    comparison = _sort_comparison_rows(pd.DataFrame(tuned_results))
    baseline_reports_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_comparison.to_csv(baseline_reports_path, index=False)
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports_path, index=False)

    trained_rows = comparison[comparison["status"] == "trained"].sort_values(
        ["mse", "r2"],
        ascending=[True, False],
    )

    best_model_name = trained_rows.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    _save_best_model(best_model_name, best_model, trained_rows.iloc[0].to_dict())
    _save_combined_prediction_plot(
        stage="before_tuning",
        y_true=y_test.reset_index(drop=True),
        predictions=baseline_predictions,
    )
    _save_combined_prediction_plot(
        stage="after_tuning",
        y_true=y_test.reset_index(drop=True),
        predictions=tuned_predictions,
    )

    return ModelingArtifacts(
        baseline_comparison=baseline_comparison,
        comparison=comparison,
        best_model_name=best_model_name,
        best_model=best_model,
    )


def _sort_comparison_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    trained_rows = comparison[comparison["status"] == "trained"].sort_values(
        ["mse", "r2"],
        ascending=[True, False],
    )
    unavailable_rows = comparison[comparison["status"] == "unavailable"]
    return pd.concat([trained_rows, unavailable_rows], ignore_index=True)


def _save_prediction_plot(stage: str, model_name: str, y_true: pd.Series, y_pred: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true.index, y_true.values, label="Actual", linewidth=1.5)
    ax.plot(y_pred.index, y_pred.values, label="Predicted", linewidth=1.5)
    ax.set_title(f"Actual vs Predicted Rainfall - {model_name} - {stage.replace('_', ' ').title()}")
    ax.set_xlabel("Test Sample")
    ax.set_ylabel("Precipitation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{stage}_{model_name}_predictions.png", dpi=200)
    plt.close(fig)


def _save_combined_prediction_plot(stage: str, y_true: pd.Series, predictions: dict[str, pd.Series]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_names = list(predictions.keys())
    fig, axes = plt.subplots(len(model_names), 1, figsize=(12, 4 * len(model_names)), sharex=True)
    if len(model_names) == 1:
        axes = [axes]
    for idx, model_name in enumerate(model_names):
        axes[idx].plot(y_true.index, y_true.values, label="Actual", linewidth=1.4)
        axes[idx].plot(predictions[model_name].index, predictions[model_name].values, label="Predicted", linewidth=1.4)
        axes[idx].set_title(model_name)
        axes[idx].set_ylabel("Precipitation")
        axes[idx].legend()
    axes[-1].set_xlabel("Test Sample")
    fig.suptitle(f"Prediction on Testing Dataset - {stage.replace('_', ' ').title()}", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{stage}_prediction_comparison.png", dpi=200)
    plt.close(fig)


def _save_feature_importance_plot(model_name: str, model: Any) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(FEATURE_COLUMNS, importances, color="#ff7f0e")
    ax.set_title(f"Feature Importance - {model_name}")
    ax.set_ylabel("Importance")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{model_name}_feature_importance.png", dpi=200)
    plt.close(fig)


def _save_best_model(best_model_name: str, best_model: Any, best_row: dict[str, Any]) -> None:
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    metadata = {
        "model": best_model_name,
        "features": FEATURE_COLUMNS,
        "metrics": {
            "mse": float(best_row["mse"]),
            "r2": float(best_row["r2"]),
        },
        "best_params": best_row["best_params"],
    }
    BEST_MODEL_INFO_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_best_model(path: Path = BEST_MODEL_PATH) -> Any:
    return joblib.load(path)
