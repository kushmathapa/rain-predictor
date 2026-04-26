from __future__ import annotations

import argparse

from rainfall_prediction.modeling import load_best_model
from rainfall_prediction.pipeline import build_prediction_frame, run_full_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rainfall prediction workflow for Kathmandu weather data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_all = subparsers.add_parser("run-all", help="Run preprocessing, EDA, training, tuning, and evaluation.")
    run_all.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip generation of EDA plots.",
    )

    predict = subparsers.add_parser("predict", help="Predict rainfall using the saved best model.")
    predict.add_argument(
        "--date",
        type=str,
        required=True,
        help="Observation date in YYYY-MM-DD format (used to derive seasonality features).",
    )
    predict.add_argument("--max-temp", type=float, required=True)
    predict.add_argument("--min-temp", type=float, required=True)
    predict.add_argument("--rel-humidity", type=float, required=True)
    predict.add_argument("--pressure", type=float, required=True)
    predict.add_argument("--wind-direction", type=float, required=True)
    predict.add_argument("--wind-speed", type=float, required=True)
    predict.add_argument(
        "--precip-lag1",
        type=float,
        default=None,
        help="Previous observation precipitation (t-1). Optional; if omitted the model will impute.",
    )
    predict.add_argument(
        "--precip-lag2",
        type=float,
        default=None,
        help="Precipitation lag (t-2). Optional; if omitted the model will impute.",
    )
    predict.add_argument(
        "--precip-lag3",
        type=float,
        default=None,
        help="Precipitation lag (t-3). Optional; if omitted the model will impute.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-all":
        artifacts = run_full_pipeline(generate_eda=not args.skip_eda)
        print("Model comparison before hyperparameter tuning:")
        print(artifacts.baseline_comparison.to_string(index=False))
        print("\nModel comparison after hyperparameter tuning:")
        print(artifacts.comparison.to_string(index=False))
        print(f"\nBest model: {artifacts.best_model_name} ({artifacts.best_model_stage})")
        return

    if args.command == "predict":
        model = load_best_model()
        features = build_prediction_frame(
            date=args.date,
            max_temp=args.max_temp,
            min_temp=args.min_temp,
            rel_humidity=args.rel_humidity,
            pressure=args.pressure,
            wind_direction=args.wind_direction,
            wind_speed=args.wind_speed,
            precip_lag1=args.precip_lag1,
            precip_lag2=args.precip_lag2,
            precip_lag3=args.precip_lag3,
        )
        prediction = model.predict(features)[0]
        print(f"Predicted rainfall: {prediction:.4f}")
        return

    parser.error("Unknown command.")


if __name__ == "__main__":
    main()
