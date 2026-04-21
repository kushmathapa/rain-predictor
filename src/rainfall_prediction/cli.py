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
    predict.add_argument("--max-temp", type=float, required=True)
    predict.add_argument("--min-temp", type=float, required=True)
    predict.add_argument("--rel-humidity", type=float, required=True)
    predict.add_argument("--pressure", type=float, required=True)
    predict.add_argument("--wind-direction", type=float, required=True)
    predict.add_argument("--wind-speed", type=float, required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-all":
        artifacts = run_full_pipeline(generate_eda=not args.skip_eda)
        print("Model comparison:")
        print(artifacts.comparison.to_string(index=False))
        print(f"\nBest model: {artifacts.best_model_name}")
        return

    if args.command == "predict":
        model = load_best_model()
        features = build_prediction_frame(
            max_temp=args.max_temp,
            min_temp=args.min_temp,
            rel_humidity=args.rel_humidity,
            pressure=args.pressure,
            wind_direction=args.wind_direction,
            wind_speed=args.wind_speed,
        )
        prediction = model.predict(features)[0]
        print(f"Predicted rainfall: {prediction:.4f}")
        return

    parser.error("Unknown command.")


if __name__ == "__main__":
    main()
