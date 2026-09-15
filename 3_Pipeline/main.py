"""
Entry point for the Binance ML pipeline.

Usage:
    python main.py
    python main.py --symbol ETHUSDT --interval 15m --start-year 2022 --start-month 1 --end-year 2024 --end-month 1
    python main.py --target-threshold 0.003 --horizon 1h
    python main.py --val-ratio 0.2 --test-ratio 0.1
"""
import argparse
import logging
import sys

from flow import binance_ml_pipeline


def configure_logging(level: str = "INFO") -> None:
    """
    Configure root logging so that logger.info(...) calls inside
    DataPreprocessor / FeatureEngineer (plain `logging.getLogger(__name__)`)
    actually get printed. Prefect's own get_run_logger() output inside
    @task-decorated functions works independently of this.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end Binance ML pipeline (acquire → preprocess/label → feature engineer → align/save → train/evaluate)."
    )

    # --- Data acquisition ---
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol (default: BTCUSDT)")
    parser.add_argument("--interval", type=str, default="15m", help="Candle interval, e.g. 15m, 1h (default: 15m)")
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--end-month", type=int, default=1)

    # --- Output directories ---
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--features-dir", type=str, default="data/features")

    # --- Target configuration ---
    parser.add_argument(
        "--target-threshold", type=float, default=0.002,
        help="Return magnitude required to label Bull/Bear vs Neutral (default: 0.002)"
    )
    parser.add_argument(
        "--horizon", type=str, default="30m",
        help="How far ahead to predict, must be a multiple of --interval (default: 30m)"
    )

    # --- Model training / evaluation ---
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Fraction of data used for validation (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Fraction of data held out for final test evaluation (default: 0.15)"
    )

    # --- Logging ---
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Launching binance_ml_pipeline flow with the following config:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    x_path, y_path, training_result = binance_ml_pipeline(
        symbol=args.symbol,
        interval=args.interval,
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        features_dir=args.features_dir,
        target_threshold=args.target_threshold,
        horizon=args.horizon,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    logger.info("Pipeline finished successfully.")
    logger.info(f"  X saved to: {x_path}")
    logger.info(f"  y saved to: {y_path}")
    logger.info(f"  Best model: {training_result['best_model_name']}")
    logger.info(f"  Test accuracy: {training_result['test_metrics']['test_accuracy']:.4f}")
    logger.info(f"  Test F1 macro: {training_result['test_metrics']['test_f1_macro']:.4f}")
    logger.info(f"  MLflow run ID: {training_result['run_id']}")


if __name__ == "__main__":
    main()