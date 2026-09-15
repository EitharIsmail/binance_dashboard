import os
from datetime import datetime
from typing import Optional
from prefect import flow, task, get_run_logger

from src.Data.data_acquisition import BinanceDataAcquisition
from src.Data.data_preprocessing import DataPreprocessor
from src.Features.feature_engineering import FeatureEngineer, align_features_and_target
from src.Models.model_training import ModelTrainingPipeline



# =============================================================================
# TASK 1 — Data Acquisition
# =============================================================================
@task(name="acquire-data", retries=3, retry_delay_seconds=10, timeout_seconds=600)
def acquire_data(start_date: datetime, end_date: datetime, symbol: str, interval: str, output_dir: str):
    """Download, process, and save the raw Binance dataset."""
    logger = get_run_logger()
    logger.info("📥 Step 1: Data Acquisition")

    acquisition = BinanceDataAcquisition(symbol=symbol, interval=interval, output_dir=output_dir)
    df_raw = acquisition.run(start_date=start_date, end_date=end_date)

    logger.info(f"✅ Loaded {len(df_raw):,} rows, {df_raw.shape[1]} columns")
    return df_raw


# =============================================================================
# TASK 2 — Data Preprocessing (ETL + Cleaning)
# =============================================================================
@task(name="preprocess-data", retries=1)
def preprocess_data(raw_df, output_dir: str, target_threshold: float, interval: str, horizon: str):
    """Clean the raw data AND create the target. Returns (X, y)."""
    logger = get_run_logger()
    logger.info(" Step 2: Data Preprocessing, Cleaning & Target Creation")

    preprocessor = DataPreprocessor(
        input_path="dummy_path",
        output_dir=output_dir,
        target_threshold=target_threshold,
        interval=interval,
        horizon=horizon,
    )

    X, y = preprocessor.run(df=raw_df)

    logger.info(f"✅ Preprocessed -> X: {X.shape}, y: {y.shape}")
    return X, y

# =============================================================================
# TASK 3 — Feature Engineering
# =============================================================================
@task(name="engineer-features", retries=1)
def engineer_features(X, output_dir: str):
    """Calculate technical indicators on X only. No target logic here."""
    logger = get_run_logger()
    logger.info("⚙️ Step 3: Feature Engineering")

    engineer = FeatureEngineer(output_dir=output_dir)
    X_features = engineer.fit_transform(X)

    logger.info(f"✅ Features calculated. Shape: {X_features.shape}")
    return X_features


@task(name="align-and-save", retries=1)
def align_and_save(X_features, y, output_dir: str):
    """Drop feature-warmup NaN rows, realign y, and persist the final X/y."""
    logger = get_run_logger()
    logger.info("🧹 Step 4: Aligning features/target and saving")

    X_final, y_final = align_features_and_target(X_features, y)

    os.makedirs(output_dir, exist_ok=True)
    x_path = os.path.join(output_dir, "X_final.parquet")
    y_path = os.path.join(output_dir, "y_final.parquet")

    X_final.to_parquet(x_path, index=False)
    y_final.to_frame(name="target").to_parquet(y_path, index=False)

    logger.info(f"✅ Saved X: {x_path} | y: {y_path}")
    return X_final, y_final, x_path, y_path


# =============================================================================
# TASK 5 — Model Training & Evaluation
# =============================================================================
@task(name="train-and-evaluate", retries=1)
def train_and_evaluate(X, y, val_ratio: float, test_ratio: float):
    """Runs the full model-selection pipeline as a single class call."""
    logger = get_run_logger()
    logger.info("🤖 Step 5: Model Training & Evaluation")

    pipeline = ModelTrainingPipeline(val_ratio=val_ratio, test_ratio=test_ratio)
    result = pipeline.run(X, y)

    logger.info(f"✅ Best model: {result['best_model_name']}")
    logger.info(f"   Test Accuracy: {result['test_metrics']['test_accuracy']:.4f}")
    logger.info(f"   Test Macro F1: {result['test_metrics']['test_f1_macro']:.4f}")

    return result



# =============================================================================
# FLOW — The main orchestrator
# =============================================================================
@flow(
    name="binance-ml-pipeline",
    description="Crypto price prediction pipeline — Prefect orchestrated",
    log_prints=True
)
def binance_ml_pipeline(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    start_year: int = 2023,
    start_month: int = 1,
    end_year: int = 2024,
    end_month: int = 1,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    features_dir: str = "data/features",
    target_threshold: float = 0.002,
    horizon: str = "30m",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Steps:
      1. Acquire data
      2. Preprocess + label
      3. Feature Eng.
      4. Align + save
      5. Train & Evaluate  <-- now implemented
    """
    logger = get_run_logger()

    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)

    logger.info("=" * 60)
    logger.info("BINANCE ML PIPELINE — PREFECT ORCHESTRATED")
    logger.info(f"  Symbol     : {symbol}")
    logger.info(f"  Interval   : {interval}")
    logger.info(f"  Date Range : {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
    logger.info("=" * 60)

    df_raw = acquire_data(
        start_date=start_date, end_date=end_date, symbol=symbol,
        interval=interval, output_dir=raw_dir
    )

    X, y = preprocess_data(
        raw_df=df_raw,
        output_dir=processed_dir,
        target_threshold=target_threshold,
        interval=interval,
        horizon=horizon,
    )

    X_features = engineer_features(X=X, output_dir=features_dir)

    X_final, y_final, x_path, y_path = align_and_save(
        X_features=X_features, y=y, output_dir=features_dir
    )

    training_result = train_and_evaluate(
        X=X_final, y=y_final,
        val_ratio=val_ratio, test_ratio=test_ratio,
    )

    logger.info(
        f"🏁 Pipeline complete! X: {x_path}, y: {y_path}, "
        f"best model: {training_result['best_model_name']}"
    )
    return x_path, y_path, training_result


if __name__ == "__main__":
    binance_ml_pipeline()