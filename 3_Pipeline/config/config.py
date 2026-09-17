"""
Binance ML Pipeline Configuration
===================================
Centralizes all pipeline settings so main.py/flow.py don't hardcode
values, and so the future dashboard can import the same source of
truth for validating user-selectable options (symbol, horizon).

Prefect handles retries via @task(retries=N, retry_delay_seconds=N),
so no separate RetryConfig is needed here.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DataConfig:
    """Data acquisition configuration."""

    symbol: str = "BTCUSDT"
    interval: str = "15m"

    start_year: int = 2023
    start_month: int = 1
    end_year: int = 2024
    end_month: int = 1

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    features_dir: str = "data/features"

    # Coins the dashboard is allowed to offer. Each one requires its own
    # trained model (see the multi-coin explanation below) -- this list is
    # the single source of truth for both training runs and dashboard UI.
    supported_symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    ])


@dataclass
class TargetConfig:
    """Target/label configuration -- what the model is trained to predict."""

    target_threshold: float = 0.002

    # Horizons the dashboard is allowed to offer. Each one requires its own
    # trained model, since the target label is horizon-dependent (see the
    # horizon explanation below) -- this list drives both training loops
    # and dashboard UI.
    supported_horizons: List[str] = field(default_factory=lambda: [
        "15m", "30m", "1h", "4h", "1d",
    ])
    default_horizon: str = "30m"


@dataclass
class ModelConfig:
    """Model training / evaluation configuration."""

    random_state: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    n_jobs: int = -1

    models_to_train: List[str] = field(default_factory=lambda: [
        "Logistic Regression", "Random Forest", "Extra Trees",
        "XGBoost", "LightGBM", "CatBoost",
        "Gradient Boosting", "AdaBoost", "SVC (RBF)", "KNN",
    ])


@dataclass
class MLflowConfig:
    """MLflow tracking and registry configuration."""

    experiment_name: str = "binance-ml-pipeline"
    tracking_uri: str = "http://127.0.0.1:5000"

    project_tag: str = "binance_ml"
    team_tag: str = "crypto_prediction"
    framework_tag: str = "scikit-learn"

    staging_stage: str = "Staging"
    production_stage: str = "Production"
    archived_stage: str = "Archived"

    # Naming convention for registered models, so the dashboard can look
    # up the right model given a user's (symbol, horizon) selection.
    @staticmethod
    def registered_model_name(symbol: str, horizon: str) -> str:
        return f"{symbol}_{horizon}_classifier"


@dataclass
class PathConfig:
    """File paths configuration."""

    base_dir: str = os.path.abspath(".")
    model_dir: str = "models"
    log_dir: str = "logs"

    def __post_init__(self):
        for dir_path in [self.model_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class LogConfig:
    """Logging configuration."""

    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    log_file: str = "pipeline.log"


@dataclass
class Config:
    """Main configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data.__dict__,
            "target": self.target.__dict__,
            "model": self.model.__dict__,
            "mlflow": self.mlflow.__dict__,
            "paths": {k: v for k, v in self.paths.__dict__.items() if not k.startswith("_")},
            "log": self.log.__dict__,
        }


def load_config() -> Config:
    """Load configuration with optional environment variable overrides."""
    config = Config()

    if os.getenv("RANDOM_STATE"):
        config.model.random_state = int(os.getenv("RANDOM_STATE"))
    if os.getenv("MLFLOW_EXPERIMENT_NAME"):
        config.mlflow.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if os.getenv("SYMBOL"):
        config.data.symbol = os.getenv("SYMBOL")
    if os.getenv("HORIZON"):
        config.target.default_horizon = os.getenv("HORIZON")

    return config