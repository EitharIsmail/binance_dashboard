import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import mlflow
from mlflow import MlflowClient

from shared.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    """Everything needed to serve predictions for one (symbol, horizon)."""
    symbol: str
    horizon: str
    model_name: str
    stage: str                  # "Production" / "Staging" -- reported as "model_alias" in API responses
    pipeline: object             # fitted sklearn Pipeline: preprocessor + classifier, one artifact
    feature_engineer: FeatureEngineer
    version: str
    run_id: str
    val_f1: Optional[float] = None
    test_f1: Optional[float] = None


def _registered_model_name(symbol: str, horizon: str) -> str:
    """Must exactly match the naming convention ModelTrainingPipeline.run()
    uses in mlflow.sklearn.log_model(..., registered_model_name=...)."""
    return f"{symbol}_{horizon}_classifier"


def load_pipeline(symbol: str, horizon: str, stage: str, tracking_uri: str):
    """Loads the bundled preprocessor+classifier Pipeline from the MLflow
    Model Registry at the given stage."""
    mlflow.set_tracking_uri(tracking_uri)
    model_name = _registered_model_name(symbol, horizon)
    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"Loading pipeline: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def load_model_metadata(symbol: str, horizon: str, stage: str, tracking_uri: str) -> dict:
    """Fetches registry + run metadata (version, run_id, logged metrics)
    for whichever version currently sits at `stage`."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model_name = _registered_model_name(symbol, horizon)

    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise ValueError(
            f"No model version found for '{model_name}' at stage '{stage}'. "
            f"Has training + deployment run for this (symbol, horizon) yet?"
        )
    version_info = versions[0]
    run = client.get_run(version_info.run_id)
    metrics = run.data.metrics

    return {
        "version": version_info.version,
        "run_id": version_info.run_id,
        "val_f1": metrics.get("val_f1_macro"),
        "test_f1": metrics.get("test_f1_macro"),
    }


def get_feature_engineer() -> FeatureEngineer:
    """FeatureEngineer has no fitted state -- a fresh instance is always
    correct, so this is a constructor, not a real 'load' from storage."""
    return FeatureEngineer()


def load_model_state(
    symbol: str,
    horizon: str,
    stage: str,
    tracking_uri: str
) -> ModelState:

    model_name = _registered_model_name(symbol, horizon)

    pipeline = load_pipeline(
        symbol,
        horizon,
        stage,
        tracking_uri,
    )

    metadata = load_model_metadata(
        symbol,
        horizon,
        stage,
        tracking_uri,
    )

    feature_engineer = get_feature_engineer()

    return ModelState(
        symbol=symbol,
        horizon=horizon,
        model_name=model_name,
        stage=stage,
        pipeline=pipeline,
        feature_engineer=feature_engineer,
        version=str(metadata["version"]),
        run_id=metadata["run_id"],
        val_f1=metadata["val_f1"],
        test_f1=metadata["test_f1"],
    )


def load_all_model_states(
    symbol: str, horizons: List[str], stage: str, tracking_uri: str
) -> Dict[str, ModelState]:
    """Loads one ModelState per supported horizon, so the API serves every
    horizon without re-hitting MLflow on every request. Horizons with no
    registered model yet are skipped with a warning -- lets the API come
    up even if not every horizon has been trained yet."""
    states: Dict[str, ModelState] = {}
    for horizon in horizons:
        try:
            states[horizon] = load_model_state(symbol, horizon, stage, tracking_uri)
            logger.info(
                f"✅ Loaded {symbol} {horizon} model "
                f"(v{states[horizon].version}, stage={stage})"
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not load {symbol} {horizon} model: {e}")
    return states