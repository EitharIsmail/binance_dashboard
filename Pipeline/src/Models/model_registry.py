"""
Model registry module for the Binance ML Pipeline.
Handles MLflow model registry operations and stage transitions.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)


@dataclass
class MLflowConfig:
    """
    Registry config for one (symbol, horizon) model. model_name is
    computed from the two -- e.g. symbol="BTCUSDT", horizon="30m" ->
    model_name="BTCUSDT_30m_classifier" -- so each coin/horizon
    combination gets its own independent registry entry and version
    history rather than colliding under one shared name.
    """
    symbol: str
    horizon: str
    staging_stage: str = "Staging"
    production_stage: str = "Production"
    archived_stage: str = "Archived"

    @property
    def model_name(self) -> str:
        return f"{self.symbol}_{self.horizon}_classifier"


class ModelRegistry:
    """Handle MLflow model registry operations."""

    def __init__(self, mlflow_config: MLflowConfig, client: MlflowClient):
        """
        Initialize model registry.

        Args:
            mlflow_config: MLflow configuration
            client: MLflow client
        """
        self.mlflow_config = mlflow_config
        self.client = client
        self.model_name = mlflow_config.model_name

    def get_all_versions(self) -> List[ModelVersion]:
        """
        Get all versions of registered model.

        Returns:
            List of model versions
        """
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            return sorted(versions, key=lambda x: int(x.version))
        except Exception as e:
            logger.warning(f"Could not retrieve model versions: {str(e)}")
            return []

    def find_version_by_run_id(self, run_id: str) -> Optional[str]:
        """
        Find model version by run ID.

        Args:
            run_id: MLflow run ID

        Returns:
            Model version or None
        """
        versions = self.get_all_versions()
        for v in versions:
            if v.run_id == run_id:
                return v.version
        return None

    def register_model(self, run_id: str, artifact_path: str = "model") -> Optional[str]:
        """
        Register a logged model artifact as a new model version.

        Note: if the model was logged via mlflow.sklearn.log_model(...,
        registered_model_name=...) (as ModelTrainingPipeline.run() does),
        a version already exists for that run and this call is usually
        redundant -- but it's kept here as a standalone, explicit
        registration path for any caller that logs a model WITHOUT
        registered_model_name and wants to register it after the fact.

        Args:
            run_id: MLflow run ID the model was logged under
            artifact_path: the artifact_path used in log_model()

        Returns:
            The new model version, or None if registration failed
        """
        logger.info(f"📦 Registering model from run {run_id[:8]}...")
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            result = self.client.create_model_version(
                name=self.model_name,
                source=model_uri,
                run_id=run_id,
            )
            logger.info(f"   ✓ Registered as version {result.version}")
            return result.version
        except Exception as e:
            logger.error(f"   ❌ Failed to register model: {str(e)}")
            return None

    def transition_to_staging(
        self,
        run_id: str,
        description: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Transition model to staging.

        Args:
            run_id: MLflow run ID
            description: Model description
            tags: Optional tags to add

        Returns:
            Model version or None
        """
        logger.info("🏛️  Transitioning model to Staging...")

        version = self.find_version_by_run_id(run_id)

        if not version:
            logger.error(f"   ❌ Model version not found for run ID: {run_id[:8]}")
            return None

        try:
            # Transition to staging
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=self.mlflow_config.staging_stage
            )
            logger.info(f"   ✓ Version {version} → Staging")

            # Update description
            self.client.update_model_version(
                name=self.model_name,
                version=version,
                description=description
            )

            # Add tags
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        self.model_name,
                        version,
                        key,
                        str(value)
                    )
                logger.info(f"   ✓ Added {len(tags)} tags")

            return version

        except Exception as e:
            logger.error(f"   ❌ Failed to transition to staging: {str(e)}")
            return None

    def transition_to_production(self, version: str) -> bool:
        """
        Transition model to production.

        Args:
            version: Model version

        Returns:
            True if successful
        """
        logger.info("🚀 Promoting model to Production...")

        try:
            # Archive current production models
            prod_versions = self.client.get_latest_versions(
                self.model_name,
                stages=[self.mlflow_config.production_stage]
            )

            for prod_v in prod_versions:
                if prod_v.version != version:
                    self.client.transition_model_version_stage(
                        name=self.model_name,
                        version=prod_v.version,
                        stage=self.mlflow_config.archived_stage
                    )
                    logger.info(f"   ✓ Version {prod_v.version} → Archived")

            # Promote to production
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=self.mlflow_config.production_stage
            )
            logger.info(f"   ✓ Version {version} → Production")

            return True

        except Exception as e:
            logger.error(f"   ❌ Failed to promote to production: {str(e)}")
            return False

    def get_model_by_stage(self, stage: str) -> Optional[ModelVersion]:
        """
        Get model by stage.

        Args:
            stage: Model stage (Staging, Production, Archived)

        Returns:
            Model version or None
        """
        try:
            versions = self.client.get_latest_versions(self.model_name, stages=[stage])
            if versions:
                return versions[0]
        except Exception as e:
            logger.warning(f"Could not retrieve model for stage {stage}: {str(e)}")
        return None

    def print_registry_status(self):
        """Print current model registry status."""
        logger.info("📊 Model Registry Status:")
        logger.info(f"   Model: {self.model_name}")

        all_versions = self.get_all_versions()
        logger.info(f"   Total versions: {len(all_versions)}")

        for stage in ["None", "Staging", "Production", "Archived"]:
            version = self.get_model_by_stage(stage)
            if version:
                logger.info(
                    f"   • {stage:<12}: v{version.version} (Run: {version.run_id[:8]})"
                )

    def get_deployment_uri(self, stage: str = "Production") -> str:
        """
        Get deployment URI for model.

        Args:
            stage: Model stage

        Returns:
            MLflow model URI
        """
        return f"models:/{self.model_name}/{stage}"


# =============================================================================
# Deployment gate — registers a trained run and promotes it if it's better
# than the current production incumbent (or if there is no incumbent yet).
# =============================================================================
def deploy_if_better(
    run_id: str,
    val_f1: float,
    test_f1: float,
    symbol: str,
    horizon: str,
    min_f1_improvement: float = 0.01,
) -> str:
    """
    Registers the model from `run_id`, promotes it to Staging always, and
    promotes it further to Production only if it beats the CURRENT
    production model's test F1 by at least `min_f1_improvement` -- or if
    there is no production model yet.

    Note: since ModelTrainingPipeline.run() already logs the model with
    registered_model_name set, a version typically already exists for this
    run_id by the time this function is called. register_model() here is
    effectively a defensive no-op / re-registration attempt in that case
    (MLflow will just create an additional version pointing at the same
    artifact if called again) -- kept for robustness against callers that
    log without registered_model_name.

    Returns a short status string describing what happened.
    """
    client = MlflowClient()
    config = MLflowConfig(symbol=symbol, horizon=horizon)
    registry = ModelRegistry(mlflow_config=config, client=client)

    version = registry.find_version_by_run_id(run_id)
    if version is None:
        version = registry.register_model(run_id=run_id, artifact_path="model")
    if version is None:
        return "registration_failed"

    registry.transition_to_staging(
        run_id=run_id,
        description=f"{symbol} {horizon} classifier, val_f1={val_f1:.4f}, test_f1={test_f1:.4f}",
        tags={"val_f1": val_f1, "test_f1": test_f1},
    )

    current_prod = registry.get_model_by_stage(config.production_stage)
    if current_prod is None:
        registry.transition_to_production(version)
        logger.info(f"🚀 No prior production model -- v{version} promoted directly.")
        return "promoted_first_production"

    # Compare against the incumbent's logged test_f1_macro metric.
    prod_run = client.get_run(current_prod.run_id)
    prod_test_f1 = prod_run.data.metrics.get("test_f1_macro", 0.0)

    if test_f1 >= prod_test_f1 + min_f1_improvement:
        registry.transition_to_production(version)
        logger.info(
            f"🚀 v{version} promoted: test_f1={test_f1:.4f} beats "
            f"incumbent's {prod_test_f1:.4f} by >= {min_f1_improvement}."
        )
        return "promoted_new_champion"

    logger.info(
        f"⏸️  v{version} stays in Staging: test_f1={test_f1:.4f} doesn't beat "
        f"incumbent's {prod_test_f1:.4f} by >= {min_f1_improvement}."
    )
    return "kept_in_staging"