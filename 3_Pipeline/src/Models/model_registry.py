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
    Minimal stand-in for a project-wide config.config.MLflowConfig -- this
    project doesn't have a config module yet, so this lives here for now.
    If/when a real config module exists, import from there instead and
    delete this class.
    """
    model_name: str = "btc_direction_classifier"
    staging_stage: str = "Staging"
    production_stage: str = "Production"
    archived_stage: str = "Archived"


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

        Assumes the model (and its preprocessing -- bundle preprocessor +
        classifier into a single sklearn Pipeline before logging, since a
        registered model version is one opaque artifact, not a pair of
        separately-tracked files) was already logged inside this run via
        mlflow.sklearn.log_model(pipeline, artifact_path=artifact_path).

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