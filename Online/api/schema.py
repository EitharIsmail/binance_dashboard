from enum import Enum

from pydantic import BaseModel, Field


class Horizon(str, Enum):
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


class CoinRequest(BaseModel):
    horizon: Horizon = Field(
        default=Horizon.MIN_30,
        description="Prediction horizon",
    )


class PredictionResponse(BaseModel):
    prediction: str = Field(
        description="Predicted market direction"
    )
    model_name: str = Field(
        description="Name of the MLflow registered model"
    )
    model_version: str = Field(
        description="Version of the MLflow registered model"
    )
