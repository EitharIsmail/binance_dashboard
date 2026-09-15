import logging

import numpy as np
import pandas as pd

from src.Models.model_training import INVERSE_LABEL_MAP
from src.Models.model_registry import load_best_model_pointer, load_preprocessor, load_model

logger = logging.getLogger(__name__)


class Predictor:
    """
    Loads whatever model the registry currently marks as "best" for a given
    output_dir, and exposes predict()/predict_proba() for new feature rows.

    Usage:
        predictor = Predictor(model_dir="models/trained")
        predictor.predict(X_new)  # -> array of -1 / 0 / 1
    """

    def __init__(self, model_dir: str = "models/trained"):
        self.model_dir = model_dir
        self.pointer = load_best_model_pointer(model_dir)
        self.model_name = self.pointer["model_name"]
        self.preprocessor = load_preprocessor(model_dir)
        self.model = load_model(self.model_name, model_dir)
        logger.info(
            f"Loaded '{self.model_name}' for inference "
            f"(val macro F1 at promotion: {self.pointer['validation_macro_f1']})"
        )

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        X_new must have the same feature columns the model was trained on
        (raw, unscaled -- the registered preprocessor handles imputation +
        scaling internally). Returns predictions in the original label
        space (-1 Bear, 0 Neutral, 1 Bull), not the 0/1/2 training space.
        """
        X_proc = self.preprocessor.transform(X_new)
        preds_mapped = self.model.predict(X_proc)
        return np.array([INVERSE_LABEL_MAP[p] for p in preds_mapped])

    def predict_proba(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Returns class probabilities in training label order (0, 1, 2 ->
        Bear, Neutral, Bull). Not every model in the bench supports this
        (e.g. AdaBoost/GradientBoosting do; check model.predict_proba
        exists before relying on it in a caller).
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"'{self.model_name}' does not support predict_proba()"
            )
        X_proc = self.preprocessor.transform(X_new)
        return self.model.predict_proba(X_proc)


def predict(X_new: pd.DataFrame, model_dir: str = "models/trained") -> np.ndarray:
    """Convenience one-shot wrapper around Predictor for scripts/notebooks
    that don't need to keep the loaded model around."""
    return Predictor(model_dir=model_dir).predict(X_new)