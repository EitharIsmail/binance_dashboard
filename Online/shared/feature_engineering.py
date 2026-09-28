import pandas as pd
import os
import logging

from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, output_dir: str = "data/features"):
        self.output_dir = output_dir

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        logger.info("📊 Calculating technical features...")

        df['return_1'] = df['close'].pct_change(1)
        df['return_3'] = df['close'].pct_change(3)
        df['volatility_10'] = df['return_1'].rolling(window=10).std()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['dist_from_ema_20'] = (df['close'] - df['ema_20']) / df['ema_20']

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        logger.info(f"   ✅ Feature calculation complete. Shape: {df.shape}")
        return df


def align_features_and_target(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop rows where feature engineering introduced NaNs (rolling-window
    warmup, e.g. the first ~50 rows for ema_50) and realign y to match.

    Must run AFTER FeatureEngineer.transform() -- these NaNs don't exist
    until indicators are calculated, which is why this can't happen inside
    DataPreprocessor.
    """
    valid_idx = X.dropna().index
    X_aligned = X.loc[valid_idx].reset_index(drop=True)
    y_aligned = y.loc[valid_idx].reset_index(drop=True)

    dropped = len(X) - len(X_aligned)
    logger.info(f"🧹 Aligning X/y: dropped {dropped} warmup rows. Final shape: {X_aligned.shape}")
    return X_aligned, y_aligned