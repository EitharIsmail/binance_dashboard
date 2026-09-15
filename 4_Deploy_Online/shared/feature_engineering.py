import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handle feature creation and target generation for crypto time-series."""

    def __init__(
        self, 
        input_path: str, 
        output_dir: str = "data/features",
        target_threshold: float = 0.002, 
        interval: str = "15m",
        horizon: str = "30m" #time i want to predict at
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_path: Optional[str] = None
        
        # 1. Validate target_threshold/ target threshold determines how large the future price movement needs to be before calling it Bullish or Bearish
        if target_threshold <= 0:
            raise ValueError(f"Target threshold must be positive, got {target_threshold}")
        self.target_threshold = target_threshold
        
        # 2. Explicitly store the related time concepts
        self.interval = interval #Amount of time represented by one candle
        self.horizon = horizon   #How far into the future you want to predict
        
        # 3. Parse to minutes for mathematical validation
        interval_mins = self._parse_interval_to_minutes(interval)
        horizon_mins = self._parse_interval_to_minutes(horizon)
        
        # 4. FAIL-FAST VALIDATIONS: chech if the values of the interval and horizon are below 0 so it is invalid value
        if interval_mins <= 0:
            raise ValueError(f"Interval must be positive, got {interval} ({interval_mins} mins)")
        if horizon_mins <= 0:
            raise ValueError(f"Horizon must be positive, got {horizon} ({horizon_mins} mins)")
            
        if horizon_mins < interval_mins:
            raise ValueError(
                f"Forecast horizon ({horizon} = {horizon_mins}m) cannot be shorter than "
                f"the candle interval ({interval} = {interval_mins}m)."
            )

         #to check that the horizon value is double the interval value   
        if horizon_mins % interval_mins != 0:
            raise ValueError(
                f"Horizon ({horizon} = {horizon_mins}m) must be an exact multiple of "
                f"the interval ({interval} = {interval_mins}m)."
            )
            
        # 5. Calculate the final integer periods
        self.horizon_periods = horizon_mins // interval_mins
        
        # 6. Clean, multi-line logging/ this tell us that we created a FeatureEngineer object, and here is the time configuration it will use.
        logger.info(
            f"📏 Time configuration: "
            f"Interval={self.interval}, "
            f"Horizon={self.horizon}, "
            f"Periods={self.horizon_periods}"
        )

    @staticmethod
    def _parse_interval_to_minutes(interval_str: str) -> int:
        """
        Converts Binance interval strings (e.g., '15m', '1h', '1d', '1w') to minutes.
        Raises ValueError if the format is invalid.
        """
        if not isinstance(interval_str, str) or len(interval_str) < 2:
            raise ValueError(f"Invalid interval format: '{interval_str}'. Expected format like '15m', '1h', '1d'.")
            
        unit = interval_str[-1].lower()
        
        try:
            value = int(interval_str[:-1]) #Gives me everything except the last character.
        except ValueError:
            raise ValueError(f"Invalid numeric value in interval: '{interval_str}'.")

        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 24 * 60
        elif unit == 'w':
            return value * 7 * 24 * 60
        else:
            raise ValueError(f"Unsupported time unit '{unit}' in interval '{interval_str}'. Use 'm', 'h', 'd', or 'w'.")

    def load_data(self) -> pd.DataFrame:
        """Load preprocessed data from Parquet."""
        logger.info(f"📂 Loading preprocessed data from: {self.input_path}")
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Preprocessed file not found: {self.input_path}")
            
        df = pd.read_parquet(self.input_path)
        logger.info(f"   Loaded {len(df):,} rows")
        return df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators (the "lenses" for the model).
        All rolling windows strictly look backward to prevent leakage.
        """
        logger.info("📊 Calculating technical features...")
        
        # 1. Returns (Speed)
        df['return_1'] = df['close'].pct_change(1)
        df['return_3'] = df['close'].pct_change(3)

        # 2. Volatility (Chaos) - 10-period rolling std of 1-step return
        df['volatility_10'] = df['return_1'].rolling(window=10).std()

        # 3. Trend (Direction) - Exponential Moving Averages
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        # Distance from EMA (helps model see if price is overextended)
        df['dist_from_ema_20'] = (df['close'] - df['ema_20']) / df['ema_20']

        # 4. Momentum (RSI) - Pure Pandas implementation of 14-period RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # 5. Volume (Conviction)
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        logger.info(f"   ✅ Calculated {len(df.columns) - 7} new features.")
        return df

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the prediction target: Future direction.
        We look 'horizon_periods' ahead. If the move is > threshold, it's Bull/Bear.
        """
        logger.info(f"🎯 Creating target (Threshold: {self.target_threshold}, Horizon: {self.horizon} = {self.horizon_periods} periods)...")
        
        # Calculate future return using the dynamically calculated horizon_periods
        df['future_close'] = df['close'].shift(-self.horizon_periods)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']
        
        # Map to classes: 1 (Bullish), -1 (Bearish), 0 (Neutral)
        def assign_target(ret):
            if pd.isna(ret): return np.nan
            if ret >= self.target_threshold: return 1   # Bullish
            elif ret <= -self.target_threshold: return -1 # Bearish
            else: return 0                              # Neutral

        df['target'] = df['future_return'].apply(assign_target)
        
        # Log distribution
        target_counts = df['target'].value_counts().to_dict()
        logger.info(f"   Target distribution (before dropping NaNs): {target_counts}")
        
        return df

    def drop_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with NaN values from rolling windows and future shifts."""
        logger.info("🧹 Dropping invalid rows (NaNs from rolling windows and future shifts)...")
        
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped = initial_rows - len(df)
        
        logger.info(f"   Dropped {dropped} rows. Final shape: {len(df):,} rows")
        return df

    def run(self) -> pd.DataFrame:
        """Execute full feature engineering pipeline."""
        logger.info("⚙️ Starting Feature Engineering Pipeline")
        
        df = self.load_data()
        df = self.calculate_features(df)
        df = self.create_target(df)
        df = self.drop_invalid_rows(df)
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = os.path.join(self.output_dir, f"engineered_{Path(self.input_path).stem}.parquet")
        df.to_parquet(self.output_path, index=False)
        
        logger.info(f"✅ Feature engineering complete. Saved to: {self.output_path}")
        return df