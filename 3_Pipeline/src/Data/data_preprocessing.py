import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """ETL, validation, leakage-free cleaning, AND target creation for raw crypto data.

    Target creation lives here (not in feature engineering) because the label
    depends only on the raw `close` column -- it needs no engineered indicator.
    """

    def __init__(
        self,
        input_path: str,
        output_dir: str = "data/processed",
        target_threshold: float = 0.002,
        interval: str = "15m",
        horizon: str = "30m",
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_path: Optional[str] = None

        self.core_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'taker_buy_base_asset_volume'
        ]

        # --- Target configuration & validation (moved from FeatureEngineer) ---
        if target_threshold <= 0:
            raise ValueError(f"Target threshold must be positive, got {target_threshold}")
        self.target_threshold = target_threshold
        self.interval = interval
        self.horizon = horizon

        interval_mins = self._parse_interval_to_minutes(interval)
        horizon_mins = self._parse_interval_to_minutes(horizon)

        if interval_mins <= 0:
            raise ValueError(f"Interval must be positive, got {interval} ({interval_mins} mins)")
        if horizon_mins <= 0:
            raise ValueError(f"Horizon must be positive, got {horizon} ({horizon_mins} mins)")
        if horizon_mins < interval_mins:
            raise ValueError(
                f"Forecast horizon ({horizon} = {horizon_mins}m) cannot be shorter than "
                f"the candle interval ({interval} = {interval_mins}m)."
            )
        if horizon_mins % interval_mins != 0:
            raise ValueError(
                f"Horizon ({horizon} = {horizon_mins}m) must be an exact multiple of "
                f"the interval ({interval} = {interval_mins}m)."
            )

        self.horizon_periods = horizon_mins // interval_mins

        logger.info(
            f"📏 Time configuration: Interval={self.interval}, "
            f"Horizon={self.horizon}, Periods={self.horizon_periods}"
        )

    @staticmethod
    def _parse_interval_to_minutes(interval_str: str) -> int:
        """Converts Binance interval strings (e.g., '15m', '1h', '1d', '1w') to minutes."""
        if not isinstance(interval_str, str) or len(interval_str) < 2:
            raise ValueError(f"Invalid interval format: '{interval_str}'. Expected format like '15m', '1h', '1d'.")

        unit = interval_str[-1].lower()
        try:
            value = int(interval_str[:-1])
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
        """Load raw data from Parquet."""
        logger.info(f"📂 Loading raw data from: {self.input_path}")
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Raw data file not found: {self.input_path}")

        df = pd.read_parquet(self.input_path)
        logger.info(f"   Loaded {len(df):,} rows, {df.shape[1]} columns")
        return df

    def validate_and_cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types."""
        logger.info("🔍 Validating and casting data types...")

        if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
            logger.info("   ⚠️ open_time is not datetime. Applying robust timestamp detection...")
            df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
            first_valid_ts = df['open_time'].dropna().iloc[0]

            if first_valid_ts > 1e14:
                df['open_time'] = pd.to_datetime(df['open_time'], unit='us')
                logger.info("   Detected and converted Microsecond (us) timestamps.")
            else:
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                logger.info("   Detected and converted Millisecond (ms) timestamps.")

        for col in self.core_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def sort_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort chronologically and drop duplicate timestamps."""
        logger.info(" Sorting chronologically and dropping duplicates...")

        initial_rows = len(df)
        df = df.sort_values('open_time').reset_index(drop=True)
        df = df.drop_duplicates(subset=['open_time'], keep='first')

        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.warning(f"   ⚠️ Dropped {dropped} duplicate timestamps.")
        else:
            logger.info("   ✅ No duplicate timestamps found.")

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values WITHOUT data leakage (forward fill only)."""
        logger.info(" Handling missing values (No Leakage: Forward Fill)...")

        missing_before = df[self.core_cols].isnull().sum().sum()
        df[self.core_cols] = df[self.core_cols].ffill()
        df = df.dropna(subset=self.core_cols)

        missing_after = df[self.core_cols].isnull().sum().sum()
        logger.info(f"   Missing values before: {missing_before} -> after: {missing_after}")

        return df

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the forward-looking Bull/Neutral/Bear target using ONLY the
        raw `close` column. Safe to run before feature engineering since the
        label has no dependency on any engineered indicator.
        """
        logger.info(
            f"🎯 Creating target (Threshold: {self.target_threshold}, "
            f"Horizon: {self.horizon} = {self.horizon_periods} periods)..."
        )

        df['future_close'] = df['close'].shift(-self.horizon_periods)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']

        conditions = [
            df['future_return'] >= self.target_threshold,
            df['future_return'] <= -self.target_threshold,
        ]
        df['target'] = np.select(conditions, [1, -1], default=0)
        df.loc[df['future_return'].isna(), 'target'] = np.nan

        logger.info(
            f"   Target distribution (before dropping tail NaNs): "
            f"{df['target'].value_counts(dropna=False).to_dict()}"
        )
        return df

    def drop_unlabelable_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the tail rows where no future close exists (shift ran off the
        end of the series). This does NOT touch feature-warmup NaNs -- those
        don't exist yet at this stage; see align_features_and_target().
        """
        initial_rows = len(df)
        df = df.dropna(subset=['target']).reset_index(drop=True)
        logger.info(f"   Dropped {initial_rows - len(df)} unlabelable tail rows. Remaining: {len(df):,}")
        return df

    def run(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Execute full preprocessing pipeline: (load) → validate → clean → target → split.

        If `df` is provided, skips load_data() and uses it directly -- this
        lets the flow pass an in-memory DataFrame from a prior task without
        a redundant disk round-trip. If omitted, loads from self.input_path.

        Returns (X, y). X excludes target-derived columns.
        """
        logger.info("🔄 Starting Data Preprocessing Pipeline")

        if df is None:
            df = self.load_data()

        df = self.validate_and_cast_types(df)
        df = self.sort_and_deduplicate(df)
        df = self.handle_missing_values(df)
        df = self.create_target(df)
        df = self.drop_unlabelable_rows(df)

        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = os.path.join(self.output_dir, f"cleaned_{Path(self.input_path).stem}.parquet")
        df.to_parquet(self.output_path, index=False)
        logger.info(f"✅ Preprocessing complete. Saved to: {self.output_path}")

        y = df['target'].astype(int)
        X = df.drop(columns=['future_close', 'future_return', 'target'])

        logger.info(f"   Final shapes -> X: {X.shape}, y: {y.shape}")
        return X, y