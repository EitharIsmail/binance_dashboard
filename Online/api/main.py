"""
FastAPI service exposing the trained Binance direction classifier.
Only BTCUSDT is currently servable; horizon is user-selectable among
whatever's been trained and promoted to Production.
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException

from api.model_loader import load_all_model_states, ModelState
from api.schema import CoinRequest, PredictionResponse

from api.config import *

import io
import zipfile

from fastapi.responses import JSONResponse

from typing import Optional



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_config = DataConfig()
target_config = TargetConfig()
flow_config = MLflowConfig()

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Longest rolling/EWM window in FeatureEngineer is ema_50 -- need at least
# that many candles so the MOST RECENT row has a real, non-NaN value for
# every feature. A buffer above 50 keeps this safe against off-by-ones.
MIN_CANDLES_FOR_FEATURES = 100

DIRECTION_NAMES = {-1: "Bear", 0: "Neutral", 1: "Bull"}  # original label space


class AppState:
    model_states: dict = {}   # {horizon: ModelState}, populated at startup

app_state = AppState()




@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up -- loading models from MLflow registry...")
    app_state.model_states = load_all_model_states(
        symbol=data_config.symbol,
        horizons=target_config.supported_horizons,
        stage="Production",
        tracking_uri= flow_config.tracking_uri,
    )
    if not app_state.model_states:
        logger.warning(
            "⚠️ No models loaded for any horizon -- /predict will return 503 "
            "until at least one (symbol, horizon) has been trained and deployed."
        )
    yield
    logger.info("🛑 Shutting down.")




app = FastAPI(title="Binance Direction Classifier API", 
              description="Predict baesd on the current price of a BITCOIN whether the price will significuntly go higher, lower or roughly unchanged after the chosen horizon",
    version="1.0.0",
    lifespan=lifespan,
    ROOT_PATH="/api")

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception(f"Unhandled error on {request.method} {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}"},
    )


def get_state(horizon: str) -> ModelState:
    state = app_state.model_states.get(horizon)

    if state is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No model currently deployed for horizon='{horizon}'. "
                f"Loaded horizons: {list(app_state.model_states.keys())}"
            ),
        )

    return state

@app.get("/")
def root():
    return {
        "message": "Binance Direction Classifier API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health")
def health():
    loaded = {
        horizon: {
            "model_name": s.model_name,
            "version": s.version,
            "model_alias": s.stage,
        }
        for horizon, s in app_state.model_states.items()
}
    return {"status" : "ok" if loaded else "no_models_loaded", "symbol" : data_config.symbol,
        "loaded_horizons":loaded}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: CoinRequest):
    try:
        s = get_state(request.horizon.value)

        if s.pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded",
            )

        raw_df = fetch_recent_klines(
            symbol=data_config.symbol,
            interval=data_config.interval,
        )

        features_df = s.feature_engineer.transform(raw_df)

        latest_row = features_df.dropna().tail(1)

        if latest_row.empty:
            raise HTTPException(
                status_code=503,
                detail="Not enough recent candle history to compute all features yet.",
            )

        prediction_mapped = s.pipeline.predict(latest_row)[0]

        prediction_original = INVERSE_LABEL_MAP[prediction_mapped]

        predicted_direction = DIRECTION_NAMES[prediction_original]

        return PredictionResponse(
            prediction=predicted_direction,
            model_name=s.model_name,
            model_version=str(s.version),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")



def fetch_recent_klines(
    symbol: str,
    interval: str,
    limit: int = MIN_CANDLES_FOR_FEATURES,
) -> pd.DataFrame:
    """
    Builds a recent candle window by combining two Binance sources:

      1. The most recent available Binance Vision monthly archive (same
         source data_acquisition.py trains on) -- provides a reliable bulk
         historical base, in case the live endpoint is rate-limited or a
         gap exists.
      2. Binance's LIVE REST API (/api/v3/klines) -- provides the current,
         still-forming candle and anything more recent than what the
         monthly archive (finalized only after month-end) could contain.

    Rows are deduplicated by open_time, with the LIVE source taking
    precedence on any overlap (it's the more current/authoritative of the
    two), then sorted chronologically and trimmed to the last `limit` rows.

    Schema matches data_acquisition.py EXACTLY (only open_time converted
    to datetime; close_time and all else stay in their raw numeric form)
    -- this must stay true, since the trained preprocessor was fit on
    that exact schema and will fail on any dtype drift (see the
    DTypePromotionError this replaced).
    """
    symbol = symbol.upper()

    # --- Source 1: most recent available monthly Vision archive ---
    archive_df = _fetch_latest_monthly_archive(symbol, interval)

    # --- Source 2: live REST API, current + very recent candles ---
    live_df = _fetch_live_klines(symbol, interval, limit=min(limit, 1000))

    # --- Merge: live wins on overlapping timestamps ---
    if archive_df is not None and not archive_df.empty:
        combined = pd.concat([archive_df, live_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["open_time"], keep="last")
    else:
        combined = live_df

    combined = combined.sort_values("open_time").reset_index(drop=True)
    return combined.tail(limit).reset_index(drop=True)


def _fetch_live_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Pulls the most recent `limit` candles directly from Binance's live
    REST API -- includes the current, still-forming candle."""
    url = "https://api.binance.com/api/v3/klines"
    response = requests.get(
        url,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10,
    )
    response.raise_for_status()
    raw = response.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
    ])
    return _standardize_schema(df)


def _fetch_latest_monthly_archive(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Downloads the most recent available Binance Vision monthly ZIP,
    walking backward month by month until one is found. Returns None
    (rather than raising) if no archive is reachable, so the live source
    alone can still serve the request."""
    current_date = datetime.utcnow()
    year, month = current_date.year, current_date.month

    for _ in range(6):  # don't walk back indefinitely if Vision is down
        month_str = f"{year:04d}-{month:02d}"
        filename = f"{symbol}-{interval}-{month_str}.zip"
        url = f"{data_config.BINANCE_VISION_BASE_URL}/{symbol}/{interval}/{filename}"

        try:
            response = requests.get(url, timeout=10)
        except requests.RequestException as e:
            logger.warning(f"Vision archive request failed: {e}")
            return None

        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_files = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_files:
                    return None
                with z.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file, header=None)
            df.columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
            ]
            return _standardize_schema(df)

        if response.status_code != 404:
            logger.warning(f"Vision archive returned {response.status_code} for {filename}")
            return None

        month -= 1
        if month == 0:
            month, year = 12, year - 1

    logger.warning(f"No Vision archive found for {symbol} {interval} in the last 6 months.")
    return None


def _standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the SAME dtype handling as data_acquisition.py: only
    open_time becomes datetime; everything else, including close_time,
    stays numeric. This must match training exactly."""
    timestamp = df["open_time"]
    if timestamp.iloc[0] > 1e14:
        df["open_time"] = pd.to_datetime(timestamp, unit="us")
    else:
        df["open_time"] = pd.to_datetime(timestamp, unit="ms")

    numeric_cols = ["open", "high", "low", "close", "volume", "taker_buy_base_asset_volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df.sort_values("open_time").reset_index(drop=True)