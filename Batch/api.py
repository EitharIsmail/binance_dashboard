import logging

from fastapi import FastAPI, HTTPException
import uvicorn

from data_aquire import fetch_klines
from feature_engineering import FeatureEngineer
from analysis import run_analysis
from config import API_HOST, API_PORT


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Binance Batch Analysis API",
    description=(
        "On-demand historical Binance market "
        "data analysis API."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "binance-batch-analysis",
    }


# --------------------------------------------------
# Analysis endpoint
# --------------------------------------------------

@app.get("/analysis")
def analysis(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
):

    logger.info(
        "🔍 Analysis request received: "
        f"symbol={symbol}, "
        f"interval={interval}, "
        f"start_date={start_date}, "
        f"end_date={end_date}"
    )

    try:

        # --------------------------------------------------
        # 1. Acquire Binance data
        # --------------------------------------------------

        logger.info("📥 Fetching Binance market data...")

        df = fetch_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(
            f"✅ Binance data acquired. Shape: {df.shape}"
        )

        # --------------------------------------------------
        # 2. Feature engineering
        # --------------------------------------------------

        logger.info(
            "⚙️ Running feature engineering..."
        )

        feature_engineer = FeatureEngineer()

        df_features = feature_engineer.transform(df)

        # --------------------------------------------------
        # 3. Analysis
        # --------------------------------------------------

        logger.info(
            "📊 Running market analysis..."
        )

        result = run_analysis(
            df_features,
            symbol,
        )

        logger.info(
            "✅ Analysis completed successfully."
        )

        return result

    except ValueError as e:

        logger.warning(
            f"⚠️ Invalid analysis request: {e}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        logger.exception(
            "❌ Analysis failed."
        )

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


# --------------------------------------------------
# Run locally
# --------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
    )