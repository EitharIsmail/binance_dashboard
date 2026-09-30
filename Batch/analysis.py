import logging

import pandas as pd


logger = logging.getLogger(__name__)


def generate_market_summary(
    df: pd.DataFrame,
    symbol: str,
) -> dict:

    logger.info("📊 Generating market summary...")

    summary = {
        "symbol": symbol,
        "records": len(df),
        "start_time": df["open_time"].min().isoformat(),
        "end_time": df["open_time"].max().isoformat(),

        "latest_price": float(
            df["close"].iloc[-1]
        ),

        "highest_price": float(
            df["high"].max()
        ),

        "lowest_price": float(
            df["low"].min()
        ),

        "average_price": float(
            df["close"].mean()
        ),

        "total_volume": float(
            df["volume"].sum()
        ),

        "average_volume": float(
            df["volume"].mean()
        ),
    }

    logger.info("✅ Market summary generated.")

    return summary


def generate_return_analysis(df: pd.DataFrame) -> dict:

    logger.info("📈 Generating return analysis...")

    returns = df["return_1"].dropna()

    return {
        "mean": float(returns.mean()),
        "median": float(returns.median()),
        "std": float(returns.std()),
        "minimum": float(returns.min()),
        "maximum": float(returns.max()),
    }


def generate_volatility_analysis(df: pd.DataFrame) -> dict:

    logger.info("📉 Generating volatility analysis...")

    volatility = df["volatility_10"].dropna()

    return {
        "mean": float(volatility.mean()),
        "median": float(volatility.median()),
        "minimum": float(volatility.min()),
        "maximum": float(volatility.max()),
        "latest": float(volatility.iloc[-1]),
    }


def generate_rsi_analysis(df: pd.DataFrame) -> dict:

    logger.info("📊 Generating RSI analysis...")

    rsi = df["rsi_14"].dropna()

    latest_rsi = float(rsi.iloc[-1])

    if latest_rsi >= 70:
        condition = "overbought"
    elif latest_rsi <= 30:
        condition = "oversold"
    else:
        condition = "neutral"

    return {
        "latest": latest_rsi,
        "average": float(rsi.mean()),
        "minimum": float(rsi.min()),
        "maximum": float(rsi.max()),
        "condition": condition,
    }


def generate_trend_analysis(df: pd.DataFrame) -> dict:

    logger.info("📈 Generating trend analysis...")

    latest = df.iloc[-1]

    return {
        "price": float(latest["close"]),
        "ema_20": float(latest["ema_20"]),
        "ema_50": float(latest["ema_50"]),

        "distance_from_ema_20": float(
            latest["dist_from_ema_20"]
        ),

        "above_ema_20": bool(
            latest["close"] > latest["ema_20"]
        ),

        "above_ema_50": bool(
            latest["close"] > latest["ema_50"]
        ),
    }


def generate_price_series(df: pd.DataFrame) -> list:

    data = df[
        [
            "open_time",
            "close",
            "ema_20",
            "ema_50",
        ]
    ].dropna()

    return [
        {
            "time": row["open_time"].isoformat(),
            "close": float(row["close"]),
            "ema_20": float(row["ema_20"]),
            "ema_50": float(row["ema_50"]),
        }
        for _, row in data.iterrows()
    ]


def run_analysis(
    df: pd.DataFrame,
    symbol: str,
) -> dict:

    logger.info("🔍 Starting batch analysis...")

    result = {
        "market_summary": generate_market_summary(
            df,
            symbol,
        ),

        "returns": generate_return_analysis(df),

        "volatility": generate_volatility_analysis(df),

        "rsi": generate_rsi_analysis(df),

        "trend": generate_trend_analysis(df),

        "price_series": generate_price_series(df),
    }

    logger.info("✅ Batch analysis complete.")

    return result