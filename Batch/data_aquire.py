import requests
import pandas as pd

from config import BINANCE_URL

def fetch_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    start_ms = int(
        pd.Timestamp(start_date, tz="UTC").timestamp() * 1000
    )

    end_ms = int(
        (
            pd.Timestamp(end_date, tz="UTC")
            + pd.Timedelta(days=1)
        ).timestamp() * 1000
    )

    rows = []
    current_start = start_ms

    while current_start < end_ms:

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }

        response = requests.get(
            BINANCE_URL,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        rows.extend(batch)

        current_start = batch[-1][0] + 1

        if len(batch) < 1000:
            break

    df = _klines_to_dataframe(rows)

    # Keep only candles before the exclusive end boundary 
    end_timestamp = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) 
    df = df[df["open_time"] < end_timestamp]

    if df.empty:
        raise ValueError(
            f"No Binance data found for {symbol} "
            f"from {start_date} to {end_date}."
        )

    return df


def _klines_to_dataframe(rows) -> pd.DataFrame:

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])

    return df
