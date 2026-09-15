import pandas as pd
import requests
import io
import zipfile
import os
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Goal : Getting the raw Binance data and saving it
class BinanceDataAcquisition:
    """Handle data acquisition from Binance Vision."""

    def __init__(self, symbol: str, interval: str, output_dir: str = "data/raw"):
        self.symbol = symbol
        self.interval = interval
        self.output_dir = output_dir
        self.base_url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/"
        self.file_path: Optional[str] = None

        #To prepare the files i need to download
    def generate_monthly_urls(self, start_date: datetime, end_date: datetime) -> list[dict]:
        """
        Generate a list of Binance Vision URLs for the specified date range.
        Retries are handled by Prefect @task in flow.py.
        """
        logger.info(f"Generating URLs for {self.symbol} ({self.interval}) from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
        
        # Use 'MS' (Month Start) to prevent the 30-day drift bug
        months = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        file_list = []
        for date in months:
            year = date.year
            month = str(date.month).zfill(2)
            filename = f"{self.symbol}-{self.interval}-{year}-{month}.zip"
            url = self.base_url + filename
            file_list.append({"filename": filename, "url": url})
            
        return file_list

        #To download specific file and turn it into DF
    def download_and_extract_zip(self, filename: str, url: str) -> pd.DataFrame:
        """
        Download a single monthly ZIP file and extract the CSV into a DataFrame.
        Retries are handled by Prefect @task in flow.py.
        """
        logger.debug(f"Downloading {filename}...")
        
        # Added timeout=30 to prevent hanging (handled gracefully by Prefect retries)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            csv_filename = zip_file.namelist()[0]
            with zip_file.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file, header=None)
                
        df.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        return df

        #Produce proper datetime values 
    def process_timestamps_and_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix the microsecond/millisecond timestamp bug and cast types.
        """
        # Fix the > 1e14 microsecond bug, it convert the value into proper Pandas datetime
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        first_valid_ts = df['open_time'].dropna().iloc[0]
        
        # Fix the > 1e14 microsecond bug
        if first_valid_ts > 1e14:
            df['open_time'] = pd.to_datetime(df['open_time'], unit='us') # us means microseconds
        else:
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms') # ms means milliseconds
            
        # Cast numeric columns, convert the values to floats to avoid being treated as string
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        return df

    def combine_and_clean_dataframes(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenates monthly dataframes, sorts chronologically, and drops duplicates."""
        logger.info("Combining and cleaning dataframes...")
        df_raw = pd.concat(dfs, ignore_index=True)
        df_raw = df_raw.sort_values('open_time').drop_duplicates(subset=['open_time']).reset_index(drop=True)
        return df_raw

    def run(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Execute full data acquisition: generate URLs → download → process → combine.
        """
        logger.info("📥 Starting Binance Data Acquisition")
        
        # 1. Prepare files
        file_list = self.generate_monthly_urls(start_date, end_date)
        
        # 2. Download and process each month
        monthly_dfs = []
        for file_info in file_list:
            df = self.download_and_extract_zip(file_info["filename"], file_info["url"])
            df_clean = self.process_timestamps_and_types(df)
            monthly_dfs.append(df_clean)
            
        # 3. Combine into one clean chronological DataFrame
        final_df = self.combine_and_clean_dataframes(monthly_dfs)
        
        # 4. Save to disk for reproducibility
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_path = os.path.join(self.output_dir, f"{self.symbol}_{self.interval}_raw.parquet")
        final_df.to_parquet(self.file_path, index=False)
        
        logger.info(f"✅ Data acquisition complete. Saved to: {self.file_path}")
        logger.info(f"   Total rows: {len(final_df):,}, Columns: {final_df.shape[1]}")
        
        return final_df