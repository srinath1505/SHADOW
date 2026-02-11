import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.connectors.mt5_connector import MT5Connector
from src.features.features import FeatureEngineer
from data.infrastructure.database import DatabaseManager
from config import settings

def main():
    print("Starting Data Population...")
    
    # Initialize components
    connector = MT5Connector()
    if not connector.initialize():
        print("Failed to initialize MT5 Connector. Make sure MT5 terminal is running.")
        return

    # Attempt login (optional if already logged in)
    connector.login()

    fe = FeatureEngineer()
    db = DatabaseManager()

    # Define date range (last 12 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    symbols = settings.SYMBOLS
    timeframes = ["M1"] # Focus on M1 as per requirements

    for symbol in symbols:
        print(f"Processing {symbol}...")
        for tf in timeframes:
            print(f"  Fetching {tf} data from {start_date.date()} to {end_date.date()}...")
            
            # Fetch data (might need batching if API limits, but copy_rates_from usually handles large requests okay or needs loop)
            # MT5 copy_rates_from returns numpy array, converted to DF in connector
            # copy_rates_from takes start date.
            # We urge caution with 12 months of M1 data (millions of rows). 
            # Let's fetch in chunks of 30 days to be safe and report progress.
            
            current_start = start_date
            while current_start < end_date:
                current_end = current_start + timedelta(days=30)
                if current_end > end_date:
                    current_end = end_date
                
                # print(f"    Chunk: {current_start.date()} to {current_end.date()}")
                
                # Valid call: get_historical_candles(symbol, timeframe_str, num_candles, start_date)
                # But our connector uses num_candles OR start_date logic.
                # Let's use start_date and calculate num_candles roughly or just use copy_rates_range if available?
                # The connector implementation uses copy_rates_from(start_date, num_candles). 
                # We need num_candles. 30 days * 1440 mins = 43200 candles.
                # Let's assume 45000 to cover weekends/gaps.
                
                # Actually, better to use copy_rates_range in connector or just request a large number from start.
                # Let's stick to the connector's interface: get_historical_candles(symbol, tf, num_candles, start_date)
                # This fetches num_candles FROM start_date going forward? 
                # mt5.copy_rates_from(symbol, timeframe, date_from, count) -> yes, from date_from forward.
                
                chunk_candles = 50000 # ~35 days of minutes
                
                df = connector.get_historical_candles(symbol, tf, num_candles=chunk_candles, start_date=current_start)
                
                if df is not None and not df.empty:
                    # Calculate features
                    # print(f"    Calculating features for {len(df)} candles...")
                    df_features = fe.calculate_features(df)
                    
                    # Save to DB
                    # print(f"    Saving to DB...")
                    db.save_candles(symbol, df_features, timeframe=tf)
                    
                    # Update start for next chunk based on last timestamp in df?
                    # Or just purely time based? Purely time based loop is safer to ensure coverage.
                    # But copy_rates_from is exact.
                    pass
                else:
                    print(f"    No data returned for chunk starting {current_start}")
                
                current_start = current_end
                
    print("Data Population Completed.")
    connector.shutdown()

if __name__ == "__main__":
    main()
