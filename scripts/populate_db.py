import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pytz

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.connectors.mt5_connector import MT5Connector
from data.infrastructure.database import DatabaseManager
from src.features.features import FeatureEngineer

def populate_data(days=365):
    """
    Populates the database with historical data.
    Fetches data in chunks to handle deep history.
    """
    print(f"--- Starting Deep Data Population ({days} Days) ---")
    
    connector = MT5Connector()
    if not connector.initialize():
        print("Failed to connect to MT5")
        return

    db = DatabaseManager()
    engineer = FeatureEngineer()
    
    symbols = ["EURUSD", "GBPUSD"]
    
    for symbol in symbols:
        print(f"\n[Processing {symbol}]")
        
        # Define Time Range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Chunking Strategy (30 days per chunk)
        current_start = start_date
        total_rows = 0
        
        all_data = []

        while current_start < end_date:
            current_end = current_start + timedelta(days=30)
            if current_end > end_date:
                current_end = end_date
                
            print(f"  Fetching: {current_start.date()} -> {current_end.date()} ...", end=" ")
            
            # Fetch using Range
            df_chunk = connector.get_candles_range(
                symbol=symbol,
                timeframe_str="M1",
                start_date=current_start,
                end_date=current_end
            )
            
            if df_chunk is not None and not df_chunk.empty:
                # Filter to exact range to avoid overlaps if we overshoot
                mask = (df_chunk['time'] >= current_start) & (df_chunk['time'] < current_end)
                df_chunk = df_chunk.loc[mask]
                
                rows = len(df_chunk)
                print(f"Got {rows} candles.")
                
                if rows > 0:
                    all_data.append(df_chunk)
                    total_rows += rows
            else:
                print("No data returned.")

            current_start = current_end
            
        # Combine
        if all_data:
            full_df = pd.concat(all_data).drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
            print(f"  Total Raw Data: {len(full_df)} rows")
            
            # Validation
            if len(full_df) < 100000:
                print("  WARNING: Data < 100k rows. Check MT5 Max Bars setting!")
            
            # Feature Engineering
            print("  Calculating Features...")
            df_features = engineer.calculate_features(full_df)
            
            # Save
            print("  Saving to Database...")
            db.save_candles(symbol, df_features)
            
        else:
            print(f"  Failed to fetch any data for {symbol}")

    connector.shutdown()
    print("\n--- Population Complete ---")

if __name__ == "__main__":
    populate_data(days=365)
