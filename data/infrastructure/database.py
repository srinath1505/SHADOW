import pandas as pd
from sqlalchemy import create_engine, text
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import settings

class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or settings.DB_PATH
        # Create directory if it deals not exist
        if not self.db_path.parent.exists():
             self.db_path.parent.mkdir(parents=True, exist_ok=True)
             
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def save_candles(self, symbol: str, df: pd.DataFrame, timeframe="M1"):
        """
        Saves candles to the database. Replaces existing records for the same time range.
        Expects df to have 'time' column.
        """
        if df.empty:
            return

        table_name = f"{symbol}_{timeframe}"
        
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        min_time = df['time'].min().strftime('%Y-%m-%d %H:%M:%S')
        max_time = df['time'].max().strftime('%Y-%m-%d %H:%M:%S')

        # Fix Unsigned 64-bit integer issue for SQLite
        # Iterate over columns and cast uint64 to int64
        for col in df.columns:
            if df[col].dtype == 'uint64':
                df[col] = df[col].astype('int64')

        with self.engine.connect() as conn:
            # Delete overlapping data to avoid duplicates (basic upsert strategy)
            try:
                # Check if table exists first (or let delete fail gracefully? No, better to check)
                # For sqlite, just try delete, if table doesn't exist it might error.
                # simpler: just strictly append? No, we need to handle re-runs.
                
                # Delete existing range
                delete_query = text(f"DELETE FROM {table_name} WHERE time >= '{min_time}' AND time <= '{max_time}'")
                conn.execute(delete_query)
                conn.commit()
            except Exception as e:
                # Table likely doesn't exist yet
                # print(f"Table may not exist or error deleting: {e}")
                pass

        # Write new data
        try:
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            
            # Create Index for performance (if not exists)
            index_name = f"idx_{table_name}_time"
            with self.engine.connect() as conn:
                 conn.execute(text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} (time)"))
                 conn.commit()
                 
            print(f"Saved {len(df)} rows to {table_name}")
        except Exception as e:
            print(f"Error saving to DB: {e}")

    def load_candles(self, symbol: str, timeframe="M1", start_date=None, end_date=None) -> pd.DataFrame:
        """
        Loads candles from the database.
        """
        table_name = f"{symbol}_{timeframe}"
        query = f"SELECT * FROM {table_name}"
        
        conditions = []
        if start_date:
            conditions.append(f"time >= '{start_date}'")
        if end_date:
            conditions.append(f"time <= '{end_date}'")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY time ASC"

        try:
            df = pd.read_sql(query, self.engine)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])
            return df
        except Exception as e:
            print(f"Error loading from DB: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Smoke Test
    db = DatabaseManager()
    
    # Create dummy data
    data = {
        'time': pd.date_range(start='2023-01-01', periods=10, freq='1min'),
        'open': [1.0] * 10,
        'high': [1.1] * 10,
        'low': [0.9] * 10,
        'close': [1.0] * 10,
        'tick_volume': [100] * 10
    }
    df = pd.DataFrame(data)
    
    db.save_candles("TEST_EURUSD", df)
    
    loaded_df = db.load_candles("TEST_EURUSD")
    print("Loaded Data:")
    print(loaded_df.head())
