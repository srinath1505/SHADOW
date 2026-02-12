import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import inspect, text

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from data.infrastructure.database import DatabaseManager

def inspect_db():
    print("--- Data Lake Inspection ---")
    db = DatabaseManager()
    
    # Get Table Names
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    
    print(f"Database Path: {db.db_path}")
    print(f"Total Tables: {len(tables)}\n")
    
    print(f"{'Table Name':<25} | {'Rows':<10} | {'Start Date':<20} | {'End Date':<20}")
    print("-" * 85)
    
    for table in tables:
        try:
            with db.engine.connect() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                
                # Check if 'time' column exists for range
                columns = [col['name'] for col in inspector.get_columns(table)]
                
                start_date = "N/A"
                end_date = "N/A"
                
                if 'time' in columns:
                    min_t = conn.execute(text(f"SELECT MIN(time) FROM {table}")).scalar()
                    max_t = conn.execute(text(f"SELECT MAX(time) FROM {table}")).scalar()
                    start_date = str(min_t) if min_t else "N/A"
                    end_date = str(max_t) if max_t else "N/A"
                
                print(f"{table:<25} | {count:<10} | {start_date:<20} | {end_date:<20}")
        except Exception as e:
            print(f"{table:<25} | ERROR: {e}")

if __name__ == "__main__":
    inspect_db()
