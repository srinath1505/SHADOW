import sys
from pathlib import Path
import pandas as pd
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.backtester import Backtester
from src.core.ensemble import AntigravityEnsemble
from data.infrastructure.database import DatabaseManager

def main():
    print("--- Antigravity T-X-H: Simulation & Stress Test ---")
    
    # 1. Initialize Components
    db = DatabaseManager()
    ensemble = AntigravityEnsemble()
    ensemble.load_models()
    
    if not ensemble.ready:
        print("❌ CRITICAL: Ensemble models not found. Train them first!")
        return

    # 2. Load Data (EURUSD)
    symbol = "EURUSD"
    print(f"Loading data for {symbol}...")
    df = db.load_candles(symbol)
    
    if df.empty:
        print("❌ No data found.")
        return

    # 3. Define Walk-Forward Split
    # Train was Nov-Dec. Test is Jan 2026 onwards.
    # We simulate on the TEST set only to verified out-of-sample performance.
    test_start_date = "2026-01-01"
    
    print(f"Starting Simulation from {test_start_date}...")
    
    # 4. Initialize Backtester
    # Initial Capital: $10,000
    # Spread: 1.0 pip (Standard)
    # Commission: $7.00/lot
    backtester = Backtester(
        ensemble=ensemble, 
        initial_capital=10000.0,
        spread_pips=1.0, 
        commission_per_lot=7.0
    )
    
    # 5. Run
    trades, equity = backtester.run(df, start_date=test_start_date)
    
    # 6. Report
    print("\n--- Performance Report ---")
    backtester.print_stats()
    
    if not trades.empty:
        # Save results
        trades.to_csv(PROJECT_ROOT / "data/processed/backtest_trades.csv")
        equity.to_csv(PROJECT_ROOT / "data/processed/backtest_equity.csv")
        print("\nDetailed logs saved to data/processed/")

if __name__ == "__main__":
    main()
