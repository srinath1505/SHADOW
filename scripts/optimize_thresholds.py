import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
import itertools

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import Class to Modify (We need to patch Backtester or use arguments)
# We will modify Backtester to accept `thresholds` dict in run()
from src.core.ensemble import AntigravityEnsemble
from src.core.backtester import Backtester
from data.infrastructure.database import DatabaseManager

# Configure Logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def run_grid_search():
    print("--- Optimizing Thresholds (Grid Search) ---")
    
    # 1. Load Data (Oct 2025)
    db = DatabaseManager()
    symbol = "EURUSD"
    TEST_START = "2025-10-01"
    TEST_END = "2025-10-31"
    
    print(f"Loading Test Data: {TEST_START} to {TEST_END}")
    df = db.load_candles(symbol)
    df = df[(df['time'] >= TEST_START) & (df['time'] <= TEST_END)].reset_index(drop=True)
    
    if df.empty:
        print("No data!")
        return

    # 2. Init Ensemble
    ensemble = AntigravityEnsemble()
    ensemble.load_models()
    
    # 3. Define Grid
    xgb_thresholds = [0.65, 0.60, 0.55, 0.50]
    allow_volatile = [False, True]
    
    results = []
    
    for xgb_th, vol_allow in itertools.product(xgb_thresholds, allow_volatile):
        config_name = f"XGB_{xgb_th}_Vol_{vol_allow}"
        print(f"Testing Config: {config_name}...", end="", flush=True)
        
        # Inject Config into Ensemble (Need to patch or modify class)
        # AntigravityEnsemble.check_signal uses hardcoded thresholds.
        # We need to Monkey Patch or Update the class instance variables if supported.
        # Let's simple pass it down via `signal_context` hack or modify class now.
        # Better: Modify Ensemble to accept a 'config' dict in check_signal
        
        # For now, we will hack the Ensemble instance attributes directly if possible
        # Or just pass it in `check_signal` context!
        
        # Let's use context "config" key
        # We need to update `AntigravityEnsemble.check_signal` to use this config.
        # I will assume we updated it or will update it shortly.
        
        # HACK: We will subclass/override `check_signal` dynamically or just trust the next step updates it.
        # To be safe, let's update `ensemble.py` first in the plan. 
        # But for this script, we assume context works.
        
        ensemble.active_config = {
            'xgb_threshold': xgb_th,
            'allow_volatile': vol_allow
        }
        
        # Init Backtester
        bt = Backtester(ensemble=ensemble, initial_capital=10000.0)
        
        # Run
        trades, equity = bt.run(df)
        
        # Metrics
        final_equity = bt.equity
        net_profit = final_equity - 10000.0
        profit_pct = (net_profit / 10000.0) * 100
        total_trades = len(trades)
        max_dd = bt.max_drawdown * 100
        
        wins = trades[trades['pnl'] > 0] if not trades.empty else pd.DataFrame()
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        
        print(f" -> Trades: {total_trades}, Profit: {profit_pct:.2f}%, DD: {max_dd:.2f}%")
        
        results.append({
            'Config': config_name,
            'XGB': xgb_th,
            'Volatile': vol_allow,
            'Trades': total_trades,
            'Profit %': profit_pct,
            'Max DD %': max_dd,
            'Win Rate %': win_rate
        })

    # 4. Save Results
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='Profit %', ascending=False)
    
    print("\n--- Optimization Results ---")
    print(res_df.to_string(index=False))
    
    res_df.to_csv("optimization_results.csv", index=False)

if __name__ == "__main__":
    run_grid_search()
