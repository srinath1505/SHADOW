import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.ensemble import AntigravityEnsemble
from src.core.backtester import Backtester
from data.infrastructure.database import DatabaseManager

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_report(trades_df, equity_df, initial_capital, final_equity):
    """
    Generates a detailed Markdown report.
    """
    report_path = PROJECT_ROOT / "ROBUST_BACKTEST_REPORT.md"
    
    with open(report_path, "w") as f:
        f.write("# Robust Backtest Report (Sprint 4.5)\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Period:** Oct 1, 2025 - Oct 31, 2025\n\n")
        
        # 1. Overall Metrics
        total_trades = len(trades_df)
        net_profit = final_equity - initial_capital
        ret_pct = (net_profit / initial_capital) * 100
        
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        
        peak = equity_df['equity'].max()
        dd_val = peak - equity_df['equity'].min() # Rough max DD from peak? 
        # Better: Calculate max drawdown % from curve
        # Backtester tracks it roughly, let's recalculate accurately
        equity_curve = equity_df['equity'].values
        running_max = pd.Series(equity_curve).cummax()
        drawdowns = (running_max - equity_curve) / running_max
        max_dd_pct = drawdowns.max() * 100
        
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Initial Capital:** ${initial_capital:,.2f}\n")
        f.write(f"- **Final Equity:** ${final_equity:,.2f}\n")
        f.write(f"- **Net Profit:** ${net_profit:,.2f} ({ret_pct:+.2f}%)\n")
        f.write(f"- **Max Drawdown:** {max_dd_pct:.2f}%\n")
        f.write(f"- **Total Trades:** {total_trades}\n")
        f.write(f"- **Win Rate:** {win_rate:.2f}%\n")
        f.write(f"- **Profit Factor:** {(wins['pnl'].sum() / abs(losses['pnl'].sum())) if not losses.empty else 'Inf':.2f}\n\n")
        
        # 2. Daily Journal
        f.write("## 2. Daily Journal (Trade-by-Trade)\n")
        if not trades_df.empty:
            trades_df['date'] = pd.to_datetime(trades_df['exit_time']).dt.date
            daily_groups = trades_df.groupby('date')
            
            for date, group in daily_groups:
                daily_pnl = group['pnl'].sum()
                daily_trades = len(group)
                daily_wins = len(group[group['pnl'] > 0])
                
                f.write(f"### {date} | P&L: ${daily_pnl:+.2f} | Trades: {daily_trades} (Win: {daily_wins})\n")
                f.write("| ID | Type | Size | Entry | Exit | P&L | Reason |\n")
                f.write("|---|---|---|---|---|---|---|\n")
                for _, t in group.iterrows():
                    f.write(f"| {t['id']} | {t['type']} | {t['size']} | {t['entry_price']:.5f} | {t['exit_price']:.5f} | ${t['pnl']:+.2f} | {t['reason']} |\n")
                f.write("\n")
        else:
            f.write("No trades executed.\n")
            
        print(f"Report generated at {report_path}")

def run_robust_test():
    print("--- Robust Backtest (Oct 2025) ---")
    
    # 1. Load Data
    db = DatabaseManager()
    symbol = "EURUSD"
    
    TEST_START = "2025-10-01"
    TEST_END = "2025-10-31"
    
    logging.info(f"Loading Test Data: {TEST_START} to {TEST_END}")
    df = db.load_candles(symbol)
    df = df[(df['time'] >= TEST_START) & (df['time'] <= TEST_END)].reset_index(drop=True)
    
    if df.empty:
        logging.error("No data for Oct 2025!")
        return

    # 2. Init AI
    ensemble = AntigravityEnsemble()
    ensemble.load_models()
    if not ensemble.ready:
        logging.error("Models not ready. Run train_models.py first!")
        return

    # 3. Init Backtester
    # Strict settings hardcoded in class defaults? 
    # Yes, we updated __init__ defaults or logic.
    bt = Backtester(ensemble=ensemble, initial_capital=10000.0)
    
    print(f"Risk Rules: Daily Limit {bt.daily_risk_limit_pct:.1%}, Global Stop {bt.global_hard_stop_pct:.1%}, Weekly Cap {bt.weekly_profit_cap_pct:.1%}")
    
    # 4. Run
    trades, equity = bt.run(df)
    
    # 5. Report
    generate_report(trades, equity, bt.initial_capital, bt.equity)

if __name__ == "__main__":
    run_robust_test()
