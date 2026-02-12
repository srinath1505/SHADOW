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
    report_path = PROJECT_ROOT / "VALIDATION_REPORT_2026.md"
    
    with open(report_path, "w") as f:
        f.write("# Validation Report (Jan-Feb 2026) - Optimized 2.0\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Period:** Jan 1, 2026 - Feb 28, 2026\n")
        f.write("**Configuration:** `XGB_Threshold=0.50`, `Allow_Volatile=True`\n\n")
        
        # 1. Overall Metrics
        total_trades = len(trades_df)
        net_profit = final_equity - initial_capital
        ret_pct = (net_profit / initial_capital) * 100
        
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        
        peak = equity_df['equity'].max()
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
        f.write(f"- **Profit Factor:** {(wins['pnl'].sum() / abs(losses['pnl'].sum())) if not losses.empty and losses['pnl'].sum() != 0 else 'Inf':.2f}\n\n")
        
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

def run_validation():
    print("--- Validation Run (Jan-Feb 2026) ---")
    
    # 1. Load Data
    db = DatabaseManager()
    symbol = "EURUSD"
    
    TEST_START = "2026-01-01"
    TEST_END = "2026-02-28" 
    # Or present if data allows, logic will take what's available
    
    logging.info(f"Loading Test Data: {TEST_START} to {TEST_END}")
    df = db.load_candles(symbol)
    df = df[(df['time'] >= TEST_START) & (df['time'] <= TEST_END)].drop_duplicates(subset=['time']).reset_index(drop=True)
    
    if df.empty:
        logging.error("No data for 2026!")
        return

    # Load Secondary Data for Transformer
    symbol_sec = "GBPUSD"
    logging.info(f"Loading Secondary Data: {symbol_sec}")
    df_sec = db.load_candles(symbol_sec)
    if not df_sec.empty:
         df_sec = df_sec[(df_sec['time'] >= TEST_START) & (df_sec['time'] <= TEST_END)].drop_duplicates(subset=['time']).reset_index(drop=True)
    
    # 2. Init AI with Optimized Thresholds
    ensemble = AntigravityEnsemble()
    ensemble.load_models()
    
    # Optimization Result: XGB > 0.50, Volatile Allowed
    ensemble.active_config = {
        'xgb_threshold': 0.50,
        'allow_volatile': True
    }
    print("Applied Optimization: XGB > 0.50, Volatile=True")

    # 3. Init Backtester
    bt = Backtester(ensemble=ensemble, initial_capital=10000.0)
    
    # 4. Run
    # Pass secondary_df (GBPUSD) to enable Transformer (Vision)
    trades, equity = bt.run(df, secondary_df=df_sec)
    
    # 5. Report
    generate_report(trades, equity, bt.initial_capital, bt.equity)

if __name__ == "__main__":
    run_validation()
