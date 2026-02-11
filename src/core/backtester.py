import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from src.core.ensemble import AntigravityEnsemble
from src.strategies.ghosting_engine import SignalStateMachine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Backtester:
    def __init__(self, ensemble: AntigravityEnsemble, initial_capital=10000.0, spread_pips=1.0, commission_per_lot=7.0):
        self.ensemble = ensemble
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.equity = initial_capital
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        
        self.trades = []
        self.equity_curve = []
        self.positions = [] # List of open positions (dicts)
        
        # Risk Management Settings
        self.lot_size = 0.1 # Fixed for now
        self.tp_pips = 10
        self.sl_pips = 10
        self.pip_value = 10.0 # Standard lot pip value approx $10 (for 1 lot) -> $1 for 0.1
        
        # Metrics
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital

    def _calculate_profit(self, entry_price, exit_price, direction, size):
        """
        Calculates profit in USD.
        """
        pip_diff = 0
        if direction == "BUY":
            pip_diff = (exit_price - entry_price) * 10000
        else:
            pip_diff = (entry_price - exit_price) * 10000
            
        # Value per pip for size
        # 1 Lot = $10 per pip
        # Size is in Lots
        gross_profit = pip_diff * 10.0 * size 
        return gross_profit

    def _apply_slippage(self, price):
        """
        Simulates random slippage (0.1 - 0.5 pips) against the trade.
        """
        slip = np.random.uniform(0.00001, 0.00005) # 0.1 to 0.5 pips
        return slip


    def run(self, df: pd.DataFrame, start_date=None, end_date=None):
        """
        Runs the backtest.
        """
        # Filter Data
        if start_date:
            df = df[df['time'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['time'] <= pd.to_datetime(end_date)]
        
        df = df.reset_index(drop=True)
        logging.info(f"Backtesting on {len(df)} candles...")
        
        # Pre-calculate Std Dev for Ghosting if missing
        if 'std_dev' not in df.columns:
             # Match logic from train_models.py
             df['std_dev'] = df['close'].rolling(window=20).std()
        
        # Initialize Ghosting Engine
        sm = SignalStateMachine()
        
        # Iteration
        for i in range(200, len(df)): # Start after warm-up
            row = df.iloc[i]
            current_time = row['time']
            current_price = row['close']
            
            # Update Ghosting State
            vwap = row.get('VWAP', current_price) # Fallback if missing
            std_dev = row.get('std_dev', 0.0005)
            
            # Engine Update
            new_state = sm.update(current_price, vwap, std_dev, i)
            
            # 1. Manage Open Positions (TP/SL)
            high = row['high']
            low = row['low']
            
            positions_to_close = []
            
            for pos in self.positions:
                # Check SL/TP
                sl_hit = False
                tp_hit = False
                exit_price = 0
                
                # Check for gap opening beyond SL/TP? 
                # For M1, we assume logic executes within candle range or at Open if gap.
                # Simplified: Check High/Low against thresholds.
                
                if pos['type'] == 'BUY':
                    if low <= pos['sl']:
                        sl_hit = True
                        exit_price = pos['sl'] - self._apply_slippage(pos['sl'])
                    elif high >= pos['tp']:
                        tp_hit = True
                        exit_price = pos['tp'] 
                else: # SELL
                    if high >= pos['sl']:
                        sl_hit = True
                        exit_price = pos['sl'] + self._apply_slippage(pos['sl'])
                    elif low <= pos['tp']:
                        tp_hit = True
                        exit_price = pos['tp']
                        
                if sl_hit or tp_hit:
                    # Calculate PnL
                    gross_pnl = self._calculate_profit(pos['entry_price'], exit_price, pos['type'], pos['size'])
                    comm = self.commission_per_lot * pos['size']
                    net_pnl = gross_pnl - comm
                    
                    self.balance += net_pnl
                    pos['exit_price'] = exit_price
                    pos['exit_time'] = current_time
                    pos['pnl'] = net_pnl
                    pos['reason'] = 'TP' if tp_hit else 'SL'
                    
                    self.trades.append(pos)
                    positions_to_close.append(pos)
            
            for p in positions_to_close:
                self.positions.remove(p)
                
            # 2. Check Signals (Only if flat)
            if not self.positions:
                # Check for Trigger from Ghosting Engine
                if new_state.name == "TRIGGER_CANDIDATE":
                    # We have a candidate! Ask the Ensemble.
                    
                    # Context for Ensemble
                    ctx = {
                        'timestamp': str(current_time),
                        'signal_id': f"BT-{i}",
                        'pair': "EURUSD",
                        'action': "CHECK"
                    }
                    
                    # Ensemble needs data up to this point.
                    # Optimization: Pass a window.
                    window = df.iloc[i-100:i+1]
                    
                    signal_result = self.ensemble.check_signal(window, ctx)
                    
                    if signal_result.final_status == "APPROVED":
                        # Execute Trade!
                        
                        # Direction? 
                        # Ghosting is Mean Reversion.
                        # If Price > VWAP -> SELL
                        # If Price < VWAP -> BUY
                        trade_type = "SELL" if current_price > vwap else "BUY"
                        
                        # Entry Price (Close of trigger candle)
                        entry_price = current_price
                        
                        # Spread Penalty
                        # If BUY, ask = price + spread. If SELL, bid = price.
                        # We simulate by shifting entry price worse by spread/2 or full spread?
                        # Usually: Buy at Ask, Sell at Bid.
                        # Data is Bid.
                        # Buy Entry = Bid + Spread. Sell Entry = Bid.
                        # Exit Buy = Bid. Exit Sell = Bid + Spread.
                        
                        spread_val = self.spread_pips * 0.0001
                        
                        real_entry = entry_price
                        if trade_type == "BUY":
                            real_entry += spread_val
                        
                        # SL/TP
                        sl_dist = self.sl_pips * 0.0001
                        tp_dist = self.tp_pips * 0.0001
                        
                        sl_price = 0
                        tp_price = 0
                        
                        if trade_type == "BUY":
                            sl_price = real_entry - sl_dist
                            tp_price = real_entry + tp_dist
                        else:
                            sl_price = real_entry + sl_dist
                            tp_price = real_entry - tp_dist
                        
                        # Record Position
                        new_pos = {
                            'id': f"TR-{i}",
                            'entry_time': current_time,
                            'type': trade_type,
                            'size': self.lot_size,
                            'entry_price': real_entry,
                            'sl': sl_price,
                            'tp': tp_price
                        }
                        self.positions.append(new_pos)

            # 3. Update Equity Curve
            open_pnl = 0
            for pos in self.positions:
                # Current Val
                # Buy closes at Bid (current_price)
                # Sell closes at Ask (current_price + spread)
                
                val_price = current_price
                if pos['type'] == 'SELL':
                     val_price += (self.spread_pips * 0.0001)
                
                if pos['type'] == 'BUY':
                    dist = val_price - pos['entry_price']
                else:
                    dist = pos['entry_price'] - val_price
                    
                open_pnl += dist * 10000 * 10.0 * pos['size']
            
            self.equity = self.balance + open_pnl
            self.equity_curve.append({'time': current_time, 'equity': self.equity, 'balance': self.balance})
            
            # Update Peak
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            
            # Update DD
            if self.peak_equity > 0:
                dd = (self.peak_equity - self.equity) / self.peak_equity
                if dd > self.max_drawdown:
                    self.max_drawdown = dd

        return pd.DataFrame(self.trades), pd.DataFrame(self.equity_curve)

    def print_stats(self):
        if not self.trades:
            print("No trades executed.")
            return

        df_t = pd.DataFrame(self.trades)
        wins = df_t[df_t['pnl'] > 0]
        losses = df_t[df_t['pnl'] <= 0]
        
        win_rate = len(wins) / len(df_t) * 100
        total_pnl = df_t['pnl'].sum()
        
        print(f"Total Trades: {len(df_t)}")
        print(f"Net Profit: ${total_pnl:.2f}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Max Drawdown: {self.max_drawdown*100:.2f}%")
        print(f"Final Equity: ${self.equity:.2f}")

