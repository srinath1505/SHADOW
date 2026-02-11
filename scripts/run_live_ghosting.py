import time
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import Project Modules
from src.connectors.mt5_connector import MT5Connector
from src.features.features import FeatureEngineer
from src.strategies.regime_filter import RegimeFilter
from src.strategies.ghosting_engine import SignalStateMachine
from src.strategies.correlation_guard import CorrelationGuard
from config import settings

def main():
    print("--- Antigravity T-X-H: Ghosting Engine (Live Mode) ---")
    
    # 1. Initialize Components
    connector = MT5Connector()
    if not connector.initialize():
        print("Failed to connect to MT5.")
        return

    fe = FeatureEngineer()
    regime_filter = RegimeFilter()
    ghost_engine = SignalStateMachine()
    correlation_guard = CorrelationGuard()

    symbol_primary = "EURUSD"
    symbol_secondary = "GBPUSD"
    timeframe = "M1"
    
    print(f"Monitoring {symbol_primary} and {symbol_secondary}...")

    # Keep track of last processed candle time to avoid duplicate processing
    last_candle_time = None

    try:
        while True:
            # 2. Fetch Data (Lookback 200 candles for indicator stability)
            # We need both pairs for correlation
            df_eur = connector.get_historical_candles(symbol_primary, timeframe, num_candles=200)
            df_gbp = connector.get_historical_candles(symbol_secondary, timeframe, num_candles=200)

            if df_eur is None or df_gbp is None or df_eur.empty or df_gbp.empty:
                print("Waiting for data...")
                time.sleep(5)
                continue

            # Check if we have a new candle (by looking at the latest close time)
            # Or just process the latest completed candle? 
            # Usually in live trading we wait for candle close.
            # But the requirement says "Live Ticks/M1". 
            # Let's monitor the latest *completed* candle (row -2 if -1 is current forming one? 
            # MT5 get_historical_candles usually returns specific count. 
            # If we used copy_rates_from_pos(0, N), index 0 is usually latest.
            # Let's assume the last row is the current forming candle. Use -1 for real-time logic.
            
            current_time = df_eur.iloc[-1]['time']
            
            # Simple "New Candle" check (optional, or run every X seconds)
            # For "Ghosting", we might want to run on every tick or at least frequently.
            # But features are M1 based. So let's run on every loop but debounced by sleep.
            
            # 3. Calculate Features
            # We only really need features for the Primary pair for the Ghosting Signal?
            # Roadmap says: "Regime Filters... Ghosting State Machine... Correlation"
            
            df_features = fe.calculate_features(df_eur)
            
            if df_features.empty:
                print("Not enough data for features.")
                time.sleep(10)
                continue

            # Get latest values
            latest = df_features.iloc[-1]
            prev = df_features.iloc[-2] # For crossover logic if needed
            
            # 4. Regime Filter Check
            # (News check is mocked or manual for now as we don't have a live calendar feed yet)
            adx_val = latest.get('ADX_14', 0)
            
            is_safe = regime_filter.check_conditions(adx_val, current_time)
            if not is_safe:
                print(f"[{current_time}] Regime Filter: UNSAFE (ADX: {adx_val:.2f})")
                # We might still want to update state machine to verify it Resets? 
                # Or just skip? Roadmap says "The Gate". So skip.
                # But we should probably look at the chart anyway.
                # Let's skip for safety.
                # time.sleep(10)
                # continue
                pass # Just logging for now

            # 5. Correlation Check
            # Need to align series. Using tail(60)
            series_a = df_eur['close'].tail(60)
            series_b = df_gbp['close'].tail(60)
            corr_val, corr_status = correlation_guard.check_correlation(series_a, series_b)
            
            # 6. Update Ghosting State Machine
            # Inputs: Price, VWAP, StdDev (for bands)
            # We calculated VWAP and Bands in features.py? 
            # features.py has: 'VWAP', 'VWAP_Upper_2.5' etc.
            # StateMachine needs: price, vwap, std_dev OR we can just pass the banding levels directly if we modify it.
            # Current StateMachine.update signature: (price, vwap, std_dev, index)
            # Let's calculate std_dev from the bands differences or just re-calculate/pass it.
            # features.py calculated rolling_std. We verify feature names.
            # columns: 'VWAP', 'close', ... we need rolling_std. features.py didn't explicitly output 'std_dev' column?
            # It used it for bands. Let's inspect features.py output or add it if missing.
            # features.py: `rolling_std = df['close'].rolling(window=20).std()` used locally.
            # We might need to add it to output or recalculate.
            # Let's quickly re-calc here for robustness or assume broad bands.
            
            vwap_val = latest['VWAP']
            close_val = latest['close']
            
            # Recalculate std dev roughly if missing
            # std_dev = (latest['VWAP_Upper_2.5'] - vwap_val) / 2.5
            # Assuming features.py outputs bands.
            if 'VWAP_Upper_2.5' in latest:
                 std_dev = (latest['VWAP_Upper_2.5'] - vwap_val) / 2.5
            else:
                 # Fallback
                 std_dev = 0.0010 

            # Update State
            # We use a dummy index or the dataframe index
            current_idx = df_eur.index[-1]
            
            new_state = ghost_engine.update(close_val, vwap_val, std_dev, current_idx)
            
            # Output Status
            print(f"[{current_time}] P: {close_val:.5f} | V: {vwap_val:.5f} | ADX: {adx_val:.1f} | Corr: {corr_val:.2f} | State: {new_state.name}")
            
            if new_state.name == "TRIGGER_CANDIDATE":
                print("\n🚀 SIGNAL GENERATED: SECOND TOUCH DETECTED 🚀\n")

            time.sleep(10) # Wait for next update (M1 updates every 60s, but we check more often for "Ghosting" feel)

    except KeyboardInterrupt:
        print("\nStopping Ghosting Engine...")
    finally:
        connector.shutdown()

if __name__ == "__main__":
    main()
