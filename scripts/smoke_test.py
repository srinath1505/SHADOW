import sys
from pathlib import Path
import pandas as pd
import MetaTrader5 as mt5

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.connectors.mt5_connector import MT5Connector
from src.features.features import FeatureEngineer
from src.strategies.correlation_guard import CorrelationGuard
from src.strategies.ghosting_engine import SignalStateMachine, GhostState
from src.core.ensemble import AntigravityEnsemble
from src.core.backtester import Backtester

def print_result(test_name, status, message=""):
    icon = "[PASS]" if status else "[FAIL]"
    print(f"{icon} {test_name}: {message}")
    return status

def smoke_test():
    print("--- Antigravity T-X-H: Smoke Test Report ---")
    all_passed = True
    
    # --- 1. Connector Check ---
    print("\n[1. Connector & Account]")
    connector = MT5Connector()
    if connector.initialize():
        print_result("MT5 Initialization", True, f"Version: {mt5.version()}")
        if connector.login():
            account_info = mt5.account_info()
            if account_info:
                print_result("Account Connection", True, f"Login: {account_info.login}, Server: {account_info.server}")
            else:
                all_passed = False
                print_result("Account Connection", False, "Could not fetch account info")
        else:
            all_passed = False
            print_result("Account Login", False, "Login failed (check credentials)")
    else:
        all_passed = False
        print_result("MT5 Initialization", False, "Failed to initialize")
        print("\n❌ CRITICAL ERROR: Cannot proceed without MT5.")
        return

    # --- 2. Data Pipeline Check ---
    print("\n[2. Data Pipeline]")
    ticks = connector.get_live_ticks("EURUSD", 10)
    if ticks is not None and not ticks.empty:
        print_result("Live Ticks", True, f"Fetched {len(ticks)} ticks")
    else:
        all_passed = False
        print_result("Live Ticks", False, "Failed to fetch ticks")

    candles = connector.get_historical_candles("EURUSD", "M1", 300)
    if candles is not None and not candles.empty:
        print_result("Historical Data", True, f"Fetched {len(candles)} M1 candles")
        
        
        # Feature Check
        fe = FeatureEngineer()
        features = fe.calculate_features(candles)
        
        # Debug Info
        # print(f"DEBUG: Features Shape: {features.shape}")
        # print(f"DEBUG: Columns: {features.columns.tolist()}")

        if not features.empty and 'RSI_14' in features.columns:
            # Check for NaN validation
            rsi_val = features['RSI_14'].iloc[-1]
            if pd.isna(rsi_val):
                 print_result("Feature Engineering", False, f"RSI_14 is NaN (Shape: {features.shape})")
                 all_passed = False
            else:
                 print_result("Feature Engineering", True, f"Calculated {len(features.columns)} features (RSI: {rsi_val:.2f})")
        else:
            all_passed = False
            print(f"   Shape: {features.shape}")
            print(f"   Columns: {features.columns.tolist()}")
            if features.empty:
                print("   Error: Result Empty (dropped too many NaNs?)")
            print_result("Feature Engineering", False, "Failed to calculate features")
    else:
        all_passed = False
        print_result("Historical Data", False, "Failed to fetch candles")

    # --- 3. Correlation Check ---
    print("\n[3. Correlation Guard]")
    eur = connector.get_historical_candles("EURUSD", "M1", 100)
    gbp = connector.get_historical_candles("GBPUSD", "M1", 100)
    
    if eur is not None and gbp is not None:
        # Align (simple tail check)
        min_len = min(len(eur), len(gbp))
        s1 = eur['close'].tail(min_len)
        s2 = gbp['close'].tail(min_len)
        
        cg = CorrelationGuard()
        corr, status = cg.check_correlation(s1, s2)
        print_result("Correlation Math", True, f"EUR/GBP Correlation: {corr:.4f} ({status})")
    else:
        all_passed = False
        print_result("Correlation Math", False, "Could not fetch data for both pairs")

    # --- 4. Ghosting Logic Check ---
    print("\n[4. Ghosting State Machine]")
    sm = SignalStateMachine()
    
    # Simulator
    # 1. Idle -> Alert (Price > Band)
    # 2. Alert -> Wait (Pullback)
    # 3. Wait -> Trigger (Re-test)
    
    # VWAP = 1.0000, Std = 0.0010
    # Band 2.5 = 1.0025. Band 1.5 = 1.0015
    
    sequence = [
        (1.0000, "IDLE"), # Baseline
        (1.0026, "ALERT"), # Spike > 1.0025
        (1.0014, "WAITING"), # Pullback < 1.0015
        (1.0024, "TRIGGER_CANDIDATE") # Retest near 1.0026 (High)
    ]
    
    logic_passed = True
    for i, (price, expected) in enumerate(sequence):
        state = sm.update(price, vwap=1.0000, std_dev=0.0010, index=i)
        if state.name != expected:
            print(f"  Step {i}: Expected {expected}, Got {state.name}")
            logic_passed = False
    
    print_result("State Logic Flow", logic_passed, "Sequence: Idle -> Alert -> Waiting -> Trigger")
    if not logic_passed:
        all_passed = False

    # --- 5. AI Models (Sprint 3) ---
    print("\n[5. AI Models (Ensemble)]")
    try:
        ensemble = AntigravityEnsemble()
        ensemble.load_models()
        
        if ensemble.ready:
            print_result("Model Loading", True, "HMM and XGBoost loaded")
            
            # Dummy Inference
            # Ensure we use valid data types (float, int) to avoid XGBoost errors
            data = {
                'RSI_14': [30.5],
                'ADX_14': [25.0],
                'ATR_14': [0.0012],
                'volatility_ratio': [1.1],
                'log_ret': [0.0005],
                'z_score_20': [-2.1],
                'hour': [14],
                'time': pd.to_datetime(['2026-02-12 14:00:00']) 
            }
            df_dummy = pd.DataFrame(data)
            
            ctx = {'timestamp': "Mock", 'signal_id': "Test", 'pair': "EURUSD", 'action': "BUY"}
            
            result = ensemble.check_signal(df_dummy, ctx)
            print_result("Inference", True, f"Status: {result.final_status}, HMM: {result.hmm_state}, XGB: {result.xgboost_prob:.2f}")
            
        else:
            all_passed = False
            print_result("Model Loading", False, "Failed to load models")
            
    except Exception as e:
        all_passed = False
        print_result("AI Verification", False, f"Exception: {e}")

    # --- 6. Backtester (Sprint 4) ---
    print("\n[6. Backtester (Simulation)]")
    try:
        # Create a tiny dataframe for backtest verification
        dates = pd.date_range(start='2026-01-01', periods=500, freq='1min')
        data = {
            'time': dates,
            'open': [1.1]*500,
            'high': [1.11]*500,
            'low': [1.09]*500,
            'close': [1.105]*500,
            'tick_volume': [100]*500,
            'spread': [1]*500,
            'real_volume': [100]*500,
            'RSI_14': [50.0]*500,
            'ADX_14': [25.0]*500,
            'ATR_14': [0.001]*500,
            'volatility_ratio': [1.0]*500,
            'z_score_20': [0.5]*500,
            'hour': [12]*500,
            'log_ret': [0.0]*500
        }
        df_bt = pd.DataFrame(data)
        
        # Instantiate Backtester
        bt = Backtester(ensemble=ensemble)
        
        # Run on dummy data
        t, e = bt.run(df_bt)
        
        # Check if it ran without error (even if 0 trades)
        print_result("Backtester Execution", True, f"Processed {len(df_bt)} candles, Equity: ${bt.equity:.2f}")
        
    except Exception as e:
        all_passed = False
        print_result("Backtester Error", False, f"Exception: {e}")

    connector.shutdown()
    
    print("\n---------------------------------------")
    if all_passed:
        print("SMOKE TEST PASSED: SYSTEM IS GO FOR SPRINT 3")
    else:
        print("SMOKE TEST FAILED: CHECK LOGS ABOVE")

if __name__ == "__main__":
    smoke_test()
