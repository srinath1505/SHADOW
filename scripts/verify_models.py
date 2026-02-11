import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.ensemble import AntigravityEnsemble

def verify_models():
    print("--- Antigravity T-X-H: Model Verification ---")
    
    ensemble = AntigravityEnsemble()
    ensemble.load_models()
    
    if not ensemble.ready:
        print("[FAIL] Ensemble not ready (Models missing?)")
        return
        
    print("[PASS] Ensemble Loaded.")
    
    # Create Dummy Data for Inference
    # HMM needs: 'ATR_14', 'volatility_ratio', 'log_ret'
    # XGBoost needs: 'RSI_14', 'ADX_14', 'ATR_14', 'volatility_ratio', 'hmm_state', 'z_score_20', 'hour', 'trans_score'
    
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
    df = pd.DataFrame(data)
    
    print("Running Inference on Dummy Data...")
    
    ctx = {
        'timestamp': "2026-02-12T14:00:00",
        'signal_id': "VERIFY-1",
        'pair': "EURUSD",
        'action': "BUY"
    }
    
    try:
        result = ensemble.check_signal(df, ctx)
        print("\n[PASS] Inference Successful!")
        print(f"   HMM State: {result.hmm_state} (Conf: {result.hmm_confidence:.2f})")
        print(f"   XGB Prob:  {result.xgboost_prob:.4f}")
        print(f"   Status:    {result.final_status}")
    except Exception as e:
        print(f"\n[FAIL] Inference Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_models()
