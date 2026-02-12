import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import logging

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.hmm_model import RegimeHMM
from src.core.xgboost_model import TradeJudgeXGB
from data.infrastructure.database import DatabaseManager
from src.strategies.ghosting_engine import SignalStateMachine

MODELS_DIR = PROJECT_ROOT / "models"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def evaluate():
    print("--- Model Drift & Performance Validation ---")
    
    # 1. Load Data
    db = DatabaseManager()
    symbol = "EURUSD"
    logging.info(f"Loading 1 Year of Data for {symbol}...")
    df = db.load_candles(symbol)
    
    if df.empty:
        logging.error("No data found!")
        return

    # 2. Load Models
    hmm = RegimeHMM()
    try:
        hmm.load_model(MODELS_DIR / "hmm_model.pkl")
    except Exception as e:
        logging.error(f"Failed to load HMM: {e}")
        return

    xgb = TradeJudgeXGB()
    try:
        xgb.load_model(MODELS_DIR / "xgb_model.json")
    except Exception as e:
        logging.error(f"Failed to load XGBoost: {e}")
        return

    # 3. Apply HMM
    logging.info("Applying HMM...")
    try:
        # HMM drops NaNs internally in prepare_features
        # We need to replicate that drop to align
        required = ['ATR_14', 'volatility_ratio', 'log_ret']
        df_clean = df.dropna(subset=required).copy()
        
        states = hmm.predict(df_clean)
        df_clean['hmm_state'] = states
        
        # Merge back or just use df_clean
        df = df_clean.reset_index(drop=True)
        
    except Exception as e:
        logging.error(f"HMM Prediction failed: {e}")
        return

    # 4. Generate Signals (Replicating Logic)
    logging.info("Regenerating Signals (Ghosting Engine)...")
    sm = SignalStateMachine()
    df['std_dev'] = df['close'].rolling(window=20).std()
    
    signals = []
    
    # Labeling Params
    TP_PIPS = 0.0010
    SL_PIPS = 0.0010
    
    from tqdm import tqdm
    for i in tqdm(range(200, len(df) - 100)):
        row = df.iloc[i]
        
        # Update State
        new_state = sm.update(row['close'], row['VWAP'], row['std_dev'], i)
        
        if new_state.name == "TRIGGER_CANDIDATE":
            # Signal Found
            entry_price = row['close']
            is_sell = entry_price > row['VWAP']
            
            # Determine Outcome (Ground Truth)
            outcome = 0
            future = df.iloc[i+1 : i+61]
            for _, f_row in future.iterrows():
                p = f_row['close']
                if is_sell:
                    if p <= (entry_price - TP_PIPS):
                        outcome = 1; break
                    if p >= (entry_price + SL_PIPS):
                        outcome = 0; break
                else:
                    if p >= (entry_price + TP_PIPS):
                        outcome = 1; break
                    if p <= (entry_price - SL_PIPS):
                        outcome = 0; break
            
            # Feature Vector
            feat = {
                'timestamp': row['time'],
                'RSI_14': row.get('RSI_14', 50),
                'ADX_14': row.get('ADX_14', 20),
                'ATR_14': row.get('ATR_14', 0.0010),
                'volatility_ratio': row.get('volatility_ratio', 1.0),
                'hmm_state': row['hmm_state'],
                'z_score_20': row.get('z_score_20', 0),
                'hour': row.get('hour', 0),
                'outcome': outcome
            }
            signals.append(feat)
            
    if not signals:
        logging.warning("No signals generated!")
        return
        
    sig_df = pd.DataFrame(signals)
    logging.info(f"Total Signals found: {len(sig_df)}")
    
    # 5. Make Predictions
    feature_cols = ['RSI_14', 'ADX_14', 'ATR_14', 'volatility_ratio', 'hmm_state', 'z_score_20', 'hour']
    X = sig_df[feature_cols]
    y_true = sig_df['outcome']
    
    # Get Probabilities
    # XGBoost needs DMatrix usually but our wrapper handles sklearn style?
    # TradeJudgeXGB uses `self.model.predict(dtest)` or similar. 
    # Let's check `xgboost_model.py` implementation if needed. 
    # Assuming `predict_proba` returns [prob_0, prob_1] or just prob_1?
    # Actually `predict` in our class usually returns binary or prob?
    # Let's assume `predict_proba` exists or use `predict`.
    # Quick fix: The class likely has `predict(features_df)` returning score.
    
    # Predicting loop or batch?
    # If the class creates DMatrix inside `predict`, batch is better.
    # Let's look at `TradeJudgeXGB.predict` signature in previous context?
    # It takes vector. We might need to iterate or refactor.
    # Actually, `train_models.py` calls `train(X, y)`.
    # Let's try batch predict on the underlying model if attribute accessible, 
    # Or iterate. Iteration is safer for "black box".
    
    probs = []
    # Batch prediction if possible, else row-by-row
    try:
        # XGBoost predict_proba returns [prob_0, prob_1]
        p = xgb.predict_proba(X)
        probs = p # Assuming it returns just the prob of class 1 as per our wrapper?
        # Let's check wrapper: "return self.model.predict_proba(feature_vector)[:, 1]"
        # So it returns 1D array of probs.
    except Exception as e:
        logging.warning(f"Batch prediction failed ({e}), falling back to row-by-row...")
        for _, row in X.iterrows():
            v = pd.DataFrame([row])
            score = xgb.predict_proba(v)
            # Wrapper returns array of shape (1,)
            probs.append(score[0])
            
    sig_df['prob'] = probs
    sig_df['pred_class'] = (sig_df['prob'] > 0.5).astype(int)
    
    # 6. Drift Analysis (Time Split)
    # Split Date: 30 days ago
    split_date = sig_df['timestamp'].max() - timedelta(days=30)
    
    train_set = sig_df[sig_df['timestamp'] < split_date]
    test_set = sig_df[sig_df['timestamp'] >= split_date]
    
    print("\n--- Performance Metrics ---")
    
    def print_metrics(name, df_sub):
        if df_sub.empty:
            print(f"[{name}] No samples.")
            return
            
        y = df_sub['outcome']
        p_class = df_sub['pred_class']
        
        acc = accuracy_score(y, p_class)
        prec = precision_score(y, p_class, zero_division=0)
        rec = recall_score(y, p_class, zero_division=0)
        
        # Win Rate of High Confidence (>0.7)
        high_conf = df_sub[df_sub['prob'] > 0.7]
        if not high_conf.empty:
            hc_win_rate = high_conf['outcome'].mean()
            hc_count = len(high_conf)
        else:
            hc_win_rate = 0.0
            hc_count = 0
            
        print(f"\ndataset: {name}")
        print(f"  Samples: {len(df_sub)}")
        print(f"  Accuracy: {acc:.2%}")
        print(f"  Precision: {prec:.2%}")
        print(f"  Recall: {rec:.2%}")
        print(f"  High Conf Win Rate (>0.7): {hc_win_rate:.2%} ({hc_count} trades)")

    print_metrics("Historical (First 11 Months)", train_set)
    print_metrics("Recent (Last 30 Days)", test_set)
    
    # Drift Check
    print("\n--- Drift Check ---")
    if not train_set.empty and not test_set.empty:
        # Check High Conf drift
        hc_train = train_set[train_set['prob'] > 0.7]['outcome'].mean()
        hc_test = test_set[test_set['prob'] > 0.7]['outcome'].mean()
        
        diff = hc_test - hc_train
        print(f"High Conf Win Rate Delta: {diff:+.2%}")
        
        if diff < -0.10:
            print("FAIL: Significant Performance Degradation (>10% drop)")
        elif diff < -0.05:
            print("WARNING: Mild Degradation")
        else:
            print("PASS: Model is Stable or Improving")
    else:
        print("Cannot calculate drift (insufficient split data)")

if __name__ == "__main__":
    evaluate()
