import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.hmm_model import RegimeHMM
from src.core.xgboost_model import TradeJudgeXGB
from src.core.transformer_model import TransformerPredictor
from src.features.features import FeatureEngineer
from data.infrastructure.database import DatabaseManager
from src.strategies.ghosting_engine import SignalStateMachine, GhostState

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_hmm(df: pd.DataFrame):
    """
    Trains HMM on the entire dataset.
    """
    logging.info("Training HMM Model...")
    hmm = RegimeHMM(n_components=3, n_iter=100)
    try:
        hmm.fit(df)
        hmm.save_model(MODELS_DIR / "hmm_model.pkl")
        logging.info("HMM Model Saved.")
        return hmm
    except Exception as e:
        logging.error(f"HMM Training Failed: {e}")
        return None

def generate_signals_and_labels(df: pd.DataFrame, hmm_model):
    """
    Simulates Ghosting Logic to find signals and labels them.
    Target: Did price go > 10 pips in favor before hitting SL?
    """
    logging.info("Generating Signals and Labels for XGBoost...")
    
    sm = SignalStateMachine()
    
    signals = []
    
    # We need to iterate through the data
    # This simulates the "Live" loop but on history
    
    # Pre-calculate features? Yes, df has them.
    # We need VWAP and StdDev for bands.
    # We need to compute HMM state for each row first?
    # Yes, obtaining HMM states for the whole history is fast.
    
    if hmm_model:
        try:
            states = hmm_model.predict(df)
            df['hmm_state'] = states
        except:
             df['hmm_state'] = 0 # Default if failed
    else:
        df['hmm_state'] = 0

    # Iterator
    # We need to look forward for labeling.
    # Collecting X (Features at trigger) and y (Outcome)
    
    # Parameters for labeling
    TP_PIPS = 0.0010 # 10 pips
    SL_PIPS = 0.0010 # 10 pips (1:1 R:R for binary classification simplicity) or as per strategy?
    # PRD said "Reversed > 10 pips". Let's use 10 pips.
    
    # Pre-compute bands to save time in loop
    # features.py computes 'VWAP', 'VWAP_Upper_2.5', 'VWAP_Lower_1.5' (pullback) etc.
    # ghosting_engine uses: price, vwap, std_dev.
    # We can reconstruct std_dev or use the bands directly if we modify engine.
    # Let's use the engine as is. Reconstruct std_dev from bands if available.
    
    # We need high band (2.5) and pullback band (1.5).
    # features.py has 2.0 and 2.5. Let's assume 1.5 is roughly (2.0 + 1.0)/2 or just calculate.
    # Actually features.py calculated rolling_std.
    # But it didn't save 'rolling_std' column? 
    # Let's recalculate rolling_std(20) here to be sure.
    df['std_dev'] = df['close'].rolling(window=20).std()
    
    # Drop initial NaNs
    df = df.dropna().reset_index(drop=True)
    
    generated_count = 0
    
    from tqdm import tqdm
    print("Generating Signals...")
    for i in tqdm(range(len(df) - 100)): # Stop before end to have future data for labeling
        row = df.iloc[i]
        price = row['close']
        vwap = row['VWAP']
        std_dev = row['std_dev']
        
        # Update State Machine
        # Engine expects (price, vwap, std_dev, index)
        # Note: Ghosting Engine prints a lot. Might want to suppress strict printing or just let it log.
        new_state = sm.update(price, vwap, std_dev, i)
        
        if new_state.name == "TRIGGER_CANDIDATE":
            # Signal Generated!
            # 1. Capture Features
            # What features does XGBoost need? 
            # 'RSI_14', 'ADX_14', 'ATR_14', 'volatility_ratio', 'hmm_state', 'dist_to_vwap', etc.
            
            trigger_idx = i
            entry_price = price
            
            # 2. Labeling (Look Forward)
            # We want to see if price goes up 10 pips (Buy) or down (Sell)?
            # Ghosting is Mean Reversion.
            # If Price > VWAP (Upper Band touch) -> SELL (Revert to mean)
            # If Price < VWAP (Lower Band touch) -> BUY
            
            # Logic:
            # If Price > VWAP: Expect Drop. Target = Entry - 10 pips. SL = Entry + 10 pips?
            # State machine handles "First Touch". Usually "Upper Band" means Price High.
            # So Signal is SELL.
            
            is_sell = price > vwap
            
            target_hit = False
            sl_hit = False
            
            # Look ahead up to 60 minutes
            future_window = df.iloc[trigger_idx+1 : trigger_idx+61]
            
            outcome = 0 # 0 = Lose/Stagnant, 1 = Win
            
            for _, future_row in future_window.iterrows():
                future_price = future_row['close']
                
                if is_sell:
                    if future_price <= (entry_price - TP_PIPS):
                        outcome = 1
                        target_hit = True
                        break
                    if future_price >= (entry_price + SL_PIPS):
                        outcome = 0
                        sl_hit = True
                        break
                else: # Buy
                    if future_price >= (entry_price + TP_PIPS):
                        outcome = 1
                        target_hit = True
                        break
                    if future_price <= (entry_price - SL_PIPS):
                        outcome = 0
                        sl_hit = True
                        break
            
            # 3. Store Data
            # Collecting relevant features
            feat_vector = {
                'RSI_14': row.get('RSI_14', 50),
                'ADX_14': row.get('ADX_14', 20),
                'ATR_14': row.get('ATR_14', 0.0010),
                'volatility_ratio': row.get('volatility_ratio', 1.0),
                'hmm_state': row['hmm_state'],
                'z_score_20': row.get('z_score_20', 0),
                'hour': row.get('hour', 0),
                'outcome': outcome
            }
            signals.append(feat_vector)
            generated_count += 1
            
            # Reset Machine logic is mostly automatic in update(), 
            # but TRIGGER_CANDIDATE auto-resets to IDLE in next update.
            
    logging.info(f"Generated {generated_count} signals from history.")
    return pd.DataFrame(signals)

def train_xgboost(signals_df: pd.DataFrame):
    """
    Trains XGBoost on labeled signals.
    """
    if signals_df.empty:
        logging.warning("No signals passed to XGBoost training.")
        return None

    logging.info("Training XGBoost Model...")
    
    X = signals_df.drop(columns=['outcome'])
    y = signals_df['outcome']
    
    # Basic Check
    if len(y.unique()) < 2:
        logging.warning("Need both classes (Win/Loss) to train. Skipping XGBoost.")
        return None
        
    xgb_model = TradeJudgeXGB()
    try:
        xgb_model.train(X, y)
        xgb_model.save_model(MODELS_DIR / "xgb_model.json")
        logging.info("XGBoost Model Saved.")
        return xgb_model
    except Exception as e:
        logging.error(f"XGBoost Training Failed: {e}")
        return None

def train_transformer_model(df_eur: pd.DataFrame, df_gbp: pd.DataFrame):
    """
    Trains the Lead/Lag Transformer.
    """
    logging.info("Training Transformer (Lead/Lag Vision)...")
    
    # Sync Data
    # Inner join on time
    df = pd.merge(df_eur[['time', 'log_ret']], df_gbp[['time', 'log_ret']], on='time', suffixes=('_eur', '_gbp'))
    df = df.sort_values('time').reset_index(drop=True)
    
    logging.info(f"Merged Data Length: {len(df)}")
    if not df.empty:
        logging.info(f"Sample Data:\n{df.head()}")
    
    if df.empty:
        logging.error("No overlapping data for EUR/GBP!")
        return None
        
    # LIMIT DATA FOR SPEED (CPU Constraints) (Sprint 4.7)
    MAX_SAMPLES = 2000 
    if len(df) > MAX_SAMPLES:
        df = df.iloc[-MAX_SAMPLES:]
        logging.info(f"Subsampled to last {MAX_SAMPLES} rows for speed.")
        
    predictor = TransformerPredictor()
    try:
        predictor.train(df['log_ret_eur'], df['log_ret_gbp'], epochs=2)
        predictor.save_model(MODELS_DIR / "tft_model.pth")
        logging.info("Transformer Model Saved.")
        return predictor
    except Exception as e:
        logging.error(f"Transformer Training Failed: {e}")
        return None

def main():
    db = DatabaseManager()
    
    # 1. Load Data
    symbol = "EURUSD" # Primary for now
    logging.info(f"Loading data for {symbol}...")
    df = db.load_candles(symbol)
    
    if df.empty:
        logging.error("No data found in database. Run populate_db.py first.")
        return

    # 1.5 Strict Date Filter for Robust Training (Sprint 4.5)
    # Train: Feb 2025 to Sep 2025
    TRAIN_START = "2025-02-01"
    TRAIN_END = "2025-09-30"
    
    logging.info(f"Filtering Training Data: {TRAIN_START} to {TRAIN_END}")
    df = df[(df['time'] >= TRAIN_START) & (df['time'] <= TRAIN_END)].reset_index(drop=True)
    
    if df.empty:
        logging.error(f"No data in training range {TRAIN_START}-{TRAIN_END}")
        return
    
    logging.info(f"Training samples: {len(df)}")

    # 2. Train HMM
    # hmm_model = train_hmm(df)
    
    # 3. Generate Labels for XGBoost
    # signals_df = generate_signals_and_labels(df, hmm_model)
    
    # 4. Train XGBoost
    # train_xgboost(signals_df)
    
    # 4. Train XGBoost (Duplicate removed)
    # train_xgboost(signals_df)
    
    # 5. Train Transformer (Sprint 4.7)
    logging.info("--- Training Transformer ---")
    symbol_sec = "GBPUSD"
    logging.info(f"Loading data for {symbol_sec}...")
    df_gbp = db.load_candles(symbol_sec)
    
    if not df_gbp.empty:
        # Filter (Reuse same range)
        df_gbp = df_gbp[(df_gbp['time'] >= TRAIN_START) & (df_gbp['time'] <= TRAIN_END)].reset_index(drop=True)
        if not df_gbp.empty:
             train_transformer_model(df, df_gbp)
        else:
             logging.warning("GBPUSD data empty after filtering.")
    else:
        logging.warning("GBPUSD data not found.")
        
    logging.info("Training Pipeline Completed.")

if __name__ == "__main__":
    main()
