from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import Models
from src.core.hmm_model import RegimeHMM
from src.core.transformer_model import LeadLagTransformer, TransformerPredictor
from src.core.xgboost_model import TradeJudgeXGB

@dataclass
class EnsembleSignal:
    timestamp: str
    signal_id: str
    pair: str
    action: str
    hmm_state: int
    hmm_confidence: float # Probability of current state
    transformer_score: float
    xgboost_prob: float
    final_status: str # "APPROVED" or "REJECTED"

class AntigravityEnsemble:
    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / "models"
        
        # Initialize Models (Training required before real use)
        # In production, these would be loaded from files.
        self.hmm = RegimeHMM() 
        self.transformer = TransformerPredictor()
        self.judge = TradeJudgeXGB()
        
        # Flags to check if models differ loaded
        self.ready = False 
        
        # Dynamic Configuration for Optimization (Sprint 4.6)
        self.active_config = {
            'xgb_threshold': 0.65, # Default Conservative
            'allow_volatile': False # Default Conservative
        }

    def load_models(self):
        """
        Loads pre-trained models.
        """
        try:
            self.hmm.load_model(self.models_dir / "hmm_model.pkl")
            self.transformer.load_model(self.models_dir / "tft_model.pth") 
            self.judge.load_model(self.models_dir / "xgb_model.json")
            self.ready = True
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Failed to load models: {e}")
            self.ready = False

    def check_signal(self, candle_data: pd.DataFrame, signal_context: dict, secondary_candle_data: pd.DataFrame = None) -> EnsembleSignal:
        """
        Evaluates a signal candidate.
        :param candle_data: DataFrame with features.
        :param check_signal: DataFrame of secondary pair (GBPUSD).
        :param signal_context: Dict with 'pair', 'action', 'timestamp', 'signal_id'.
        """
        # 1. HMM Judgment
        # In real inference, we'd predict on latest window.
        # Check if fitted, else mock.
        if self.hmm.is_fitted:
            hmm_state = self.hmm.predict(candle_data)[-1]
            hmm_probs = self.hmm.predict_proba(candle_data)[-1]
            hmm_conf = hmm_probs[hmm_state]
        else:
            hmm_state = 0 # Calm
            hmm_conf = 0.85 # Mock

        # 2. Transformer Judgment (Sprint 4.7)
        trans_score = 0.5 # Neutral
        
        if secondary_candle_data is not None and len(secondary_candle_data) >= 60:
             # Compute Log Returns on the fly
             # We assume data is sorted and continuous
             eur = np.log(candle_data['close'] / candle_data['close'].shift(1)).fillna(0).values
             gbp = np.log(secondary_candle_data['close'] / secondary_candle_data['close'].shift(1)).fillna(0).values
             
             # Predict
             trans_score = self.transformer.predict(eur, gbp)

        # 3. XGBoost Judgment
        # Feature Vector: Features + HMM + Trans
        # Extract last row features
        # features = candle_data.iloc[-1:].copy() # This keeps all columns including time
        
        # We must construct the exact feature vector used in training
        # Training features: ['RSI_14', 'ADX_14', 'ATR_14', 'volatility_ratio', 'hmm_state', 'z_score_20', 'hour']
        
        input_data = pd.DataFrame([{
            'RSI_14': candle_data['RSI_14'].iloc[-1],
            'ADX_14': candle_data['ADX_14'].iloc[-1],
            'ATR_14': candle_data['ATR_14'].iloc[-1],
            'volatility_ratio': candle_data['volatility_ratio'].iloc[-1],
            'hmm_state': int(hmm_state),
            'z_score_20': candle_data['z_score_20'].iloc[-1],
            'hour': candle_data['hour'].iloc[-1]
        }])
        
        # Add model outputs as features for XGBoost if used in training
        # features['hmm_state'] = hmm_state
        # features['trans_score'] = trans_score # Not used in training yet
        
        # Valid columns? XGBoost very strict.
        if self.judge.is_fitted:
            xgb_prob = self.judge.predict_proba(input_data)[0]
        else:
            xgb_prob = 0.72 # Mock

        # 4. Final Decision Rule
        # Dynamic Thresholds (Sprint 4.6)
        xgb_threshold = self.active_config.get('xgb_threshold', 0.65)
        allow_volatile = self.active_config.get('allow_volatile', False)
        
        status = "APPROVED"
        action = signal_context.get('action', 'CHECK')
        
        # Vision Filter (Sprint 4.7)
        # 0.55 / 0.45 Buffers
        if action == "BUY" and trans_score < 0.45:
             status = "REJECTED_VISION"
        elif action == "SELL" and trans_score > 0.55:
             status = "REJECTED_VISION"
        
        if hmm_state == 2 and not allow_volatile: # Volatile
            status = "REJECTED_REGIME"
        elif xgb_prob < xgb_threshold:
            status = "REJECTED_SCORE"

        return EnsembleSignal(
            timestamp=signal_context.get('timestamp'),
            signal_id=signal_context.get('signal_id', 'N/A'),
            pair=signal_context.get('pair'),
            action=signal_context.get('action'),
            hmm_state=hmm_state,
            hmm_confidence=hmm_conf,
            transformer_score=trans_score,
            xgboost_prob=xgb_prob,
            final_status=status
        )

if __name__ == "__main__":
    # Test
    ensemble = AntigravityEnsemble()
    ensemble.load_models()
    
    # Dummy Data
    df = pd.DataFrame({
        'ATR_14': [0.001],
        'volatility_ratio': [1.0],
        'log_ret': [0.0001],
        'RSI_14': [50],
        # Add other features XGB expects
    })
    
    ctx = {
        'timestamp': "2026-02-12T10:00:00",
        'signal_id': "TEST-1",
        'pair': "EURUSD",
        'action': "BUY"
    }
    
    result = ensemble.check_signal(df, ctx)
    print("Ensemble Result:", result)
