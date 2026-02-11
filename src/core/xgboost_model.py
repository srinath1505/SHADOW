import xgboost as xgb
import pandas as pd
import numpy as np
import joblib

class TradeJudgeXGB:
    def __init__(self, model_path=None):
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        if model_path:
            self.load_model(model_path)
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the XGBoost model.
        :param X: Feature DataFrame (including HMM state, Transformer score).
        :param y: Target Series (1 for Win, 0 for Loss).
        """
        self.model.fit(X, y)
        self.is_fitted = True
        print(f"XGBoost trained on {len(X)} samples.")
        return self

    def predict_proba(self, feature_vector: pd.DataFrame):
        """
        Returns probability of winning trade.
        """
        if not self.is_fitted:
             raise RuntimeError("Model is not fitted yet.")
        
        # XGBoost expects specific input format? 
        # Yes, ensure feature names match training.
        probs = self.model.predict_proba(feature_vector)[:, 1] # Probability of Class 1
        return probs

    def save_model(self, filepath):
        self.model.save_model(filepath)
        print(f"XGBoost model saved to {filepath}")

    def load_model(self, filepath):
        self.model.load_model(filepath)
        self.is_fitted = True
        print(f"XGBoost model loaded from {filepath}")

if __name__ == "__main__":
    # Test
    # Dummy Features: 'RSI', 'ADX', 'HMM_State', 'Trans_Score'
    data = {
        'RSI': np.random.rand(100) * 100,
        'ADX': np.random.rand(100) * 50,
        'HMM_State': np.random.randint(0, 3, 100),
        'Trans_Score': np.random.rand(100)
    }
    y = np.random.randint(0, 2, 100)
    
    df = pd.DataFrame(data)
    y_series = pd.Series(y)
    
    judge = TradeJudgeXGB()
    judge.train(df, y_series)
    
    sample = df.iloc[:1]
    prob = judge.predict_proba(sample)
    print(f"Win Probability: {prob[0]:.4f}")
