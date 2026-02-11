import numpy as np
import joblib
from hmmlearn.hmm import GaussianHMM
import pandas as pd
from pathlib import Path

class RegimeHMM:
    def __init__(self, n_components=3, covariance_type="full", n_iter=100, random_state=42):
        """
        Initializes the Hidden Markov Model for Regime Detection.
        :param n_components: Number of hidden states (default 3: Calm, Trend, Volatile).
        :param covariance_type: Type of covariance parameters to use.
        """
        self.model = GaussianHMM(
            n_components=n_components, 
            covariance_type=covariance_type, 
            n_iter=n_iter,
            random_state=random_state,
            verbose=False
        )
        self.is_fitted = False
        
        # Mapping states to human-readable labels (requires analysis after training)
        # For now, we just identify them by ID.
        self.state_labels = {0: "State 0", 1: "State 1", 2: "State 2"}

    def prepare_features(self, df: pd.DataFrame):
        """
        Extracts relevant features for HMM.
        Expected features: 'ATR_14', 'volatility_ratio', 'log_ret' (absolute?)
        We focus on Volatility and Trend Strength.
        """
        # Feature Engineering check
        required = ['ATR_14', 'volatility_ratio', 'log_ret']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing feature: {col}")
        
        # Use log returns (absolute values for volatility magnitude) and ATR ratio
        X = df[required].copy()
        X['log_ret'] = X['log_ret'].abs() # We care about magnitude of returns for volatility regime
        
        # Fill NaNs
        X = X.dropna()
        return X.values

    def fit(self, df: pd.DataFrame):
        """
        Trains the HMM.
        """
        X = self.prepare_features(df)
        self.model.fit(X)
        self.is_fitted = True
        print(f"HMM trained on {len(X)} samples.")
        
        # Basic analysis to label states?
        # Typically:
        # State with lowest Variance = Calm
        # State with highest Variance = Volatile/Chaos
        # Intermediate = Trend/Normal
        
        # We can inspect model.means_ and model.covars_
        # But for now let's just train.
        return self

    def predict(self, df: pd.DataFrame):
        """
        Predicts the hidden state for each sample.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
            
        X = self.prepare_features(df)
        states = self.model.predict(X)
        return states

    def predict_proba(self, df: pd.DataFrame):
        """
        Returns probabilities of each state for the samples.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        X = self.prepare_features(df)
        probs = self.model.predict_proba(X)
        return probs

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)
        print(f"HMM model saved to {filepath}")

    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        self.is_fitted = True
        print(f"HMM model loaded from {filepath}")

if __name__ == "__main__":
    # Test
    # Create dummy data
    data = {
        'ATR_14': np.random.rand(100) * 0.001,
        'volatility_ratio': np.random.rand(100) + 0.5,
        'log_ret': np.random.normal(0, 0.001, 100)
    }
    df = pd.DataFrame(data)
    
    hmm = RegimeHMM()
    hmm.fit(df)
    states = hmm.predict(df)
    print("Predicted States:", states[:10])
    
    probs = hmm.predict_proba(df)
    print("State Probabilities:", probs[:2])
