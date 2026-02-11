import pandas as pd
import pandas_ta as ta
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ingests raw OHLCV DataFrame and returns DataFrame with engineering features.
        Expected columns: 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'
        """
        # Ensure df is sorted by time
        df = df.sort_values('time').reset_index(drop=True)

        # Basic Checks
        if len(df) < 50:
            return df # Not enough data for indicators

        # --- 1. Standard Indicators ---
        # RSI(14)
        df['RSI_14'] = ta.rsi(df['close'], length=14)

        # ADX(14)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
             df = pd.concat([df, adx], axis=1) # Adds ADX_14, DMP_14, DMN_14

        # ATR(14)
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # VWAP
        # pandas_ta vwap requires high, low, close, volume. It calculates based on session usually,
        # but for M1 data across days, a rolling VWAP might be more appropriate or a session-reset one.
        # Here we use a rolling VWAP approximation or standard VWAP if session info is clear.
        # For simplicity, let's use a rolling VWAP for now or reliable TA lib function.
        # pandas_ta.vwap(high, low, close, volume, anchor="D") returns VWAP anchored to day.
        try:
             # Ensure index is datetime for anchored VWAP
             df_temp = df.set_index('time')
             vwap = ta.vwap(df_temp['high'], df_temp['low'], df_temp['close'], df_temp['tick_volume'])
             # Reset index to align
             df['VWAP'] = vwap.values
        except Exception as e:
            print(f"VWAP calculation error: {e}")
            # Fallback to simple rolling VWAP if anchored fails
            tp = (df['high'] + df['low'] + df['close']) / 3
            df['VWAP'] = (tp * df['tick_volume']).rolling(window=100).sum() / df['tick_volume'].rolling(window=100).sum()

        # VWAP Bands (User specified 2.0 and 2.5 std dev)
        # We need a standard deviation from VWAP.
        # Approximating using Rolling STD of Close
        rolling_std = df['close'].rolling(window=20).std() # Using 20 as proxy for immediate volatility
        if 'VWAP' in df.columns:
            df['VWAP_Upper_2.0'] = df['VWAP'] + (2.0 * rolling_std)
            df['VWAP_Lower_2.0'] = df['VWAP'] - (2.0 * rolling_std)
            df['VWAP_Upper_2.5'] = df['VWAP'] + (2.5 * rolling_std)
            df['VWAP_Lower_2.5'] = df['VWAP'] - (2.5 * rolling_std)

        # --- 2. Advanced Features ---
        # Log Returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        # Rolling Z-Score of Close (Window 20)
        df['z_score_20'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()

        # Volatility Ratio (ATR / Rolling ATR Mean)
        # feat_volatility: ATR(14) / RollingMean(ATR, 100)
        df['ATR_100_MA'] = df['ATR_14'].rolling(window=100).mean()
        df['volatility_ratio'] = df['ATR_14'] / df['ATR_100_MA']
        
        # Stretch: (Close - VWAP) / StdDev (already similar to z-score but relative to VWAP)
        if 'VWAP' in df.columns:
             df['stretch'] = (df['close'] - df['VWAP']) / rolling_std

        # --- 3. Temporal Features ---
        # Sine/Cosine of Hour and Minute
        # time column is already datetime
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        
        df['time_hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['time_hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['time_min_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['time_min_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

        # Cleanup intermediate columns if needed
        # df.drop(columns=['hour', 'minute'], inplace=True) 

        return df.dropna()

if __name__ == "__main__":
    # Test with dummy data
    data = {
        'time': pd.date_range(start='2023-01-01', periods=200, freq='1min'),
        'open': np.random.rand(200) + 100,
        'high': np.random.rand(200) + 101,
        'low': np.random.rand(200) + 99,
        'close': np.random.rand(200) + 100,
        'tick_volume': np.random.randint(1, 100, 200),
        'spread': np.zeros(200),
        'real_volume': np.zeros(200)
    }
    df = pd.DataFrame(data)
    
    fe = FeatureEngineer()
    df_features = fe.calculate_features(df)
    print(df_features.tail())
    print(df_features.columns)
