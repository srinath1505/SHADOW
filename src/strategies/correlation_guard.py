import pandas as pd

class CorrelationGuard:
    def __init__(self, threshold=0.85, window=60):
        self.threshold = threshold
        self.window = window

    def calculate_correlation(self, series_a: pd.Series, series_b: pd.Series):
        """
        Calculates Pearson correlation between two price series.
        Expects series to be aligned by time index.
        """
        if len(series_a) != len(series_b):
             # Align them? Or assume caller handles alignment
             pass
             
        corr = series_a.rolling(window=self.window).corr(series_b)
        return corr.iloc[-1]

    def check_correlation(self, series_a, series_b):
        """
        Returns True if correlation is safely LOW (or whatever logic requires).
        Wait, roadmap says "If > 0.85, flag High Correlation".
        So returns True if HIGH? Or False if Safe?
        Let's return the actual value and a status flag.
        """
        current_corr = self.calculate_correlation(series_a, series_b)
        
        is_high = abs(current_corr) > self.threshold
        
        status = "HIGH" if is_high else "NORMAL"
        return current_corr, status

if __name__ == "__main__":
    # Test
    # Create valid series
    s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    s2 = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1]) # Perfectly correlated
    
    cg = CorrelationGuard(window=5)
    corr, status = cg.check_correlation(s1, s2)
    print(f"Correlation: {corr}, Status: {status}")
