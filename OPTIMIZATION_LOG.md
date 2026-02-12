# Optimization Log: Finding the "Sweet Spot" 🎯

## 1. Baseline Configuration (Oct 2025)
**Status:** 🛡️ Too Conservative
-   **XGBoost Threshold:** > 0.65
-   **HMM Regime Filter:** REJECT Volatile State (State 2)
-   **Risk Rules:** Daily Limit 2.5%, Weekly Cap 21%, Global Stop 5%

### Baseline Results (Oct 1 - Oct 31, 2025)
-   **Total Trades:** 3
-   **Net Profit:** +0.07% ($7.42)
-   **Max Drawdown:** 0.14%
-   **Win Rate:** 66.67%
-   **Conclusion:** Failed to meet 7% Weekly Profit Target due to lack of volume.

---

## 2. Optimization Strategy (Grid Search)
We will test the following permutations on the **Oct 2025** dataset:

| ID | XGBoost Threshold | HMM Volatile Allowed? | Hypothesis |
| :--- | :--- | :--- | :--- |
| **A** | 0.60 | No | Mild relaxation to catch "near miss" high quality trades. |
| **B** | 0.55 | No | Significant relaxation. Should double trade volume. |
| **C** | 0.65 | **Yes** | High confidence only, but allows trading in chopped markets. |
| **D** | 0.60 | **Yes** | Balanced approach. Moderate confidence, all regimes. |
| **E** | 0.55 | **Yes** | Aggressive. Maximum volume. Risk of higher drawdown. |

### Target Metrics
-   **Weekly Profit:** > 7% (primary goal)
-   **Max Drawdown:** < 5% (hard constraint)
-   **Win Rate:** > 55%

---

## 3. Experiment Results

| Config | Trades | Profit % | Max DD % | Win Rate % | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGB_0.65_Vol_False** | 3 | **+0.078%** | 0.14% | 66.7% | 🛡️ Safe but Slow |
| **XGB_0.65_Vol_True** | 3 | +0.078% | 0.14% | 66.7% | Same as Baseline |
| **XGB_0.50_Vol_True** | **5** | +0.058% | 0.30% | 60.0% | 🚀 **Highest Vol** |
| **XGB_0.55_Vol_False** | 4 | -0.034% | 0.28% | 50.0% | ❌ Loss |
| **XGB_0.60_Vol_True** | 4 | -0.034% | 0.28% | 50.0% | ❌ Loss |

## 4. "Sweet Spot" Selection 🏆
**Selected Config:** `XGB_0.50` + `Allow Volatile`

**Reasoning:**
-   Baseline (0.65) is profitable but only took 3 trades in a month.
-   Aggressive (0.50 + Volatile) increased volume to **5 trades** while maintaining profitability (+0.058%) and acceptable drawdown (0.30%).
-   Intermediate settings (0.55-0.60) actually performed worse (-0.03% loss), suggesting that some "lower confidence" trades detected by 0.50 were winners that cushioned the losers.

**Next Step:**
Apply `XGB > 0.50` and `Volatile=True` to the **Jan-Feb 2026** Validation Test to see if improved frequency holds up.
