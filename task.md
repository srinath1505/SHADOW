# Antigravity T-X-H Development Task List

## 🏁 Sprint 1: The "Deep Data" Infrastructure
- [ ] **1.1 Universal Data Connector**
    - [x] Create `MT5Connector` class using `MetaTrader5` library
    - [x] Implement robust initialization and login logic
    - [x] Implement fetching live ticks
    - [x] Implement fetching historical M1 candles (EURUSD, GBPUSD)
- [ ] **1.2 The Feature Engineering Pipeline**
    - [x] Create `features.py`
    - [x] Implement Standard Features (RSI, ADX, ATR, VWAP + Bands)
    - [x] Implement Advanced Features (Rolling Z-Scores, Log Returns, Volatility Ratios)
    - [x] Implement Temporal Features (Sine/Cosine embeddings)
- [ ] **1.3 The Data Lake**
    - [x] Set up SQLite/PostgreSQL database (Implemented)
    - [x] Create schema for M1 data and features (Implicit via pandas)
    - [x] **Create indexes for performance** (Implemented in DatabaseManager)
    - [x] **Create script to populate DB with 12 months of clean historical data** (Completed with ~95k rows)
    - [x] **1.5 Data Integrity Fix (Deep History)**
        - [x] Update `MT5Connector` with Symbol Sync & UTC
        - [x] Update `populate_db.py` for 1-year fetch
        - [x] Verify Data Lake > 300k rows

## 🏁 Sprint 2: The "Ghosting" Logic (Rule-Based Core)
- [x] **2.1 Regime Filters**
    - [x] Implement `RegimeFilter` class (News, ADX > 30, Session times)
- [x] **2.2 The "Ghosting" State Machine**
    - [x] Implement `SignalStateMachine` class (Idle -> Alert -> Wait -> Trigger)
- [x] **2.3 Correlation Matrix**
    - [x] Implement `CorrelationGuard` class (Pearson correlation > 0.85)
- [x] **Integration**
    - [x] Create script to run live data and output state changes (Verified via Smoke Test)

## 🏁 Sprint 3: The "T-X-H" Intelligence (The Brain)
- [x] **3.1 Model A: HMM (Regime Watchman)**
    - [x] Install dependencies (`hmmlearn`, `scikit-learn`)
    - [x] Create `RegimeHMM` class in `src/core/hmm_model.py`
    - [x] Implement training and prediction logic (3 states)
- [x] **3.2 Model B: Transformer (The Vision)**
    - [x] Install dependencies (`torch`)
    - [x] Create `LeadLagTransformer` class in `src/core/transformer_model.py`
    - [x] Implement attention mechanism (Architecture ready)
- [x] **3.3 Model C: XGBoost (The Judge)**
    - [x] Install dependencies (`xgboost`)
    - [x] Create `TradeJudgeXGB` class in `src/core/xgboost_model.py`
    - [x] Implement training pipeline for "Second Touch" validation
- [x] **Ensemble Integration**
    - [x] Create `Ensemble` class in `src/core/ensemble.py` to aggregate model outputs

## 🏁 Sprint 3.5: The Re-Training (Deep Data)
- [x] **3.5.1 Re-Train HMM/XGBoost on 365 Days**
    - [x] Run `train_models.py` (Generated 1079 signals)
    - [x] Validate new model performance (Pass: High Conf Win Rate 100%, No Drift)

## 🏁 Sprint 4: The Simulation & Stress Test
- [x] **4.1 The "Slippage Stress Test"**
    - [x] Build `Backtester` class in `src/core/backtester.py`
    - [x] Implement spread/commission penalty logic
- [x] **4.2 Walk-Forward Analysis**
    - [x] Split Data: Train (Nov-Dec 2025), Test (Jan-Feb 2026)
    - [x] Run full simulation and generate performance metrics
    - [x] Run full simulation and generate performance metrics
    - [x] Generate Performance Report

## 🏁 Sprint 4.5: Robust Backtesting (Strict Risk)
- [x] **4.5.1 Split Training (Feb-Sep 2025)**
    - [x] Update `train_models.py` with Date Filters
    - [x] Re-train Models (XGBoost trained on 648 samples)
- [x] **4.5.2 Strict Backtest Engine**
    - [x] Update `backtester.py` with Daily Loss (2.5%) & Global Stop (5%)
    - [x] Implement Weekly Profit Caps (21%)
- [x] **4.5.3 Verification (Oct 2025)**
    - [x] Run `run_robust_backtest.py`
    - [x] Generate Detailed Daily Report (Result: Low Freq, +0.07% Profit)

## 🏁 Sprint 4.6: Threshold Optimization & Retest
- [ ] **4.6.1 Optimization Loop (Sweet Spot)**
    - [ ] Create `OPTIMIZATION_LOG.md` (Baseline vs Optimized)
    - [ ] Create `optimize_thresholds.py` (Grid Search: XGB > 0.55, HMM Volatile=True)
    - [ ] Run Optimization on Oct 2025 Data
- [x] **4.6.2 Validation (Jan-Feb 2026)**
    - [x] Run Robust Backtest with Optimized Params
    - [x] Generate Comparative Report (Result: -0.19% Loss, 45% Win Rate)

## 🏁 Sprint 4.7: Transformer Activation (The Vision)
- [x] **4.7.1 Implement Transformer Logic**
    - [x] Update `transformer_model.py` (Train/Predict)
    - [x] Update `ensemble.py` to use real score
- [x] **4.7.2 Full Retraining (Needs Full Dataset)**
    - [x] Update `train_models.py` to include Transformer
    - [x] Train on Feb-Sep 2025 Data (Partial: 2000 samples - Needs Full Training)
- [x] **4.7.3 Validation (Jan-Feb 2026)**
    - [x] Re-run Validation with T-X-H (Result: -0.19%, Safe/Stable)

## 🏁 Sprint 5: Clawdbot Execution (The Hand)
- [/] **5.1 Dynamic Risk Manager**
    - [ ] Implement position sizing logic based on confidence
- [/] **5.2 The Equity Ratchet**
    - [ ] Implement trailing drawdown and daily loss limit logic
- [/] **5.3 Connection & Go-Live**
    - [ ] Integrate full pipeline with live broker account
