# SHADOW (Antigravity T-X-H)

**SHADOW** is an advanced, AI-driven algorithmic trading system designed for the forex markets (specifically EURUSD/GBPUSD). It combines a rule-based "Ghosting" engine with an ensemble of Machine Learning models to identify, validate, and execute Mean Reversion trades with high precision.

## 🚀 Project Status
- **Sprint 1 (Deep Data)**: ✅ Complete (Universal MT5 Connector, Data Lake, Feature Engineering)
- **Sprint 2 (Ghosting Logic)**: ✅ Complete (Regime Filters, State Machine, Correlation Guard)
- **Sprint 3 (The Intelligence)**: ✅ Complete (HMM, Transformer, XGBoost, Ensemble Integration)
- **Sprint 4 (Simulation)**: 🚧 Pending
- **Sprint 5 (Execution)**: ⏳ Planned

## 🧠 System Architecture

### 1. The "Ghosting" Engine (Rule-Based Core)
A state machine that tracks price action relative to VWAP bands:
- **Idle**: Price is inside bands.
- **Alert**: Price pierces the outer band (2.5 std).
- **Waiting**: Price pulls back into the "Kill Zone" (between 1.5 and 2.5 std).
- **Trigger**: Price re-tests the outer band (Second Touch).

### 2. The "Ensemble" (AI Layer)
Once a trigger is generated, it must pass the AI vote:
- **Regime Watchman (HMM)**: Filters out "Volatile/Choppy" market states.
- **The Vision (Lead/Lag Transformer)**: Analyzes relationship between EURUSD and GBPUSD.
- **The Judge (XGBoost)**: Scores the probability of the trade based on 30+ features.

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- MetaTrader 5 Terminal (Installed and Logged In)
- Windows OS (Required for MT5-Python integration)

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/srinath1505/SHADOW.git
   cd SHADOW
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment:**
   Create a `.env` file in the root directory (copy from `.env.example` if available) or set environment variables for `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`.

## 🏃 Usage

### 1. Populate Data
Fetch historical data to build the local database:
```bash
python scripts/populate_db.py
```

### 2. Train Models
Train the AI Ensemble on the populated data:
```bash
python scripts/train_models.py
```

### 3. Run Smoke Test
Verify system integrity:
```bash
python scripts/smoke_test.py
```

### 4. Live Trading (Ghosting Mode)
Run the live scanner:
```bash
python scripts/run_live_ghosting.py
```

## 📊 Features
- **Data Source**: MetaTrader 5 (Real-time & History)
- **Database**: Local SQLite with SQLAlchemy
- **ML Models**: `hmmlearn`, `xgboost`, `pytorch`
- **Safety**: Correlation Guards, News Filters (Placeholder), Regime Filters

## 📜 License
Private/Proprietary.
