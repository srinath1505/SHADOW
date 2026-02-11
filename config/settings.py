from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# MetaTrader 5 Configuration
MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", "C:/Program Files/MetaTrader 5/terminal64.exe")  # Path to terminal.exe

# Data Configuration
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "infrastructure" / "market_data.db"

# Symbols
SYMBOLS = ["EURUSD", "GBPUSD"]
TIMEFRAMES = ["M1", "M5"]
