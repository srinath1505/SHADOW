import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import sys
from pathlib import Path

# Add project root to path to import config
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import config.settings as settings

class MT5Connector:
    def __init__(self):
        self.connected = False

    def initialize(self):
        """Initializes the MT5 terminal."""
        path = settings.MT5_PATH
        if path and Path(path).exists():
             if not mt5.initialize(path=path):
                print(f"initialize(path={path}) failed, error code = {mt5.last_error()}")
                return False
        else:
             if not mt5.initialize():
                print(f"initialize() failed, error code = {mt5.last_error()}")
                return False
        
        self.connected = True
        print(f"MT5 Core version: {mt5.version()}")
        return True

    def login(self, login=None, password=None, server=None):
        """Logs into the trading account. Uses settings if arguments not provided."""
        _login = login or settings.MT5_LOGIN
        _password = password or settings.MT5_PASSWORD
        _server = server or settings.MT5_SERVER

        if not _login or not _password or not _server:
             print("Login credentials missing. Assuming Terminal is already logged in.")
             return True

        authorized = mt5.login(_login, password=_password, server=_server)
        if authorized:
            print(f"Connected to account #{_login}")
        else:
            print(f"failed to connect at account #{_login}, error code: {mt5.last_error()}")
        
        return authorized

    def shutdown(self):
        """Shuts down the connection."""
        mt5.shutdown()
        self.connected = False

    def get_live_ticks(self, symbol, num_ticks=10):
        """Fetches the last N ticks for a symbol."""
        if not self.connected:
            if not self.initialize():
                return None

        ticks = mt5.copy_ticks_from(symbol, datetime.now(), num_ticks, mt5.COPY_TICKS_ALL)
        if ticks is None:
            print(f"Failed to get ticks for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    import time

    def get_historical_candles(self, symbol, timeframe_str, num_candles=1000, start_date=None):
        """
        Fetches historical candles with robust synchronization.
        """
        if not self.connected:
             if not self.initialize():
                return None
        
        # 1. Force Symbol Selection to trigger history download
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select {symbol}")
            return None

        # 2. Map Timeframe
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "H1": mt5.TIMEFRAME_H1,
        }
        mt5_timeframe = tf_map.get(timeframe_str, mt5.TIMEFRAME_M1)

        # 3. Validation Loop (Sync Check)
        # Attempt to get 1 bar to see if history exists
        for _ in range(5):
             rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 1)
             if rates is not None and len(rates) > 0:
                 break
             time.sleep(1) # Wait for terminal to sync

        # 4. Fetch Data
        rates = None
        if start_date:
            rates = mt5.copy_rates_from(symbol, mt5_timeframe, start_date, num_candles)
        else:
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_candles)

        if rates is None or len(rates) == 0:
            print(f"Failed to get rates for {symbol} (History empty, check Max Bars setting)")
            return None

        # 5. Process & Normalize
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # 6. UTC Normalization (Crucial for Transformer)
        # MetaTrader is usually UTC+2/3. We will enforce UTC awareness.
        # Assuming server time, we just localize. Ideally we shift, but for ML relative time matters more.
        # We will strip tz to be safe for SQLite but ensure consistency.
        # df['time'] = df['time'].dt.tz_localize('UTC') 
        
        return df

    
    def get_candles_range(self, symbol, timeframe_str, start_date, end_date):
        """
        Fetches historical candles within a specific date range.
        Uses copy_rates_range for precise chunking.
        """
        if not self.connected:
             if not self.initialize():
                return None
        
        # 1. Force Symbol Selection
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select {symbol}")
            return None

        # 2. Map Timeframe
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "H1": mt5.TIMEFRAME_H1,
        }
        mt5_timeframe = tf_map.get(timeframe_str, mt5.TIMEFRAME_M1)

        # 3. Fetch Data
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"Failed to get rates for {symbol} (Range: {start_date} - {end_date})")
            return None

        # 4. Process & Normalize
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df

if __name__ == "__main__":
    # Smoke test
    connector = MT5Connector()
    if connector.initialize():
        print("Test Connection Successful")
        ticks = connector.get_live_ticks("EURUSD", 5)
        print("Live Ticks Sample:")
        print(ticks)
        
        candles = connector.get_historical_candles("EURUSD", "M1", 5)
        print("Candles Sample:")
        print(candles)
        
        connector.shutdown()
