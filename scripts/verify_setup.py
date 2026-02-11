import sys
from pathlib import Path
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

def check_mt5():
    print("Checking MetaTrader 5 Connection...")
    if not mt5.initialize():
        print(f"❌ MT5 Initialize Failed: {mt5.last_error()}")
        return False
    
    print(f"✅ MT5 Initialized (Version: {mt5.version()})")
    
    # Check Terminal Info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"   Path: {terminal_info.path}")
        print(f"   Connected: {terminal_info.connected}")
    
    mt5.shutdown()
    return True

def check_dependencies():
    print("\nChecking Dependencies...")
    try:
        import pandas_ta
        print("✅ pandas_ta found")
    except ImportError:
        print("❌ pandas_ta MISSING")
        
    try:
        import ta
        print("✅ ta-lib (wrapper) found")
    except ImportError:
        print("⚠️ ta-lib wrapper missing (optional if pandas_ta used)")

if __name__ == "__main__":
    check_dependencies()
    if check_mt5():
        print("\n🎉 SETUP COMPLETE. You are ready for Sprint 3.")
    else:
        print("\n⚠️ SETUP INCOMPLETE. Please install/start MT5.")
