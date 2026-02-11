from enum import Enum, auto

class GhostState(Enum):
    IDLE = auto()
    ALERT = auto()          # First Touch (Spike outside bands)
    WAITING = auto()        # Pullback (Trap set)
    TRIGGER_CANDIDATE = auto() # Second Touch (Re-test)

class SignalStateMachine:
    def __init__(self, high_band_std=2.5, pullback_std=1.5, timeout_candles=5):
        self.state = GhostState.IDLE
        self.high_band_std = high_band_std
        self.pullback_std = pullback_std
        self.timeout_candles = timeout_candles
        
        self.state_start_index = 0
        self.alert_high_price = 0.0
        self.current_index = 0

    def update(self, price, vwap, std_dev, index):
        """
        Updates the state machine with new candle data.
        :param price: Current Close price.
        :param vwap: Current VWAP value.
        :param std_dev: Current Standard Deviation value.
        :param index: Current candle index (for timeout logic).
        :return: Current State.
        """
        upper_band_limit = vwap + (self.high_band_std * std_dev)
        pullback_limit = vwap + (self.pullback_std * std_dev)
        
        self.current_index = index

        # --- State Logic ---
        
        if self.state == GhostState.IDLE:
            # Condition: Price > VWAP + 2.5σ
            if price > upper_band_limit:
                self.state = GhostState.ALERT
                self.state_start_index = index
                self.alert_high_price = price
                print(f"[{index}] State Changed: IDLE -> ALERT (Price: {price:.5f} > Band: {upper_band_limit:.5f})")

        elif self.state == GhostState.ALERT:
            # Update High Water Mark
            if price > self.alert_high_price:
                self.alert_high_price = price
            
            # Timeout Condition: Too long outside bands? (Trend)
            if (index - self.state_start_index) > self.timeout_candles:
                 self.state = GhostState.IDLE
                 print(f"[{index}] State Timeout: ALERT -> IDLE (Trend detected)")
                 return self.state

            # Condition: Price closes back inside VWAP + 1.5σ (Pullback)
            if price < pullback_limit:
                self.state = GhostState.WAITING
                print(f"[{index}] State Changed: ALERT -> WAITING (Pullback: {price:.5f} < {pullback_limit:.5f})")

        elif self.state == GhostState.WAITING:
            # Condition: Price touches the High of State 1 (± 3 pips tolerance? let's genericize to price proximity)
            # Re-test logic: Price > (Alert High - Tolerance)
            # Let's say tolerance is 0.0003 (3 pips)
            tolerance = 0.0003
            
            if price >= (self.alert_high_price - tolerance):
                self.state = GhostState.TRIGGER_CANDIDATE
                print(f"[{index}] State Changed: WAITING -> TRIGGER_CANDIDATE (Re-test: {price:.5f} ~ {self.alert_high_price:.5f})")
            
            # Reset Condition: Price drops below VWAP? invalidates setup
            if price < vwap:
                 self.state = GhostState.IDLE
                 print(f"[{index}] State Reset: WAITING -> IDLE (Price crossed VWAP)")

        elif self.state == GhostState.TRIGGER_CANDIDATE:
            # This state is ephemeral, one tick only. Reset after signal.
            # In real system, this triggers the AI Ensemble call.
            self.state = GhostState.IDLE # Reset for next cycle
            
        return self.state

if __name__ == "__main__":
    # Test Simulation
    sm = SignalStateMachine()
    
    # Mock Data series to trigger states
    vwap = 1.0800
    std = 0.0010
    # Band 2.5 = 1.0825
    # Band 1.5 = 1.0815
    
    prices = [
        1.0800, # Idle
        1.0810, # Idle
        1.0826, # Alert! (> 1.0825)
        1.0830, # Alert (New High)
        1.0820, # Alert (Still above 1.5 sig?) 1.5 sig is 1.0815. Yes.
        1.0814, # Waiting (Pullback < 1.0815)
        1.0818, # Waiting
        1.0828  # Trigger! (Re-test 1.0830 within tolerance)
    ]
    
    for i, p in enumerate(prices):
        sm.update(p, vwap, std, i)
