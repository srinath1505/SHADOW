from datetime import time, datetime
import pytz

class RegimeFilter:
    def __init__(self, adx_threshold=30, london_open=8, ny_close=17):
        """
        Initializes the Regime Filter.
        :param adx_threshold: Maximum ADX allowed for mean reversion (default 30).
        :param london_open: Hour of London Open (UTC, default 8).
        :param ny_close: Hour of NY Close (UTC, default 17).
        """
        self.adx_threshold = adx_threshold
        self.london_open = london_open
        self.ny_close = ny_close
        
        # Define session times (simplified for now, ideally configurable)
        self.trading_start_time = time(self.london_open, 0)
        self.trading_end_time = time(self.ny_close, 0)

    def is_news_event(self, current_time, news_feed):
        """
        Checks if a high-impact news event is near.
        :param current_time: datetime object.
        :param news_feed: List of news events (dicts with 'time' and 'impact').
        :return: True if restricted, False otherwise.
        """
        # Placeholder for news logic. 
        # Ideally, check if current_time is within +/- 30 mins of 'High' impact news.
        if not news_feed:
            return False
            
        for event in news_feed:
            if event.get('impact') == 'High':
                event_time = event.get('time')
                # Assume event_time is datetime
                delta = abs((current_time - event_time).total_seconds())
                if delta < 1800: # 30 minutes
                    return True
        return False

    def is_session_open(self, current_time: datetime):
        """
        Checks if current time is within allowed trading session (London/NY).
        """
        # UTC check
        if current_time.tzinfo is None:
             # Assume UTC for now if naive
             pass
        
        current_hour = current_time.hour
        
        # Simple check: between 8 AM and 5 PM UTC
        if self.london_open <= current_hour < self.ny_close:
            return True
        return False

    def check_conditions(self, adx_value, current_time, news_feed=None):
        """
        Main filter function.
        :return: True if SAFE TO TRADE, False if REJECT.
        """
        # 1. ADX Filter (Trend Filter)
        if adx_value > self.adx_threshold:
            print(f"Filter: ADX {adx_value:.2f} > {self.adx_threshold} (Too Trending)")
            return False

        # 2. News Filter
        if self.is_news_event(current_time, news_feed):
             print("Filter: High Impact News Event Nearby")
             return False

        # 3. Time Filter
        if not self.is_session_open(current_time):
             print(f"Filter: Outside Trading Session ({current_time.strftime('%H:%M')})")
             return False

        return True

if __name__ == "__main__":
    # Test
    rf = RegimeFilter()
    
    # Validation Cases
    print(f"Check 1 (Valid): {rf.check_conditions(25, datetime(2023, 1, 1, 10, 0))}") # Should be True
    print(f"Check 2 (High ADX): {rf.check_conditions(40, datetime(2023, 1, 1, 10, 0))}") # Should be False
    print(f"Check 3 (Bad Time): {rf.check_conditions(25, datetime(2023, 1, 1, 20, 0))}") # Should be False
