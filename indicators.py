import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
import config

# Module-level functions for direct import
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator from a list of prices."""
    if len(prices) < period + 1:
        return 50  # Default to neutral if not enough data
    
    # Calculate price changes
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    
    if down == 0:  # Handle division by zero
        return 100
    
    rs = up/down
    rsi = 100. - 100./(1. + rs)
    
    # For longer price arrays, calculate the full RSI series
    if len(prices) > period + 1:
        for i in range(period+1, len(prices)):
            delta = deltas[i-1]
            
            if delta > 0:
                up_val = delta
                down_val = 0
            else:
                up_val = 0
                down_val = -delta
                
            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period
            
            if down == 0:
                rs = float('inf')
            else:
                rs = up/down
            
            rsi = 100. - 100./(1. + rs)
    
    return rsi

def calculate_ema(prices, period=20):
    """Calculate EMA indicator from a list of prices."""
    if len(prices) < period:
        return prices[-1] if len(prices) > 0 else 0
    
    return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]

class TechnicalIndicators:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        self.logger = logging.getLogger(__name__)
        
    def fetch_klines(self, symbol='BTCUSDT', interval='1m', limit=100):
        """Fetch candlestick data from Binance."""
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert price columns to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            # Convert timestamps to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching klines from Binance: {e}")
            return None
    
    def calculate_rsi(self, df, period=config.RSI_PERIOD):
        """Calculate RSI indicator."""
        close_prices = df['close'].values
        deltas = np.diff(close_prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        
        if down == 0:
            rs = float('inf')
        else:
            rs = up/down
        
        rsi = np.zeros_like(close_prices)
        rsi[:period] = 100. - 100./(1. + rs)
        
        for i in range(period, len(close_prices)):
            delta = deltas[i-1]
            
            if delta > 0:
                up_val = delta
                down_val = 0
            else:
                up_val = 0
                down_val = -delta
                
            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period
            
            if down == 0:
                rs = float('inf')
            else:
                rs = up/down
            
            rsi[i] = 100. - 100./(1. + rs)
            
        # Return the most recent RSI value
        return rsi[-1]
        
    def calculate_ema(self, df, period=config.EMA_PERIOD):
        """Calculate EMA indicator."""
        close_prices = df['close'].values
        ema = pd.Series(close_prices).ewm(span=period, adjust=False).mean().values
        return ema[-1]
    
    def get_current_price(self, symbol='BTCUSDT'):
        """Get current price from Binance."""
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return float(ticker['lastPrice'])
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching current price from Binance: {e}")
            return None
    
    def get_indicators(self, symbol='BTCUSDT'):
        """Get all indicators and current price at once."""
        df = self.fetch_klines(symbol=symbol, limit=100)
        if df is None:
            return None
        
        current_price = self.get_current_price(symbol)
        ema = self.calculate_ema(df)
        rsi = self.calculate_rsi(df)
        
        previous_price = float(df['close'].values[-2])
        price_crossed_above_ema = previous_price < ema and current_price > ema
        price_dropped_below_ema = previous_price > ema and current_price < ema
        
        return {
            'price': current_price,
            'ema': ema,
            'rsi': rsi,
            'price_crossed_above_ema': price_crossed_above_ema,
            'price_dropped_below_ema': price_dropped_below_ema
        } 