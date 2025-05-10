"""
Multi-Timeframe Analysis Module

Provides functionality to analyze trading signals across multiple timeframes
to increase signal reliability through higher timeframe confirmation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from binance import AsyncClient
import asyncio
from config import (
    TIMEFRAMES, 
    PRIMARY_TIMEFRAME,
    CONFIRMATION_TIMEFRAMES,
    TIMEFRAME_WEIGHT,
    RSI_PERIODS,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    EMA_PERIODS
)
import indicators

# Configure logging
logger = logging.getLogger(__name__)

class TimeframeData:
    """Store analysis data for a specific timeframe"""
    
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.close_prices = []
        self.rsi = 50.0
        self.ema = 0.0
        self.price_above_ema = False
        self.is_bullish = False
        self.is_bearish = False
        self.last_updated = None

async def fetch_timeframe_data(client: AsyncClient, symbol: str, timeframe: str, limit: int = 100) -> List:
    """Fetch candlestick data for a specific timeframe"""
    klines = await client.get_klines(
        symbol=symbol,
        interval=timeframe,
        limit=limit
    )
    return klines

async def analyze_timeframe(client: AsyncClient, symbol: str, timeframe: str) -> TimeframeData:
    """Perform technical analysis for a specific timeframe"""
    tf_data = TimeframeData(timeframe)
    
    try:
        # Fetch candlestick data
        klines = await fetch_timeframe_data(client, symbol, timeframe, limit=100)
        
        if not klines or len(klines) < RSI_PERIODS + 1:
            logger.warning(f"Not enough data for {timeframe} analysis")
            return tf_data
        
        # Extract close prices
        tf_data.close_prices = [float(k[4]) for k in klines]
        current_price = tf_data.close_prices[-1]
        
        # Calculate indicators
        tf_data.rsi = indicators.calculate_rsi(tf_data.close_prices, RSI_PERIODS)
        tf_data.ema = indicators.calculate_ema(tf_data.close_prices, EMA_PERIODS)
        
        # Determine if price is above EMA
        tf_data.price_above_ema = current_price > tf_data.ema
        
        # Determine if timeframe is bullish/bearish
        tf_data.is_bullish = tf_data.rsi < RSI_OVERSOLD and tf_data.price_above_ema
        tf_data.is_bearish = tf_data.rsi > RSI_OVERBOUGHT and not tf_data.price_above_ema
        
        tf_data.last_updated = datetime.now()
        
        logger.info(
            f"{timeframe} Analysis: RSI={tf_data.rsi:.1f}, "
            f"EMA=${tf_data.ema:.2f}, Price=${current_price:.2f} "
            f"({'Above' if tf_data.price_above_ema else 'Below'} EMA)"
        )
        
    except Exception as e:
        logger.error(f"Error analyzing {timeframe} timeframe: {str(e)}")
    
    return tf_data

async def perform_multi_timeframe_analysis(client: AsyncClient, symbol: str) -> Dict[str, TimeframeData]:
    """Analyze multiple timeframes and return consolidated results"""
    results = {}
    
    # Create tasks for all timeframe analyses
    tasks = [analyze_timeframe(client, symbol, tf) for tf in TIMEFRAMES]
    
    # Run all analyses concurrently
    timeframe_data_list = await asyncio.gather(*tasks)
    
    # Map results by timeframe
    for tf_data in timeframe_data_list:
        results[tf_data.timeframe] = tf_data
    
    return results

def calculate_signal_strength(timeframe_data: Dict[str, TimeframeData], signal_type: str) -> Tuple[float, Dict]:
    """Calculate the strength of a signal based on multi-timeframe confirmation"""
    if not timeframe_data or PRIMARY_TIMEFRAME not in timeframe_data:
        return 0.0, {"error": "Primary timeframe data missing"}
    
    # Check if we have a valid signal in the primary timeframe
    primary_tf = timeframe_data[PRIMARY_TIMEFRAME]
    
    # Primary timeframe must confirm the signal
    if signal_type == "BUY" and not primary_tf.is_bullish:
        return 0.0, {"reason": f"No buy signal in primary timeframe {PRIMARY_TIMEFRAME}"}
    
    if signal_type == "SELL" and not primary_tf.is_bearish:
        return 0.0, {"reason": f"No sell signal in primary timeframe {PRIMARY_TIMEFRAME}"}
    
    # Count confirmations from other timeframes
    confirmations = {}
    confirmed_count = 0
    total_timeframes = len(TIMEFRAMES)
    
    for tf, data in timeframe_data.items():
        if signal_type == "BUY":
            confirms_signal = data.is_bullish
        else:  # SELL
            confirms_signal = data.is_bearish
            
        confirmations[tf] = {
            "confirms": confirms_signal,
            "rsi": data.rsi,
            "price_above_ema": data.price_above_ema
        }
        
        if confirms_signal:
            confirmed_count += 1
    
    # Calculate overall strength (0-100%)
    strength = (confirmed_count / total_timeframes) * 100
    
    details = {
        "primary_timeframe": PRIMARY_TIMEFRAME,
        "confirmations": confirmations,
        "confirmed_timeframes": confirmed_count,
        "total_timeframes": total_timeframes,
        "strength_percent": strength
    }
    
    return strength, details 