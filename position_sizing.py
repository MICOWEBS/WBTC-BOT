"""
Position Sizing Module for Trading Bot

This module handles dynamic position sizing based on volatility metrics,
confidence scores, and customizable risk parameters.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from config import (
    DYNAMIC_POSITION_SIZING,
    BASE_POSITION_SIZE_PERCENT,
    MIN_POSITION_SIZE_PERCENT, 
    MAX_POSITION_SIZE_PERCENT,
    VOLATILITY_SCALING_FACTOR,
    CONFIDENCE_SCALING_FACTOR
)

# Configure logging
logger = logging.getLogger(__name__)

def calculate_volatility(prices: List[float], window: int = 14) -> float:
    """
    Calculate price volatility using standard deviation of returns
    
    Args:
        prices: List of historical prices
        window: Window size for calculating volatility
        
    Returns:
        Volatility as percentage
    """
    if len(prices) < window + 1:
        return 0.0
    
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns[-window:]) * np.sqrt(window)  # Annualized
    return volatility * 100  # Convert to percentage

def calculate_confidence_score(
    rsi: float, 
    ema_diff_percent: float, 
    price_spread: float,
    volume_ratio: float = 1.0,
    price_momentum: float = 0.0
) -> float:
    """
    Calculate a confidence score (0-100) based on multiple indicators
    
    Args:
        rsi: Current RSI value
        ema_diff_percent: Percentage difference between price and EMA
        price_spread: Price difference between DEX and Binance as percentage
        volume_ratio: Current volume to average volume ratio (optional)
        price_momentum: Short-term price momentum indicator (optional)
        
    Returns:
        Confidence score (0-100)
    """
    # Confidence from RSI (higher near extremes)
    if rsi <= 30:
        rsi_confidence = 100 - (rsi * 3)  # 10 → 70, 30 → 10
    elif rsi >= 70:
        rsi_confidence = (rsi - 70) * 3  # 70 → 0, 90 → 60
    else:
        rsi_confidence = 0  # Neutral zone has no confidence
    
    # Confidence from EMA distance (stronger signal when price is further from EMA)
    ema_confidence = min(abs(ema_diff_percent) * 20, 100)
    
    # Confidence from price spread (higher is better)
    spread_confidence = min(abs(price_spread) * 40, 100)  # 1% spread → 40 confidence
    
    # Confidence from volume (higher volume is better signal)
    volume_confidence = min(volume_ratio * 50, 100)
    
    # Confidence from momentum (aligned with trade direction)
    momentum_confidence = min(abs(price_momentum) * 30, 100)
    
    # Weighted average of all factors
    confidence = (
        (rsi_confidence * 0.3) + 
        (ema_confidence * 0.2) + 
        (spread_confidence * 0.3) + 
        (volume_confidence * 0.1) + 
        (momentum_confidence * 0.1)
    )
    
    return max(0, min(confidence, 100))  # Ensure between 0-100

def get_position_size(
    account_balance: float,
    volatility: float,
    confidence_score: float,
    signal_type: str = "BUY"
) -> Tuple[float, Dict]:
    """
    Calculate the position size based on account balance, market volatility and confidence
    
    Args:
        account_balance: Total account balance in quote currency
        volatility: Market volatility as percentage
        confidence_score: Signal confidence score (0-100)
        signal_type: Type of trading signal (BUY/SELL)
        
    Returns:
        Tuple of (position_size_in_quote_currency, position_sizing_details)
    """
    if not DYNAMIC_POSITION_SIZING:
        position_size = account_balance * (BASE_POSITION_SIZE_PERCENT / 100)
        return position_size, {
            "position_size_percent": BASE_POSITION_SIZE_PERCENT,
            "reason": "Static position sizing (dynamic sizing disabled)"
        }
    
    # Base position size from configuration
    base_size_percent = BASE_POSITION_SIZE_PERCENT
    
    # Adjust for volatility (lower position size for higher volatility)
    volatility_factor = 1.0
    if volatility > 0:
        # Normalize volatility (consider 1% daily volatility as baseline)
        normalized_volatility = volatility / 1.0
        volatility_factor = 1.0 / (normalized_volatility ** VOLATILITY_SCALING_FACTOR)
    
    # Adjust for confidence (higher position size for higher confidence)
    confidence_factor = 0.5 + ((confidence_score / 100) * CONFIDENCE_SCALING_FACTOR)
    
    # Calculate adjusted position size percentage
    adjusted_size_percent = base_size_percent * volatility_factor * confidence_factor
    
    # Apply min/max constraints
    final_size_percent = max(MIN_POSITION_SIZE_PERCENT, 
                             min(adjusted_size_percent, MAX_POSITION_SIZE_PERCENT))
    
    # Calculate actual position size in quote currency
    position_size = account_balance * (final_size_percent / 100)
    
    # Log for transparency
    sizing_details = {
        "base_size_percent": base_size_percent,
        "volatility": volatility,
        "volatility_factor": volatility_factor,
        "confidence_score": confidence_score,
        "confidence_factor": confidence_factor,
        "adjusted_size_percent": adjusted_size_percent,
        "final_size_percent": final_size_percent,
        "account_balance": account_balance,
        "position_size": position_size
    }
    
    logger.info(
        f"Dynamic position sizing: {final_size_percent:.2f}% of balance "
        f"(Volatility: {volatility:.2f}%, Confidence: {confidence_score:.1f}/100)"
    )
    
    return position_size, sizing_details 