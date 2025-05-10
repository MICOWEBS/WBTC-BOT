"""
Advanced Trading Module

Integrates dynamic position sizing, multi-timeframe analysis, 
and detailed performance tracking into the trading strategy.
"""

import logging
import asyncio
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from binance import AsyncClient
import numpy as np  # Import numpy

# Import our custom modules
import position_sizing
import multi_timeframe
from performance_tracker import performance_tracker
import indicators
import config

# Configure logging
logger = logging.getLogger(__name__)

# Helper function to convert numpy types to Python native types
def convert_numpy_to_python(obj):
    """Convert numpy data types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_to_python(obj.tolist())
    else:
        return obj

class AdvancedTrader:
    """
    Advanced trading strategy with enhanced features:
    - Dynamic position sizing based on volatility and confidence
    - Multi-timeframe analysis for signal confirmation
    - Detailed P&L tracking and performance metrics
    """
    
    def __init__(self, wallet_balance: float = 1000.0):
        """Initialize the advanced trader with wallet balance"""
        self.wallet_balance = wallet_balance
        self.binance_client = None
        self.current_position = None
        self.is_in_position = False
        self.last_signal = "WAIT"
        self.performance_tracker = performance_tracker
        
        # Initialize performance tracker with starting balance
        self.performance_tracker.set_initial_balance(wallet_balance)
        
        logger.info(f"AdvancedTrader initialized with ${wallet_balance:.2f} balance")
    
    async def initialize_client(self) -> None:
        """Initialize Binance client for API access"""
        try:
            self.binance_client = await AsyncClient.create(
                api_key=config.BINANCE_API_KEY,
                api_secret=config.BINANCE_API_SECRET
            )
            logger.info("Binance client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {str(e)}")
    
    async def analyze_market(self, symbol: str = config.TRADING_PAIR) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis including:
        - Multi-timeframe technical indicators
        - Signal strength calculation
        - Volatility assessment
        - Confidence scoring
        
        Returns:
            Dictionary containing all analysis results
        """
        if self.binance_client is None:
            await self.initialize_client()
        
        # Get multi-timeframe analysis
        tf_data = await multi_timeframe.perform_multi_timeframe_analysis(
            self.binance_client, 
            symbol
        )
        
        # Extract primary timeframe data
        primary_tf = tf_data.get(config.PRIMARY_TIMEFRAME)
        if not primary_tf:
            logger.error(f"Missing primary timeframe data for {config.PRIMARY_TIMEFRAME}")
            return {}
        
        # Determine signal type based on primary timeframe
        signal_type = "WAIT"
        if primary_tf.is_bullish:
            signal_type = "BUY"
        elif primary_tf.is_bearish:
            signal_type = "SELL"
        
        # Calculate signal strength using multi-timeframe confirmation
        signal_strength, signal_details = multi_timeframe.calculate_signal_strength(
            tf_data, 
            signal_type
        )
        
        # Calculate volatility for position sizing
        volatility = position_sizing.calculate_volatility(
            primary_tf.close_prices,
            window=14
        )
        
        # Calculate confidence score for position sizing
        if primary_tf.close_prices and len(primary_tf.close_prices) > 0:
            current_price = primary_tf.close_prices[-1]
            ema_diff_percent = ((current_price / primary_tf.ema) - 1) * 100 if primary_tf.ema > 0 else 0
            
            # Get DEX price difference (simulated here)
            # In a real implementation, this would be fetched from DEX
            dex_price_diff = -0.3 if signal_type == "BUY" else 0.3
            
            confidence_score = position_sizing.calculate_confidence_score(
                rsi=primary_tf.rsi,
                ema_diff_percent=ema_diff_percent,
                price_spread=dex_price_diff
            )
        else:
            confidence_score = 0
        
        # Calculate appropriate position size
        position_size, position_details = position_sizing.get_position_size(
            account_balance=self.wallet_balance,
            volatility=volatility,
            confidence_score=confidence_score,
            signal_type=signal_type
        )
        
        # DEX price simulation (would be fetched from DEX in production)
        dex_price = primary_tf.close_prices[-1] * (1 + (dex_price_diff / 100)) if primary_tf.close_prices else 0
        
        # Compile all results
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "signal": signal_type,
            "signal_strength": signal_strength,
            "binance_price": primary_tf.close_prices[-1] if primary_tf.close_prices else 0,
            "dex_price": dex_price,
            "rsi": primary_tf.rsi,
            "ema": primary_tf.ema,
            "price_above_ema": primary_tf.price_above_ema,
            "volatility": volatility,
            "confidence_score": confidence_score,
            "position_size": position_size,
            "timeframe_data": {tf: {"rsi": data.rsi, "ema": data.ema} for tf, data in tf_data.items()},
            "position_sizing_details": position_details,
            "signal_details": signal_details,
        }
        
        return analysis_results
    
    async def execute_trade(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade based on the analysis results
        
        Args:
            analysis_results: Dictionary containing market analysis
            
        Returns:
            Dictionary with trade execution details
        """
        # Simulation only - in production this would interact with DEX
        if config.SIMULATION_MODE:
            return await self.simulate_trade(analysis_results)
        
        # Real trading implementation would go here
        logger.warning("Real trading mode not implemented yet")
        return {"status": "error", "message": "Real trading not implemented"}
    
    async def simulate_trade(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a trade for backtesting and demo purposes
        
        Args:
            analysis: Dictionary containing market analysis
            
        Returns:
            Dictionary with simulated trade details
        """
        signal = analysis.get("signal", "WAIT")
        current_price = analysis.get("binance_price", 0)
        position_size = analysis.get("position_size", 0)
        
        # If no change in signal, do nothing
        if signal == self.last_signal and (signal == "WAIT" or (signal == "HOLD" and self.is_in_position)):
            return {
                "status": "no_action",
                "message": f"No change in signal: {signal}",
                "timestamp": datetime.now().isoformat()
            }
        
        # Process BUY signal
        if signal == "BUY" and not self.is_in_position:
            # Calculate quantity based on position size
            quantity = position_size / current_price if current_price > 0 else 0
            
            # Record the simulated trade
            trade_data = {
                "side": "BUY",
                "entry_price": current_price,
                "quantity": quantity,
                "entry_time": datetime.now().isoformat(),
                "signal_strength": analysis.get("signal_strength", 0),
                "confidence_score": analysis.get("confidence_score", 0),
                "position_size_percent": position_size / self.wallet_balance * 100 if self.wallet_balance > 0 else 0,
                "trade_id": f"sim_{int(datetime.now().timestamp())}"
            }
            
            # Update current position
            self.current_position = trade_data
            self.is_in_position = True
            
            # Record in performance tracker
            self.performance_tracker.record_unrealized_trade(trade_data)
            
            logger.info(f"BUY SIGNAL executed: {quantity:.6f} BTC @ ${current_price:.2f}")
            
            return {
                "status": "success",
                "action": "BUY",
                "price": current_price,
                "quantity": quantity,
                "position_size": position_size,
                "timestamp": datetime.now().isoformat(),
                "trade_id": trade_data["trade_id"]
            }
        
        # Process SELL signal
        elif signal == "SELL" and self.is_in_position and self.current_position:
            entry_price = self.current_position.get("entry_price", 0)
            quantity = self.current_position.get("quantity", 0)
            trade_id = self.current_position.get("trade_id", "")
            
            # Calculate profit/loss
            profit_loss = (current_price - entry_price) * quantity
            profit_loss_percent = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            
            # Close the unrealized trade in the tracker
            self.performance_tracker.close_unrealized_trade(
                trade_id=trade_id,
                exit_price=current_price,
                exit_reason="sell_signal"
            )
            
            # Reset position
            self.current_position = None
            self.is_in_position = False
            
            # Update wallet balance
            new_balance = self.wallet_balance + profit_loss
            self.wallet_balance = new_balance
            self.performance_tracker.update_balance(new_balance)
            
            logger.info(
                f"SELL SIGNAL executed: {quantity:.6f} BTC @ ${current_price:.2f}, "
                f"P&L: ${profit_loss:.2f} ({profit_loss_percent:.2f}%)"
            )
            
            return {
                "status": "success",
                "action": "SELL",
                "price": current_price,
                "quantity": quantity,
                "profit_loss": profit_loss,
                "profit_loss_percent": profit_loss_percent,
                "timestamp": datetime.now().isoformat(),
                "trade_id": trade_id,
                "entry_price": entry_price
            }
        
        # Update unrealized trade if in position
        elif self.is_in_position and self.current_position:
            trade_id = self.current_position.get("trade_id", "")
            if trade_id:
                self.performance_tracker.update_unrealized_trade(
                    trade_id=trade_id,
                    current_price=current_price
                )
            
            return {
                "status": "holding",
                "message": "Holding current position",
                "current_price": current_price,
                "entry_price": self.current_position.get("entry_price", 0),
                "timestamp": datetime.now().isoformat()
            }
        
        # Default case - no action
        return {
            "status": "no_action",
            "message": f"No action taken for signal: {signal}",
            "timestamp": datetime.now().isoformat()
        }
    
    async def run_trading_cycle(self) -> Dict[str, Any]:
        """
        Run one complete trading cycle:
        1. Analyze market data
        2. Generate trading signal
        3. Execute trade if appropriate
        4. Update performance metrics
        
        Returns:
            Combined dictionary with analysis and trade execution results
        """
        # Ensure client is initialized
        if self.binance_client is None:
            await self.initialize_client()
        
        # Analyze market
        analysis = await self.analyze_market()
        
        if not analysis:
            return {"error": "Failed to analyze market"}
        
        # Execute trade based on analysis
        trade_result = await self.execute_trade(analysis)
        
        # Remember last signal
        self.last_signal = analysis.get("signal", "WAIT")
        
        # Combine results
        combined_result = {
            **analysis,
            "trade_result": trade_result,
            "wallet_balance": self.wallet_balance,
            "in_position": self.is_in_position,
            "performance_metrics": self.performance_tracker.metrics
        }
        
        # Convert NumPy types to native Python types for JSON serialization
        return convert_numpy_to_python(combined_result)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return self.performance_tracker.get_performance_report()
    
    async def close(self) -> None:
        """Close connections and clean up resources"""
        if self.binance_client:
            await self.binance_client.close_connection()
            logger.info("Binance client connection closed")


# Example usage
async def main():
    # Initialize trader
    trader = AdvancedTrader(wallet_balance=1000.0)
    
    # Run a few trading cycles
    for _ in range(5):
        result = await trader.run_trading_cycle()
        print(f"Signal: {result.get('signal')}, Strength: {result.get('signal_strength'):.1f}%")
        
        # Wait 5 seconds between cycles
        await asyncio.sleep(5)
    
    # Get performance report
    report = trader.get_performance_report()
    print(f"Performance: {report.get('metrics', {}).get('win_rate', 0):.1f}% win rate")
    
    # Clean up
    await trader.close()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the main function
    asyncio.run(main()) 