import json
import os
import time
import aiohttp
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bot_dashboard_integration")

class BotDashboardConnector:
    """
    Handles communication between the trading bot and the dashboard API.
    This class provides methods to send real-time data to the API server 
    and update the dashboard.
    """
    
    def __init__(self, api_url: str = None):
        """
        Initialize the dashboard connector.
        
        Args:
            api_url: URL of the API server. Defaults to localhost:5000 if not provided.
        """
        self.api_url = api_url or os.getenv("API_URL", "http://localhost:5000")
        logger.info(f"Dashboard connector initialized with API URL: {self.api_url}")
        
    async def send_bot_data(self, data: Dict[str, Any]) -> bool:
        """
        Send trading bot data to the API server asynchronously.
        
        Args:
            data: Dict containing trading data (signals, prices, indicators)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Add timestamp if not provided
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().isoformat()
                
            # Send data to API server
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/api/update-bot-data", json=data) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        if response_data.get("success"):
                            logger.info("Bot data successfully sent to dashboard API")
                            return True
                        else:
                            logger.warning(f"API returned error: {response_data.get('error')}")
                    else:
                        logger.error(f"Failed to send data. Status: {response.status}")
                        
            return False
        except Exception as e:
            logger.error(f"Error sending bot data to API: {str(e)}")
            return False
            
    async def send_telegram_notification(self, signal: str, binance_price: float, dex_price: float, 
                             rsi: float, ema: float, spread: float) -> bool:
        """
        Send a Telegram notification via the API server asynchronously.
        
        Args:
            signal: Trading signal (BUY, SELL, HOLD, WAIT)
            binance_price: Current BTC price on Binance
            dex_price: Current WBTC price on DEX
            rsi: Current RSI value
            ema: Current EMA value
            spread: Price spread percentage
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            notification_data = {
                "signal": signal,
                "binancePrice": binance_price,
                "dexPrice": dex_price,
                "rsi": rsi,
                "ema": ema,
                "spread": spread,
                "timestamp": datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/api/send-telegram", json=notification_data) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        if response_data.get("success"):
                            logger.info("Telegram notification sent successfully")
                            return True
                        else:
                            logger.warning(f"Failed to send Telegram notification: {response_data.get('message')}")
                    else:
                        logger.error(f"Failed to send notification. Status: {response.status}")
                        
            return False
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {str(e)}")
            return False

    async def check_api_status(self) -> Optional[Dict[str, Any]]:
    """
        Check if the API server is online and get its status.
    
    Returns:
            Dict with API status or None if unreachable
    """
    try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/status") as response:
                    if response.status == 200:
                        status_data = await response.json()
                        logger.info(f"API server is online. Version: {status_data.get('version')}")
                        return status_data
                    else:
                        logger.warning(f"API server returned status {response.status}")
                        return None
        except Exception as e:
            logger.error(f"API server appears to be offline: {str(e)}")
            return None
    
    def format_trading_data(self, signal: str, in_position: bool, buy_signal: bool, 
                          binance_price: float, dex_price: float, rsi: float, 
                          ema: float, price_difference: float) -> Dict[str, Any]:
        """
        Format trading data for the API.
        
        Args:
            Various trading parameters and indicators
            
        Returns:
            Dict: Formatted data ready to be sent to the API
        """
        return {
            "signal": signal,
            "in_position": in_position,
            "buy_signal": buy_signal,
            "binance_price": binance_price,
            "dex_price": dex_price,
            "rsi": rsi,
            "ema": ema,
            "price_difference": price_difference,
            "timestamp": datetime.now().isoformat()
        }

    def save_bot_data_locally(self, data: Dict[str, Any]) -> bool:
        """
        Save bot data to a local JSON file as backup.
        
        Args:
            data: Dictionary containing bot data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure timestamps are strings
            if "timestamp" in data and isinstance(data["timestamp"], datetime):
                data["timestamp"] = data["timestamp"].isoformat()
                
            # Save to local file
            file_path = os.path.join(os.path.dirname(__file__), "bot_data.json")
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Bot data saved locally to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving bot data locally: {str(e)}")
            return False


# Example usage in async context
async def example_usage():
    """Example of how to use the BotDashboardConnector."""
    connector = BotDashboardConnector()
    
    # Check if API is online
    status = await connector.check_api_status()
    if not status:
        logger.warning("API server is not available. Data will only be saved locally.")
    
    # Example trading data
    trading_data = connector.format_trading_data(
        signal="BUY",
        in_position=False,
        buy_signal=True,
        binance_price=65000.00,
        dex_price=64675.50,
        rsi=28.5,
        ema=64000.00,
        price_difference=-0.5
    )
    
    # Save locally as backup regardless of API availability
    connector.save_bot_data_locally(trading_data)
    
    # Send to API if available
    if status:
        await connector.send_bot_data(trading_data)
        
        # Send Telegram notification for significant signals
        if trading_data["signal"] in ["BUY", "SELL"]:
            await connector.send_telegram_notification(
                signal=trading_data["signal"],
                binance_price=trading_data["binance_price"],
                dex_price=trading_data["dex_price"],
                rsi=trading_data["rsi"],
                ema=trading_data["ema"],
                spread=trading_data["price_difference"]
            )

# Run the example if this file is executed directly
if __name__ == "__main__":
    try:
        # Run the example in an asyncio event loop
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        logger.info("Dashboard connector test stopped by user")
    except Exception as e:
        logger.error(f"Error in dashboard connector test: {str(e)}") 