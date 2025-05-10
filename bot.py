import logging
import sys
import os
from dotenv import load_dotenv

from trader import WBTCScalpTrader
import config

def setup_basic_logging():
    """Set up basic logging before full configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("main")

def check_environment():
    """Check if all required environment variables are set."""
    logger = logging.getLogger("main")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.error("No .env file found. Please create one based on .env.example")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # List of required environment variables
    required_vars = [
        'BINANCE_API_KEY', 
        'BINANCE_API_SECRET',
        'WALLET_PRIVATE_KEY',
        'WALLET_ADDRESS'
    ]
    
    # Check each required variable
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set them in your .env file")
        return False
    
    return True

def main():
    """Main entry point for the bot."""
    logger = setup_basic_logging()
    
    logger.info("WBTC Scalp Trading Bot starting up...")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create and run the trader
    try:
        trader = WBTCScalpTrader()
        
        # Print initial configuration
        logger.info("----- Bot Configuration -----")
        logger.info(f"Simulation Mode: {config.SIMULATION_MODE}")
        logger.info(f"Trade Size: {config.TRADE_SIZE_PERCENT}% of wallet")
        logger.info(f"Max Slippage: {config.MAX_SLIPPAGE_PERCENT}%")
        logger.info(f"Profit Target: {config.PROFIT_TARGET_PERCENT}%")
        logger.info(f"Trailing Stop: {config.TRAILING_STOP_PERCENT}%")
        logger.info(f"RSI Oversold: {config.RSI_OVERSOLD}")
        logger.info(f"RSI Overbought: {config.RSI_OVERBOUGHT}")
        logger.info(f"Check Interval: {config.PRICE_CHECK_INTERVAL} seconds")
        logger.info("-----------------------------")
        
        # Start trading
        trader.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 