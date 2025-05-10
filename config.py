import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Dynamic position sizing configuration
DYNAMIC_POSITION_SIZING = os.getenv("DYNAMIC_POSITION_SIZING", "false").lower() == "true"
BASE_POSITION_SIZE_PERCENT = float(os.getenv("TRADE_SIZE_PERCENT", 5))
MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 2.5))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", 10))
VOLATILITY_SCALING_FACTOR = float(os.getenv("VOLATILITY_SCALING_FACTOR", 1.0))
CONFIDENCE_SCALING_FACTOR = float(os.getenv("CONFIDENCE_SCALING_FACTOR", 1.0))

# Binance API configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# BSC configuration
BSC_NODE_URL = os.getenv("BSC_NODE_URL", "https://bsc-dataseed.binance.org/")
WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")

# Token addresses
WBTC_ADDRESS = os.getenv("WBTC_ADDRESS", "0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c")
WBNB_ADDRESS = os.getenv("WBNB_ADDRESS", "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c")
BUSD_ADDRESS = os.getenv("BUSD_ADDRESS", "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56")

# PancakeSwap configuration
PANCAKESWAP_ROUTER_ADDRESS = os.getenv("PANCAKESWAP_ROUTER_ADDRESS", "0x10ED43C718714eb63d5aA57B78B54704E256024E")

# Trading parameters
TRADE_SIZE_PERCENT = float(os.getenv("TRADE_SIZE_PERCENT", "5"))
MAX_SLIPPAGE_PERCENT = float(os.getenv("MAX_SLIPPAGE_PERCENT", "0.2"))
PROFIT_TARGET_PERCENT = float(os.getenv("PROFIT_TARGET_PERCENT", "0.6"))
TRAILING_STOP_PERCENT = float(os.getenv("TRAILING_STOP_PERCENT", "0.5"))

# Operation modes
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "true").lower() == "true"
AUTO_TRADING = os.getenv("AUTO_TRADING", "false").lower() == "true"

# Dashboard configuration
DASHBOARD_API_URL = os.getenv("DASHBOARD_API_URL", "http://localhost:5000/api/update-bot-data")

# Technical indicators configuration
RSI_PERIOD = 14
EMA_PERIOD = 20
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
PRICE_CHECK_INTERVAL = 300  # 5 minutes between signal checks

# Technical indicators config
RSI_PERIODS = int(os.getenv("RSI_PERIODS", 14))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", 70))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", 30))
EMA_PERIODS = int(os.getenv("EMA_PERIODS", 20))
PRICE_DIFFERENCE_THRESHOLD = float(os.getenv("PRICE_DIFFERENCE_THRESHOLD", 0.5))

# Timeframes configuration for multi-timeframe analysis
TIMEFRAMES = os.getenv("TIMEFRAMES", "5m,15m,1h").split(",")
PRIMARY_TIMEFRAME = os.getenv("PRIMARY_TIMEFRAME", "5m")
CONFIRMATION_TIMEFRAMES = [tf for tf in TIMEFRAMES if tf != PRIMARY_TIMEFRAME]
TIMEFRAME_WEIGHT = {
    "5m": 0.5,
    "15m": 0.3,
    "1h": 0.2,
}

# API credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TRADING_PAIR = os.getenv("TRADING_PAIR", "BTCUSDT")

# PnL tracking
TRACK_PERFORMANCE = os.getenv("TRACK_PERFORMANCE", "true").lower() == "true"
PERFORMANCE_LOG_FILE = os.getenv("PERFORMANCE_LOG_FILE", "performance_log.json") 