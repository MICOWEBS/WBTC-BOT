from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import json
import os
import aiohttp
import time
import numpy as np
import pandas as pd
from binance import AsyncClient
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import logging
import advanced_trader
import performance_tracker
from config import TRACK_PERFORMANCE

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_server")

# Initialize FastAPI app
app = FastAPI(title="Trading Bot API", description="Low-latency trading signals API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MAX_HISTORY = 20
RSI_PERIODS = int(os.getenv("RSI_PERIODS", 14))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", 70))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", 30))
EMA_PERIODS = int(os.getenv("EMA_PERIODS", 20))
PRICE_DIFFERENCE_THRESHOLD = float(os.getenv("PRICE_DIFFERENCE_THRESHOLD", 0.5))
TRADING_PAIR = os.getenv("TRADING_PAIR", "BTCUSDT")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Global variables for Binance client and cache
binance_client = None
last_binance_data = {
    "price": 0,
    "rsi": 50,
    "ema": 0,
    "last_updated": None
}

# Storage for connected WebSocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except RuntimeError:
                disconnected.append(connection)
        
        # Clean up any disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

# In-memory storage for signal history
signal_history = []

# Data models
class SignalData:
    def __init__(
        self,
        signal: str = "WAIT",
        binance_price: float = 0,
        dex_price: float = 0,
        rsi: float = 0,
        ema: float = 0,
        spread: float = 0,
        timestamp: str = None
    ):
        self.signal = signal
        self.binance_price = binance_price
        self.dex_price = dex_price
        self.rsi = rsi
        self.ema = ema
        self.spread = spread
        self.timestamp = timestamp or datetime.now().isoformat()
    
    def to_dict(self):
        return {
            "signal": self.signal,
            "binancePrice": self.binance_price,
            "dexPrice": self.dex_price,
            "rsi": self.rsi,
            "ema": self.ema,
            "spread": self.spread,
            "timestamp": self.timestamp
        }

# Technical indicator calculation functions
def calculate_rsi(prices, periods=14):
    """Calculate the RSI for a price series"""
    if len(prices) < periods + 1:
        return 50  # Default to neutral if not enough data
    
    # Calculate price changes
    deltas = np.diff(prices)
    seed = deltas[:periods+1]
    up = seed[seed >= 0].sum()/periods
    down = -seed[seed < 0].sum()/periods
    
    if down == 0:  # Handle division by zero
        return 100
    
    rs = up/down
    return 100 - (100 / (1 + rs))

def calculate_ema(prices, span=20):
    """Calculate the EMA for a price series"""
    if len(prices) < span:
        return prices[-1] if len(prices) > 0 else 0
    
    return pd.Series(prices).ewm(span=span, adjust=False).mean().iloc[-1]

# Initialize Binance client
async def initialize_binance_client():
    global binance_client
    try:
        if BINANCE_API_KEY and BINANCE_API_SECRET:
            binance_client = await AsyncClient.create(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_API_SECRET
            )
            logger.info("Binance client initialized with API key")
        else:
            binance_client = await AsyncClient.create()
            logger.info("Binance client initialized without API key (limited rate)")
    except Exception as e:
        logger.error(f"Failed to initialize Binance client: {str(e)}")
        binance_client = None

# Function to get Binance data
async def fetch_binance_data():
    """Fetch real-time price data and calculate indicators from Binance"""
    global last_binance_data
    
    try:
        if not binance_client:
            await initialize_binance_client()
            if not binance_client:
                logger.warning("Could not initialize Binance client, using fallback data")
                return await fetch_fallback_data()
        
        try:
        # Get klines (candlestick data) for technical indicators
        klines = await binance_client.get_klines(
            symbol=TRADING_PAIR, 
            interval=AsyncClient.KLINE_INTERVAL_5MINUTE,
            limit=50  # Get enough data to calculate RSI and EMA
        )
        
        # Extract close prices
        close_prices = [float(k[4]) for k in klines]
        
        # Calculate indicators
        current_price = close_prices[-1]
        rsi_value = calculate_rsi(close_prices, RSI_PERIODS)
        ema_value = calculate_ema(close_prices, EMA_PERIODS)
        
        # Update cache
        last_binance_data = {
            "price": current_price,
            "rsi": rsi_value,
            "ema": ema_value,
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(f"Binance data updated: Price=${current_price}, RSI={rsi_value:.2f}, EMA=${ema_value:.2f}")
        return last_binance_data
        except Exception as binance_error:
            logger.error(f"Error fetching from Binance API: {str(binance_error)}")
            return await fetch_fallback_data()
    
    except Exception as e:
        logger.error(f"Error fetching Binance data: {str(e)}")
        return await fetch_fallback_data()

# Function to fetch DEX (PancakeSwap) price for WBTC
async def fetch_dex_price():
    """Fetch WBTC price from PancakeSwap or other DEX"""
    # This would normally use Web3 to interact with the DEX
    # For our simulation, we'll use a price with a small random spread from Binance
    try:
        binance_price = last_binance_data["price"]
        if binance_price <= 0:
            return binance_price
        
        # Simulate DEX price with a realistic spread
        spread_percent = (1.2 * np.random.random() - 0.6)  # -0.6% to +0.6% spread
        dex_price = binance_price * (1 + (spread_percent / 100))
        
        return dex_price, spread_percent
    except Exception as e:
        logger.error(f"Error simulating DEX price: {str(e)}")
        return last_binance_data["price"], 0

# Function to get signal data from bot or generate simulated data
async def get_signal_data():
    try:
        # Check if we have real bot data
        bot_data_path = os.path.join(os.path.dirname(__file__), 'bot_data.json')
        
        if os.path.exists(bot_data_path):
            with open(bot_data_path, 'r') as f:
                data = json.load(f)
                
            return SignalData(
                signal=data.get('signal') or ('HOLD' if data.get('in_position', False) else 'BUY' if data.get('buy_signal', False) else 'WAIT'),
                binance_price=data.get('binance_price', 0),
                dex_price=data.get('dex_price', 0),
                rsi=data.get('rsi', 0),
                ema=data.get('ema', 0),
                spread=data.get('price_difference', 0),
                timestamp=data.get('timestamp', datetime.now().isoformat())
            )
        
        # Otherwise get real BTC price from Binance
        await fetch_binance_data()
        binance_price = last_binance_data["price"]
        rsi_value = last_binance_data["rsi"]
        ema_value = last_binance_data["ema"]
        
        # Get DEX price (simulated or real)
        dex_price, spread_percent = await fetch_dex_price()
        
        # Generate realistic signal based on conditions
        # Buy Signal: RSI < 30, price crosses above EMA, DEX price is cheaper than Binance by at least threshold
        # Sell Signal: RSI > 70, price drops below EMA, DEX price is more expensive than Binance by at least threshold
        
        signal = 'WAIT'
        
        # Check buy condition
        if rsi_value < RSI_OVERSOLD and binance_price > ema_value and spread_percent < -PRICE_DIFFERENCE_THRESHOLD:
            signal = 'BUY'
        
        # Check sell condition
        elif rsi_value > RSI_OVERBOUGHT and binance_price < ema_value and spread_percent > PRICE_DIFFERENCE_THRESHOLD:
            signal = 'SELL'
        
        # Check hold condition (we already own and conditions don't suggest selling)
        elif last_signal_was_buy() and rsi_value < RSI_OVERBOUGHT:
            signal = 'HOLD'
        
        return SignalData(
            signal=signal,
            binance_price=binance_price,
            dex_price=dex_price,
            rsi=rsi_value,
            ema=ema_value,
            spread=spread_percent,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting signal data: {str(e)}")
        return SignalData()  # Return default values

def last_signal_was_buy():
    """Check if the last trading signal was a buy"""
    if signal_history and len(signal_history) > 0:
        last_signal = signal_history[0].get("signal", "WAIT")
        if last_signal == "BUY" or last_signal == "HOLD":
            return True
    return False

# Function to send Telegram notification
async def send_telegram_notification(signal_data: SignalData):
    # Create a new session for each notification
    async with aiohttp.ClientSession() as session:
    try:
        # Your Telegram API key and chat ID from environment variables
        telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not telegram_bot_token or not telegram_chat_id:
            logger.warning("Telegram credentials not configured")
            return {
                "success": False,
                "message": "Telegram credentials not configured"
            }
        
        # Helper functions for message formatting
        def get_signal_emoji(signal_type):
            signal_type = signal_type.upper() if signal_type else ""
            if signal_type == 'BUY':
                return '🟢 💰 BUY SIGNAL'
            elif signal_type == 'SELL':
                return '🔴 💸 SELL SIGNAL'
            elif signal_type == 'HOLD':
                return '🟡 🔒 HOLD SIGNAL'
            else:
                return '⚪ ⏳ WAITING'
        
        def get_price_action(s):
            if s > 0:
                return f'📈 +{s:.2f}%'
            elif s < 0:
                return f'📉 {s:.2f}%'
            else:
                return f'➖ {s:.2f}%'
        
        def get_rsi_status(rsi_value):
            if rsi_value <= 30:
                return '🟢 Oversold'
            elif rsi_value >= 70:
                return '🔴 Overbought'
            else:
                return '⚪ Neutral'
        
        # Format message with rich formatting and emojis
        action_text = ""
        if signal_data.signal == 'BUY':
            action_text = "✅ *ACTION: BUY WBTC on DEX*\nEntry opportunity detected!"
        elif signal_data.signal == 'SELL':
            action_text = "🛑 *ACTION: SELL WBTC on DEX*\nExit opportunity detected!"
        else:
            action_text = "📢 *ACTION: MONITOR MARKET*\nWaiting for better conditions..."
            
        price_comparison = "💹 DEX price higher than Binance" if signal_data.spread > 0 else "📉 DEX price lower than Binance"
            
        message_text = f"""
*📊 WBTC SCALP BOT ALERT 📊*
{get_signal_emoji(signal_data.signal)}

⚡ *Market Conditions*:
• RSI: `{signal_data.rsi:.1f}` {get_rsi_status(signal_data.rsi)}
• EMA: `{signal_data.ema:.2f}`
• Binance BTC: `${signal_data.binance_price:.2f}`
• DEX WBTC: `${signal_data.dex_price:.2f}`
• Spread: {get_price_action(signal_data.spread)}
{price_comparison}

⏰ Signal Time: `{datetime.fromisoformat(signal_data.timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}`

{action_text}
""".strip()
        
        # Send Telegram message
        telegram_url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
        
            async with session.post(telegram_url, json={
                "chat_id": telegram_chat_id,
                "text": message_text,
                "parse_mode": "Markdown"
            }) as response:
                telegram_response = await response.json()
                
                return {
                    "success": True,
                    "message": "Telegram notification sent",
                    "response": telegram_response
                }
    except Exception as e:
        logger.error(f"Error sending Telegram notification: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to send Telegram notification: {str(e)}"
        }

# Generate initial history data
async def generate_initial_history():
    global signal_history
    signal_history = []
    
    # Initialize Binance client first
    await initialize_binance_client()
    
    # Generate history with actual Binance data
    await fetch_binance_data()
    
    for i in range(5):
        data = await get_signal_data()
        
        # Adjust timestamp to create a history
        # Use timedelta instead of replace to properly handle minute arithmetic
        timestamp = datetime.now() - timedelta(minutes=i * 30)
        data.timestamp = timestamp.isoformat()
        
        # Generate different signals for history
        if i % 3 == 0:
            data.signal = 'BUY'
        elif i % 3 == 1:
            data.signal = 'SELL'
        else:
            data.signal = 'HOLD'
        
        # Use list to maintain order
        signal_history.insert(0, data.to_dict())

# Background task for signal updates
async def periodic_signal_update():
    global signal_history
    
    # Initialize signal_history if it doesn't exist
    if signal_history is None:
        signal_history = []
    
    # Track last Telegram notification time
    last_telegram_notification = datetime.now()
    telegram_notification_interval = 30 * 60  # 30 minutes in seconds
        
    while True:
        try:
            # Update Binance data first
            await fetch_binance_data()
            
            # Get signal data based on updated Binance data
            signal_data = await get_signal_data()
            signal_dict = signal_data.to_dict()
            
            # Add to history if signal is BUY or SELL
            if signal_data.signal in ['BUY', 'SELL']:
                # Check if it's different from the last signal
                last_signal = signal_history[0] if signal_history else None
                
                if not last_signal or last_signal["signal"] != signal_data.signal:
                    signal_history.insert(0, signal_dict)
                    
                    # Send Telegram notification for new signal
                    await send_telegram_notification(signal_data)
                    last_telegram_notification = datetime.now()
                    
                    # Trim history to maximum length
                    if len(signal_history) > MAX_HISTORY:
                        signal_history = signal_history[:MAX_HISTORY]
            
            # Send periodic updates to Telegram every 30 minutes
            time_since_last_notification = (datetime.now() - last_telegram_notification).total_seconds()
            if time_since_last_notification > telegram_notification_interval:
                await send_telegram_notification(signal_data)
                last_telegram_notification = datetime.now()
                logger.info("Sent periodic Telegram update")
            
            # Broadcast to all connected clients
            await manager.broadcast(signal_dict)
            
            # Sleep for 5 seconds - more frequent updates for scalping
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in periodic update: {str(e)}")
            await asyncio.sleep(5)  # Sleep and retry

# ROUTES
@app.get("/api/signal")
async def get_signal():
    signal_data = await get_signal_data()
    return signal_data.to_dict()

@app.get("/api/signal/history")
async def get_signal_history():
    return signal_history

@app.post("/api/send-telegram")
async def send_telegram(request: Request):
    try:
        data = await request.json()
        signal_data = SignalData(
            signal=data.get("signal", "WAIT"),
            binance_price=data.get("binancePrice", 0),
            dex_price=data.get("dexPrice", 0),
            rsi=data.get("rsi", 0),
            ema=data.get("ema", 0),
            spread=data.get("spread", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )
        
        result = await send_telegram_notification(signal_data)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing Telegram request: {str(e)}")
        return JSONResponse(
            status_code=200,  # Use 200 to handle in frontend
            content={
                "success": False,
                "message": f"Failed to send Telegram notification: {str(e)}"
            }
        )

@app.post("/api/update-bot-data")
async def update_bot_data(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        data["timestamp"] = datetime.now().isoformat()
        
        # Save to file
        bot_data_path = os.path.join(os.path.dirname(__file__), 'bot_data.json')
        with open(bot_data_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Create signal data and broadcast
        signal_data = SignalData(
            signal=data.get('signal') or ('HOLD' if data.get('in_position', False) else 'BUY' if data.get('buy_signal', False) else 'WAIT'),
            binance_price=data.get('binance_price', 0),
            dex_price=data.get('dex_price', 0),
            rsi=data.get('rsi', 0),
            ema=data.get('ema', 0),
            spread=data.get('price_difference', 0),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )
        
        signal_dict = signal_data.to_dict()
        
        # Add to history if signal is BUY or SELL and broadcast to websocket clients
        if signal_data.signal in ['BUY', 'SELL']:
            # Check if it's different from the last signal
            last_signal = signal_history[0] if signal_history else None
            
            if not last_signal or last_signal["signal"] != signal_data.signal:
                signal_history.insert(0, signal_dict)
                
                # Trim history to maximum length
                if len(signal_history) > MAX_HISTORY:
                    signal_history = signal_history[:MAX_HISTORY]
                
                # Send Telegram notification in background
                background_tasks.add_task(send_telegram_notification, signal_data)
        
        # Broadcast to all connected clients
        background_tasks.add_task(manager.broadcast, signal_dict)
        
        return JSONResponse(content={"success": True})
    except Exception as e:
        logger.error(f"Error updating bot data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "version": "1.0.0",
        "uptime": time.time(),
        "connected_clients": len(manager.active_connections),
        "data_source": "Binance API",
        "last_update": last_binance_data["last_updated"]
    }

# Test endpoint for Telegram notifications
@app.get("/api/test-telegram")
async def test_telegram():
    try:
        # Create test signal data
        signal_data = SignalData(
            signal="BUY",
            binance_price=last_binance_data.get("price", 40000),
            dex_price=last_binance_data.get("price", 40000) * 0.99,  # Simulate 1% lower price on DEX
            rsi=28.5,  # Oversold
            ema=last_binance_data.get("ema", 40000 * 0.98),
            spread=-1.0,  # 1% discount on DEX
            timestamp=datetime.now().isoformat()
        )
        
        # Send test notification
        result = await send_telegram_notification(signal_data)
        
        return JSONResponse(content={
            "success": True,
            "message": "Test Telegram notification sent",
            "details": result
        })
    except Exception as e:
        logger.error(f"Error sending test Telegram notification: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Failed to send test Telegram notification: {str(e)}"
            }
        )

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Send the current signal data immediately upon connection
    signal_data = await get_signal_data()
    await websocket.send_json(signal_data.to_dict())
    
    try:
        # Keep the connection alive, waiting for disconnection
        while True:
            # We are just keeping the connection open
            # Data is sent via the broadcast method
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    global binance_client
    
    # Close Binance client
    if binance_client:
        try:
        await binance_client.close_connection()
        logger.info("Binance client connection closed")
        except Exception as e:
            logger.error(f"Error closing Binance client: {str(e)}")
    
    logger.info("Application shutdown complete")

# Startup event
@app.on_event("startup")
async def startup_event():
    # Initialize Binance client
    await initialize_binance_client()
    
    # Generate initial history data
    await generate_initial_history()
    
    # Start background task
    asyncio.create_task(periodic_signal_update())
    
    logger.info("Application startup complete")

# Initialize advanced trader
advanced_trader_instance = None

# Additional API endpoints for advanced features
@app.get("/api/performance")
async def get_performance():
    """Get performance tracking metrics"""
    try:
        # Create a temporary instance just to get the report
        from performance_tracker import performance_tracker
        
        # Get performance report
        performance_report = performance_tracker.get_performance_report()
        
        return JSONResponse(
            status_code=200,
            content=performance_report
        )
    except Exception as e:
        logger.error(f"Error getting performance data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get performance data: {str(e)}"}
        )

@app.get("/api/multi-timeframe")
async def get_multi_timeframe_analysis():
    """Get multi-timeframe analysis data"""
    try:
        if not binance_client:
            await initialize_binance_client()
            
        # Import the multi_timeframe module
        import multi_timeframe
        
        # Perform multi-timeframe analysis
        result = await multi_timeframe.perform_multi_timeframe_analysis(
            binance_client,
            TRADING_PAIR
        )
        
        # Convert TimeframeData objects to dictionaries
        tf_data = {}
        for timeframe, data in result.items():
            tf_data[timeframe] = {
                "rsi": data.rsi,
                "ema": data.ema,
                "price_above_ema": data.price_above_ema,
                "is_bullish": data.is_bullish,
                "is_bearish": data.is_bearish
            }
            
        return JSONResponse(
            status_code=200,
            content={"timeframe_data": tf_data}
        )
    except Exception as e:
        logger.error(f"Error getting multi-timeframe analysis: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get multi-timeframe analysis: {str(e)}"}
        )

@app.get("/api/position-size")
async def get_position_size():
    """Get position sizing recommendation"""
    try:
        # Import position sizing module
        import position_sizing
        
        # Get latest data
        await fetch_binance_data()
        
        # Calculate volatility (using recent price history)
        price_history = []
        
        # Get klines for volatility calculation
        if binance_client:
            klines = await binance_client.get_klines(
                symbol=TRADING_PAIR,
                interval=AsyncClient.KLINE_INTERVAL_1HOUR,
                limit=30  # Get enough data for volatility
            )
            price_history = [float(k[4]) for k in klines]
        
        volatility = position_sizing.calculate_volatility(price_history) if price_history else 2.0
        
        # Simulate confidence score based on RSI extremes and spread
        ema_diff_percent = ((last_binance_data["price"] / last_binance_data["ema"]) - 1) * 100 if last_binance_data["ema"] > 0 else 0
        
        # Get DEX price difference
        dex_price, spread_percent = await fetch_dex_price()
        
        # Calculate confidence score
        confidence_score = position_sizing.calculate_confidence_score(
            rsi=last_binance_data["rsi"],
            ema_diff_percent=ema_diff_percent,
            price_spread=spread_percent
        )
        
        # Get position size (assuming $1000 account balance for demo)
        account_balance = 1000.0
        position_size, details = position_sizing.get_position_size(
            account_balance=account_balance,
            volatility=volatility,
            confidence_score=confidence_score
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "account_balance": account_balance,
                "position_size": position_size,
                "position_size_percent": details["final_size_percent"],
                "volatility": volatility,
                "confidence_score": confidence_score,
                "details": details
            }
        )
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to calculate position size: {str(e)}"}
        )

@app.get("/api/advanced-signal")
async def get_advanced_signal():
    """Get advanced trading signal incorporating all metrics"""
    try:
        # Initialize advanced trader
        trader = advanced_trader.AdvancedTrader(wallet_balance=1000.0)
        
        # Run trading cycle to get full analysis
        result = await trader.run_trading_cycle()
        
        # Close trader
        await trader.close()
        
        return JSONResponse(
            status_code=200,
            content=result
        )
    except Exception as e:
        logger.error(f"Error getting advanced signal: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get advanced signal: {str(e)}"}
        )

# Fallback data source using CoinGecko or simulated data
async def fetch_fallback_data():
    """Fetch data from alternative source when Binance is not available"""
    
    # Use a new session for each request and ensure it's closed properly
    async with aiohttp.ClientSession() as session:
        try:
            # Try to get data from CoinGecko API
            logger.info("Using CoinGecko API as fallback data source")
            
            # Get current BTC price
            async with session.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd") as response:
                if response.status == 200:
                    data = await response.json()
                    current_price = data["bitcoin"]["usd"]
                    
                    # Get historical data for RSI and EMA calculation
                    days_ago = int(time.time()) - (86400 * 2)  # 2 days of data
                    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={days_ago}&to={int(time.time())}"
                    
                    async with session.get(url) as hist_response:
                        if hist_response.status == 200:
                            hist_data = await hist_response.json()
                            prices = [price[1] for price in hist_data["prices"]]
                            
                            # Sample to simulate 5-minute data
                            sampled_prices = prices[-50:]
                            
                            # Calculate indicators
                            rsi_value = calculate_rsi(sampled_prices, RSI_PERIODS)
                            ema_value = calculate_ema(sampled_prices, EMA_PERIODS)
                            
                            # Update cache
                            updated_data = {
                                "price": current_price,
                                "rsi": rsi_value,
                                "ema": ema_value,
                                "last_updated": datetime.now().isoformat(),
                                "data_source": "CoinGecko"
                            }
                            
                            last_binance_data.update(updated_data)
                            logger.info(f"CoinGecko data fetched: Price=${current_price}, RSI={rsi_value:.2f}, EMA=${ema_value:.2f}")
                            return last_binance_data
            
            # If CoinGecko fails, use simulation
            logger.warning("CoinGecko API failed, using simulation")
            return generate_simulated_data()
        
        except Exception as e:
            logger.error(f"Error fetching fallback data: {str(e)}")
            return generate_simulated_data()

# Run the server
if __name__ == "__main__":
    # Use environment variable for port or default to 5000
    port = int(os.getenv("PORT", 5000))
    
    # Configure server with proper lifecycle management
    config = uvicorn.Config(
        "api_server:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        reload=True,
        loop="asyncio"
    )
    
    # Run with proper signal handling
    server = uvicorn.Server(config)
    server.run() 