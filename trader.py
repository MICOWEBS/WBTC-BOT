import time
import logging
from datetime import datetime
from decimal import Decimal

import config
from indicators import TechnicalIndicators
from dex import DexClient

class WBTCScalpTrader:
    def __init__(self):
        # Set up logging
        self.logger = self.setup_logger()
        
        # Initialize Binance client for technical indicators
        self.indicators = TechnicalIndicators(
            config.BINANCE_API_KEY,
            config.BINANCE_API_SECRET
        )
        
        # Initialize DEX client for PancakeSwap
        self.dex = DexClient(
            config.BSC_NODE_URL,
            config.WALLET_ADDRESS,
            config.WALLET_PRIVATE_KEY
        )
        
        # Trading state
        self.in_position = False
        self.entry_price = None
        self.position_size = None
        self.high_since_entry = None
        self.last_signal = None
        self.last_signal_time = None
        self.signal_cooldown = 300  # 5 minutes cooldown between signals
        
        self.logger.info("WBTC Scalp Trader initialized")
        self.logger.info(f"Simulation mode: {config.SIMULATION_MODE}")
        self.logger.info(f"Auto trading: {config.AUTO_TRADING}")
        
        if not config.AUTO_TRADING:
            self.logger.info("Auto trading is OFF - Bot will generate signals but NOT execute trades")
    
    def setup_logger(self):
        """Set up logging configuration."""
        logger = logging.getLogger("wbtc_scalp_trader")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler(f"wbtc_scalp_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def calculate_price_difference_percent(self, binance_price, dex_price):
        """Calculate percentage difference between Binance and DEX prices."""
        if binance_price is None or dex_price is None:
            return None
        
        difference = ((dex_price / binance_price) - 1) * 100
        return difference
    
    def should_buy(self, binance_indicators, dex_wbtc_price):
        """Check if the buy conditions are met."""
        # Check if we already have a position
        if self.in_position:
            return False
            
        # Check signal cooldown
        if self.last_signal_time and (datetime.now() - self.last_signal_time).total_seconds() < self.signal_cooldown:
            return False
        
        # Extract Binance indicators
        binance_price = binance_indicators['price']
        binance_rsi = binance_indicators['rsi']
        binance_ema = binance_indicators['ema']
        price_crossed_above_ema = binance_indicators['price_crossed_above_ema']
        
        # Calculate price difference between Binance and DEX
        price_diff_percent = self.calculate_price_difference_percent(binance_price, dex_wbtc_price)
        
        # Log current indicators
        self.logger.info(f"Binance BTC Price: ${binance_price:.2f}")
        self.logger.info(f"DEX WBTC Price: ${dex_wbtc_price:.2f}")
        self.logger.info(f"Price difference: {price_diff_percent:.2f}%")
        self.logger.info(f"Binance RSI: {binance_rsi:.2f}")
        self.logger.info(f"Binance EMA: {binance_ema:.2f}")
        
        # Check entry conditions
        rsi_oversold = binance_rsi < config.RSI_OVERSOLD
        dex_price_cheaper = price_diff_percent < -0.5  # WBTC on DEX is 0.5% cheaper
        expected_profit = abs(price_diff_percent) - 0.5  # Account for fees
        profit_within_range = 0.6 <= expected_profit <= 1.5
        
        # Enhanced conditions: check for RSI upward trend
        rsi_rising = False
        if hasattr(self, 'previous_rsi') and self.previous_rsi is not None:
            rsi_rising = binance_rsi > self.previous_rsi
            
        # Store current RSI for next comparison
        self.previous_rsi = binance_rsi
        
        # Check if all conditions are met, with improved signal quality
        if (rsi_oversold and 
            (price_crossed_above_ema or (binance_price > binance_ema and rsi_rising)) and 
            dex_price_cheaper and 
            profit_within_range):
            
            # Only check liquidity and gas price if we will actually execute trades
            if config.AUTO_TRADING:
                # Check liquidity
                liquidity = self.dex.check_liquidity(config.WBTC_ADDRESS)
                if not liquidity['sufficient']:
                    self.logger.warning("Insufficient liquidity for WBTC on DEX, skipping trade")
                    return False
                
                # Check gas price
                gas_price = self.dex.estimate_gas_price()
                if gas_price is not None and gas_price > 10:  # Arbitrary threshold for high gas
                    self.logger.warning(f"Gas price too high ({gas_price} Gwei), skipping trade")
                    return False
            
            self.logger.info("Buy signal detected!")
            self.logger.info(f"Expected profit: {expected_profit:.2f}%")
            
            # Update signal state
            self.last_signal = "BUY"
            self.last_signal_time = datetime.now()
            
            return True
        
        return False
    
    def should_sell(self, binance_indicators, dex_wbtc_price):
        """Check if the sell conditions are met."""
        # Check if we have a position
        if not self.in_position:
            return False
            
        # Check signal cooldown
        if self.last_signal_time and (datetime.now() - self.last_signal_time).total_seconds() < self.signal_cooldown:
            return False
        
        # Extract Binance indicators
        binance_price = binance_indicators['price']
        binance_rsi = binance_indicators['rsi']
        binance_ema = binance_indicators['ema']
        price_dropped_below_ema = binance_indicators['price_dropped_below_ema']
        
        # Calculate price difference between Binance and DEX
        price_diff_percent = self.calculate_price_difference_percent(binance_price, dex_wbtc_price)
        
        # Log current indicators
        self.logger.info(f"Binance BTC Price: ${binance_price:.2f}")
        self.logger.info(f"DEX WBTC Price: ${dex_wbtc_price:.2f}")
        self.logger.info(f"Price difference: {price_diff_percent:.2f}%")
        self.logger.info(f"Binance RSI: {binance_rsi:.2f}")
        self.logger.info(f"Binance EMA: {binance_ema:.2f}")
        
        # Calculate current profit percentage
        current_profit_percent = ((dex_wbtc_price / self.entry_price) - 1) * 100
        self.logger.info(f"Current profit: {current_profit_percent:.2f}%")
        
        # Update highest price since entry for trailing stop
        if self.high_since_entry is None or dex_wbtc_price > self.high_since_entry:
            self.high_since_entry = dex_wbtc_price
            
        # Check if trailing stop is triggered
        trailing_stop_triggered = False
        if self.high_since_entry is not None:
            price_drop_percent = ((dex_wbtc_price / self.high_since_entry) - 1) * 100
            trailing_stop_triggered = price_drop_percent <= -config.TRAILING_STOP_PERCENT
            
        # Enhanced conditions: check for RSI downward trend
        rsi_falling = False
        if hasattr(self, 'previous_rsi') and self.previous_rsi is not None:
            rsi_falling = binance_rsi < self.previous_rsi
        
        # Store current RSI for next comparison
        self.previous_rsi = binance_rsi
            
        # Check exit conditions
        rsi_overbought = binance_rsi > config.RSI_OVERBOUGHT
        dex_price_expensive = price_diff_percent > 0.5  # WBTC on DEX is 0.5% more expensive
        
        # Profit taking - if we've reached target profit
        profit_target_reached = current_profit_percent >= config.PROFIT_TARGET_PERCENT
        
        # Check if any exit condition is met with improved signal quality
        if (rsi_overbought or 
            (price_dropped_below_ema and rsi_falling) or 
            dex_price_expensive or 
            trailing_stop_triggered or
            profit_target_reached):
            
            self.logger.info("Sell signal detected!")
            
            if rsi_overbought:
                self.logger.info("Reason: RSI overbought")
            elif price_dropped_below_ema:
                self.logger.info("Reason: Price dropped below EMA")
            elif dex_price_expensive:
                self.logger.info("Reason: WBTC price on DEX is expensive")
            elif trailing_stop_triggered:
                self.logger.info("Reason: Trailing stop triggered")
            elif profit_target_reached:
                self.logger.info(f"Reason: Profit target reached ({current_profit_percent:.2f}%)")
                
            # Update signal state
            self.last_signal = "SELL"
            self.last_signal_time = datetime.now()
                
            return True
        
        return False
    
    def execute_buy(self, dex_wbtc_price):
        """Execute the buy trade if auto trading is enabled, otherwise just log the signal."""
        try:
            # If auto trading is disabled, just log the signal and track position
            if not config.AUTO_TRADING:
                self.logger.info("⚠️ AUTO TRADING DISABLED - Not executing buy trade, but tracking signal")
                
                # Update tracking data as if we made the trade
                self.in_position = True
                self.entry_price = dex_wbtc_price
                self.position_size = 1.0  # Placeholder value
                self.high_since_entry = dex_wbtc_price
                
                self.logger.info(f"Tracked Entry price: ${self.entry_price:.2f}")
                return True
        
            # Get wallet balances
            balances = self.dex.get_wallet_balances()
            if balances is None:
                self.logger.error("Failed to get wallet balances")
                return False
            
            self.logger.info(f"Wallet balances: {balances}")
            
            # Calculate amount to buy based on percentage of BUSD balance
            busd_balance = balances['busd']
            busd_amount_to_use = busd_balance * (config.TRADE_SIZE_PERCENT / 100)
            
            # Make sure we have enough BUSD
            if busd_amount_to_use <= 0:
                self.logger.error("Not enough BUSD to trade")
                return False
                
            self.logger.info(f"Using {busd_amount_to_use:.4f} BUSD for trade")
            
            # Approve BUSD spending if not in simulation mode
            if not config.SIMULATION_MODE:
                self.logger.info("Approving BUSD spending")
                approval = self.dex.approve_token(config.BUSD_ADDRESS, busd_amount_to_use * 2)
                if approval is None:
                    self.logger.error("Failed to approve BUSD spending")
                    return False
            
            # Buy WBTC with BUSD
            self.logger.info(f"Buying WBTC with {busd_amount_to_use:.4f} BUSD")
            result = self.dex.buy_wbtc_with_busd(busd_amount_to_use)
            
            if result['success']:
                # Update trading state
                self.in_position = True
                self.entry_price = dex_wbtc_price
                self.position_size = busd_amount_to_use / dex_wbtc_price
                self.high_since_entry = dex_wbtc_price
                
                self.logger.info(f"Buy executed successfully")
                self.logger.info(f"Entry price: ${self.entry_price:.2f}")
                self.logger.info(f"Position size: {self.position_size:.8f} WBTC")
                
                if not config.SIMULATION_MODE:
                    self.logger.info(f"Transaction hash: {result['tx_hash']}")
                    
                return True
            else:
                self.logger.error(f"Buy failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing buy: {e}")
            return False
    
    def execute_sell(self, dex_wbtc_price):
        """Execute the sell trade if auto trading is enabled, otherwise just log the signal."""
        try:
            # Calculate profit statistics for tracking
            exit_price = dex_wbtc_price
            profit_percent = ((exit_price / self.entry_price) - 1) * 100
            profit_amount = self.position_size * (exit_price - self.entry_price)
            
            # If auto trading is disabled, just log the signal and reset position
            if not config.AUTO_TRADING:
                self.logger.info("⚠️ AUTO TRADING DISABLED - Not executing sell trade, but tracking signal")
                
                self.logger.info(f"Tracked Exit price: ${exit_price:.2f}")
                self.logger.info(f"Tracked Profit: {profit_amount:.4f} BUSD ({profit_percent:.2f}%)")
                
                # Reset trading state
                self.in_position = False
                self.entry_price = None
                self.position_size = None
                self.high_since_entry = None
                
                return True
            
            # Get wallet balances
            balances = self.dex.get_wallet_balances()
            if balances is None:
                self.logger.error("Failed to get wallet balances")
                return False
            
            self.logger.info(f"Wallet balances: {balances}")
            
            # Get WBTC balance
            wbtc_balance = balances['wbtc']
            
            # Make sure we have enough WBTC
            if wbtc_balance <= 0:
                self.logger.error("Not enough WBTC to sell")
                return False
                
            # If in simulation mode, use the position size we recorded
            if config.SIMULATION_MODE:
                wbtc_to_sell = self.position_size
            else:
                # Otherwise, sell actual balance
                wbtc_to_sell = wbtc_balance
                
            self.logger.info(f"Selling {wbtc_to_sell:.8f} WBTC")
            
            # Approve WBTC spending if not in simulation mode
            if not config.SIMULATION_MODE:
                self.logger.info("Approving WBTC spending")
                approval = self.dex.approve_token(config.WBTC_ADDRESS, wbtc_to_sell * 2)
                if approval is None:
                    self.logger.error("Failed to approve WBTC spending")
                    return False
            
            # Sell WBTC for BUSD
            result = self.dex.sell_wbtc_for_busd(wbtc_to_sell)
            
            if result['success']:
                self.logger.info(f"Sell executed successfully")
                self.logger.info(f"Exit price: ${exit_price:.2f}")
                self.logger.info(f"Profit: {profit_amount:.4f} BUSD ({profit_percent:.2f}%)")
                
                if not config.SIMULATION_MODE:
                    self.logger.info(f"Transaction hash: {result['tx_hash']}")
                    
                # Reset trading state
                self.in_position = False
                self.entry_price = None
                self.position_size = None
                self.high_since_entry = None
                
                return True
            else:
                self.logger.error(f"Sell failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing sell: {e}")
            return False
    
    def get_current_signal(self, binance_indicators, dex_wbtc_price):
        """Get the current signal without executing trades."""
        if self.in_position:
            if self.should_sell(binance_indicators, dex_wbtc_price):
                return "SELL"
            return "HOLD"
        else:
            if self.should_buy(binance_indicators, dex_wbtc_price):
                return "BUY"
            return "WAIT"
    
    def run(self):
        """Run the trading bot."""
        self.logger.info("Starting trading bot")
        
        while True:
            try:
                # Get Binance indicators
                binance_indicators = self.indicators.get_indicators()
                if binance_indicators is None:
                    self.logger.error("Failed to get Binance indicators")
                    time.sleep(10)
                    continue
                
                # Get WBTC price on DEX
                dex_wbtc_price = self.dex.get_wbtc_price_in_busd()
                if dex_wbtc_price is None:
                    self.logger.error("Failed to get DEX WBTC price")
                    time.sleep(10)
                    continue
                
                # Get current signal (without executing)
                current_signal = self.get_current_signal(binance_indicators, dex_wbtc_price)
                
                # Log the current signal
                if current_signal != self.last_signal:
                    self.logger.info(f"Signal: {current_signal}")
                
                # Handle trading based on signals
                if self.in_position:
                    # Check for sell signal
                    if self.should_sell(binance_indicators, dex_wbtc_price):
                        self.execute_sell(dex_wbtc_price)
                else:
                    # Check for buy signal
                    if self.should_buy(binance_indicators, dex_wbtc_price):
                        self.execute_buy(dex_wbtc_price)
                
                # Update dashboard data
                self.update_dashboard_data(binance_indicators, dex_wbtc_price, current_signal)
                
                # Wait until next check
                time.sleep(config.PRICE_CHECK_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                time.sleep(60)  # Wait a bit longer if there's an error
    
    def update_dashboard_data(self, binance_indicators, dex_wbtc_price, current_signal):
        """Update dashboard data for frontend display."""
        try:
            import requests
            import json
            import os
            
            # URL for dashboard API
            dashboard_url = os.getenv("DASHBOARD_API_URL", "http://localhost:5000/api/update-bot-data")
            
            # Prepare data
            data = {
                "binance_price": binance_indicators['price'],
                "dex_price": dex_wbtc_price,
                "rsi": binance_indicators['rsi'],
                "ema": binance_indicators['ema'],
                "price_difference": self.calculate_price_difference_percent(binance_indicators['price'], dex_wbtc_price),
                "in_position": self.in_position,
                "buy_signal": current_signal == "BUY",
                "signal": current_signal,
                "auto_trading": config.AUTO_TRADING,
                "simulation_mode": config.SIMULATION_MODE,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send data to dashboard
            try:
                headers = {'Content-Type': 'application/json'}
                requests.post(dashboard_url, data=json.dumps(data), headers=headers, timeout=2)
            except Exception as e:
                # Silently fail - dashboard updates are not critical
                pass
                
        except Exception as e:
            # Dashboard updates are nice to have but not critical
            pass 