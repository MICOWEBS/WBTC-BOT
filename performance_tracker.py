"""
Performance Tracking Module

Tracks detailed P&L metrics, trade statistics, and performance analytics
for the trading bot. Provides metrics for optimizing the strategy.
"""

import json
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import statistics
import numpy as np
from config import PERFORMANCE_LOG_FILE, TRACK_PERFORMANCE

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks trading performance metrics and P&L statistics"""
    
    def __init__(self, data_file: str = PERFORMANCE_LOG_FILE):
        """Initialize performance tracker with data file path"""
        self.data_file = data_file
        self.trades = []
        self.unrealized_trades = []
        self.initial_balance = 0
        self.current_balance = 0
        self.metrics = {}
        self.start_time = time.time()
        self.load_data()
    
    def load_data(self) -> None:
        """Load performance data from file if it exists"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.trades = data.get('trades', [])
                    self.unrealized_trades = data.get('unrealized_trades', [])
                    self.initial_balance = data.get('initial_balance', 0)
                    self.current_balance = data.get('current_balance', 0)
                    self.metrics = data.get('metrics', {})
                    self.start_time = data.get('start_time', time.time())
                logger.info(f"Loaded performance data: {len(self.trades)} completed trades")
            except Exception as e:
                logger.error(f"Error loading performance data: {str(e)}")
    
    def save_data(self) -> None:
        """Save performance data to file"""
        if not TRACK_PERFORMANCE:
            return
            
        try:
            data = {
                'trades': self.trades,
                'unrealized_trades': self.unrealized_trades,
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'metrics': self.metrics,
                'start_time': self.start_time
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Performance data saved")
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
    
    def set_initial_balance(self, balance: float) -> None:
        """Set the initial account balance"""
        if self.initial_balance == 0:
            self.initial_balance = balance
            self.current_balance = balance
            logger.info(f"Initial balance set to ${balance:.2f}")
            self.save_data()
    
    def update_balance(self, new_balance: float) -> None:
        """Update the current account balance"""
        self.current_balance = new_balance
        self.calculate_metrics()
        self.save_data()
    
    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Record a completed trade
        
        Args:
            trade_data: Dictionary containing trade details:
                - trade_id: Unique identifier for the trade
                - entry_time: Time of entry
                - exit_time: Time of exit
                - entry_price: Price at entry
                - exit_price: Price at exit
                - quantity: Amount traded
                - side: 'BUY' or 'SELL'
                - profit_loss: Profit or loss amount
                - profit_loss_percent: Profit or loss percentage
                - fees: Trading fees
                - slippage: Price slippage
                - signal_strength: Signal strength at entry
                - trade_duration: Duration of trade in seconds
                - exit_reason: Reason for exiting the trade
        """
        if not TRACK_PERFORMANCE:
            return
            
        # Add timestamp if not provided
        if 'exit_time' not in trade_data:
            trade_data['exit_time'] = datetime.now().isoformat()
            
        # Calculate profit/loss if not provided
        if 'profit_loss' not in trade_data and 'entry_price' in trade_data and 'exit_price' in trade_data:
            entry_price = trade_data['entry_price']
            exit_price = trade_data['exit_price']
            quantity = trade_data.get('quantity', 1)
            side = trade_data.get('side', 'BUY')
            
            if side == 'BUY':
                trade_data['profit_loss'] = (exit_price - entry_price) * quantity
                if entry_price > 0:
                    trade_data['profit_loss_percent'] = ((exit_price / entry_price) - 1) * 100
            else:  # SELL
                trade_data['profit_loss'] = (entry_price - exit_price) * quantity
                if entry_price > 0:
                    trade_data['profit_loss_percent'] = ((entry_price / exit_price) - 1) * 100
        
        # Add to completed trades
        self.trades.append(trade_data)
        
        # Remove from unrealized trades if exists
        self.unrealized_trades = [t for t in self.unrealized_trades 
                                if t.get('trade_id') != trade_data.get('trade_id')]
        
        # Update metrics
        self.calculate_metrics()
        
        # Log the trade
        profit_loss = trade_data.get('profit_loss', 0)
        profit_loss_percent = trade_data.get('profit_loss_percent', 0)
        logger.info(
            f"Trade completed: {trade_data.get('side')} {trade_data.get('quantity', 0)} @ "
            f"${trade_data.get('exit_price', 0):.2f}, P&L: "
            f"${profit_loss:.2f} ({profit_loss_percent:.2f}%)"
        )
        
        # Save data
        self.save_data()
    
    def record_unrealized_trade(self, trade_data: Dict[str, Any]) -> None:
        """Record an unrealized (paper) trade for simulation mode"""
        if not TRACK_PERFORMANCE:
            return
            
        # Add timestamp if not provided
        if 'entry_time' not in trade_data:
            trade_data['entry_time'] = datetime.now().isoformat()
            
        # Set initial unrealized P&L
        trade_data['unrealized_profit_loss'] = 0
        trade_data['unrealized_profit_loss_percent'] = 0
        
        # Generate trade ID if not provided
        if 'trade_id' not in trade_data:
            trade_data['trade_id'] = f"sim_{int(time.time())}_{len(self.unrealized_trades)}"
        
        # Add to unrealized trades
        self.unrealized_trades.append(trade_data)
        
        # Log the trade
        logger.info(
            f"Unrealized trade opened: {trade_data.get('side')} {trade_data.get('quantity', 0)} @ "
            f"${trade_data.get('entry_price', 0):.2f}"
        )
        
        # Save data
        self.save_data()
    
    def update_unrealized_trade(self, trade_id: str, current_price: float) -> Dict[str, Any]:
        """Update unrealized P&L for a simulated trade"""
        if not TRACK_PERFORMANCE:
            return {}
            
        # Find the trade
        for trade in self.unrealized_trades:
            if trade.get('trade_id') == trade_id:
                entry_price = trade.get('entry_price', 0)
                quantity = trade.get('quantity', 1)
                side = trade.get('side', 'BUY')
                
                # Calculate unrealized P&L
                if side == 'BUY':
                    trade['unrealized_profit_loss'] = (current_price - entry_price) * quantity
                    if entry_price > 0:
                        trade['unrealized_profit_loss_percent'] = ((current_price / entry_price) - 1) * 100
                else:  # SELL
                    trade['unrealized_profit_loss'] = (entry_price - current_price) * quantity
                    if entry_price > 0:
                        trade['unrealized_profit_loss_percent'] = ((entry_price / current_price) - 1) * 100
                
                # Update current price
                trade['current_price'] = current_price
                trade['last_updated'] = datetime.now().isoformat()
                
                # Save data (but not on every update to avoid excessive I/O)
                if int(time.time()) % 60 == 0:  # Save once per minute
                    self.save_data()
                    
                return trade
                
        return {}
    
    def close_unrealized_trade(self, trade_id: str, exit_price: float, exit_reason: str = 'manual') -> Dict[str, Any]:
        """
        Close an unrealized trade and move it to completed trades
        
        Args:
            trade_id: ID of the trade to close
            exit_price: Exit price
            exit_reason: Reason for closing the trade
            
        Returns:
            Completed trade data
        """
        if not TRACK_PERFORMANCE:
            return {}
            
        # Find the trade
        for i, trade in enumerate(self.unrealized_trades):
            if trade.get('trade_id') == trade_id:
                # Update with exit information
                trade['exit_price'] = exit_price
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_reason'] = exit_reason
                
                # Calculate trade duration
                try:
                    entry_time = datetime.fromisoformat(trade.get('entry_time', ''))
                    exit_time = datetime.now()
                    duration = (exit_time - entry_time).total_seconds()
                    trade['trade_duration'] = duration
                except:
                    trade['trade_duration'] = 0
                
                # Calculate final P&L
                entry_price = trade.get('entry_price', 0)
                quantity = trade.get('quantity', 1)
                side = trade.get('side', 'BUY')
                
                if side == 'BUY':
                    trade['profit_loss'] = (exit_price - entry_price) * quantity
                    if entry_price > 0:
                        trade['profit_loss_percent'] = ((exit_price / entry_price) - 1) * 100
                else:  # SELL
                    trade['profit_loss'] = (entry_price - exit_price) * quantity
                    if entry_price > 0:
                        trade['profit_loss_percent'] = ((entry_price / exit_price) - 1) * 100
                
                # Remove from unrealized and add to completed trades
                completed_trade = self.unrealized_trades.pop(i)
                self.trades.append(completed_trade)
                
                # Update metrics
                self.calculate_metrics()
                
                # Log the trade
                profit_loss = completed_trade.get('profit_loss', 0)
                profit_loss_percent = completed_trade.get('profit_loss_percent', 0)
                logger.info(
                    f"Unrealized trade closed: {completed_trade.get('side')} {completed_trade.get('quantity', 0)} @ "
                    f"${exit_price:.2f}, P&L: ${profit_loss:.2f} ({profit_loss_percent:.2f}%), "
                    f"Reason: {exit_reason}"
                )
                
                # Save data
                self.save_data()
                
                return completed_trade
                
        return {}
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from trade history"""
        if not self.trades:
            self.metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit_loss': 0,
                'total_profit_loss_percent': 0,
                'average_profit_loss': 0,
                'average_winning_trade': 0,
                'average_losing_trade': 0,
                'largest_winning_trade': 0,
                'largest_losing_trade': 0,
                'profit_factor': 0,
                'average_trade_duration': 0,
                'sharpe_ratio': 0
            }
            return self.metrics
        
        # Basic counts
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('profit_loss', 0) < 0]
        
        # Win rate
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit/Loss metrics
        total_profit_loss = sum(t.get('profit_loss', 0) for t in self.trades)
        average_profit_loss = total_profit_loss / total_trades if total_trades > 0 else 0
        
        # Average winners and losers
        average_winning_trade = sum(t.get('profit_loss', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        average_losing_trade = sum(t.get('profit_loss', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Largest trades
        largest_winning_trade = max([t.get('profit_loss', 0) for t in winning_trades]) if winning_trades else 0
        largest_losing_trade = min([t.get('profit_loss', 0) for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_gross_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
        total_gross_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
        profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 0
        
        # Trade duration
        durations = [t.get('trade_duration', 0) for t in self.trades if 'trade_duration' in t]
        average_trade_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate Sharpe ratio (if we have enough trades)
        if len(self.trades) >= 10:
            returns = [t.get('profit_loss_percent', 0) / 100 for t in self.trades]
            if returns and statistics.stdev(returns) > 0:
                sharpe_ratio = (statistics.mean(returns) * 252) / (statistics.stdev(returns) * np.sqrt(252))
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Total profit/loss percentage (compounded)
        total_profit_loss_percent = ((self.current_balance / self.initial_balance) - 1) * 100 if self.initial_balance > 0 else 0
        
        # Update metrics
        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,  # as percentage
            'total_profit_loss': total_profit_loss,
            'total_profit_loss_percent': total_profit_loss_percent,
            'average_profit_loss': average_profit_loss,
            'average_winning_trade': average_winning_trade,
            'average_losing_trade': average_losing_trade,
            'largest_winning_trade': largest_winning_trade,
            'largest_losing_trade': largest_losing_trade,
            'profit_factor': profit_factor,
            'average_trade_duration': average_trade_duration,
            'sharpe_ratio': sharpe_ratio,
            
            # Current positions
            'open_positions': len(self.unrealized_trades),
            'unrealized_profit_loss': sum(t.get('unrealized_profit_loss', 0) for t in self.unrealized_trades),
            
            # Timing
            'trading_days': (time.time() - self.start_time) / (60 * 60 * 24),
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance
        }
        
        return self.metrics
    
    def get_trades_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary statistics for trades within a specific time period"""
        # Filter trades by date
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = []
        
        for trade in self.trades:
            try:
                exit_time = datetime.fromisoformat(trade.get('exit_time', ''))
                if exit_time >= cutoff_date:
                    recent_trades.append(trade)
            except:
                continue
        
        # Count trades by day
        trades_by_day = {}
        for trade in recent_trades:
            try:
                exit_time = datetime.fromisoformat(trade.get('exit_time', ''))
                day_key = exit_time.strftime('%Y-%m-%d')
                
                if day_key not in trades_by_day:
                    trades_by_day[day_key] = {
                        'count': 0,
                        'profit': 0,
                        'winners': 0,
                        'losers': 0
                    }
                    
                trades_by_day[day_key]['count'] += 1
                trades_by_day[day_key]['profit'] += trade.get('profit_loss', 0)
                
                if trade.get('profit_loss', 0) > 0:
                    trades_by_day[day_key]['winners'] += 1
                else:
                    trades_by_day[day_key]['losers'] += 1
            except:
                continue
        
        # Get recent performance metrics
        if recent_trades:
            win_rate = sum(1 for t in recent_trades if t.get('profit_loss', 0) > 0) / len(recent_trades)
            total_profit = sum(t.get('profit_loss', 0) for t in recent_trades)
            avg_profit = total_profit / len(recent_trades)
        else:
            win_rate = 0
            total_profit = 0
            avg_profit = 0
        
        return {
            'period_days': days,
            'total_trades': len(recent_trades),
            'win_rate': win_rate * 100,
            'total_profit': total_profit,
            'average_profit': avg_profit,
            'trades_by_day': trades_by_day
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        # Ensure metrics are up to date
        self.calculate_metrics()
        
        # Get time-based summaries
        daily_summary = self.get_trades_summary(days=1)
        weekly_summary = self.get_trades_summary(days=7)
        monthly_summary = self.get_trades_summary(days=30)
        
        # Return full report
        return {
            'metrics': self.metrics,
            'daily': daily_summary,
            'weekly': weekly_summary,
            'monthly': monthly_summary,
            'unrealized_trades': self.unrealized_trades,
            'recent_trades': self.trades[-10:] if len(self.trades) > 10 else self.trades
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics (for testing or restarting)"""
        self.trades = []
        self.unrealized_trades = []
        self.initial_balance = 0
        self.current_balance = 0
        self.metrics = {}
        self.start_time = time.time()
        self.save_data()
        logger.info("Performance metrics reset")

# Create a global instance
performance_tracker = PerformanceTracker() 