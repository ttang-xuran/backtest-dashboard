#!/usr/bin/env python3
"""
Risk Management Module for Statistical Arbitrage Trading
Provides comprehensive risk controls and position sizing
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

class RiskManager:
    def __init__(self, config: Dict):
        """
        Initialize Risk Manager
        
        Args:
            config: Configuration dictionary with risk parameters
        """
        self.max_position_size = config.get("max_position_size", 0.1)
        self.max_daily_loss = config.get("max_daily_loss", 0.02)  # 2% of account
        self.stop_loss_pct = config.get("stop_loss_pct", 0.05)  # 5% stop loss
        self.max_drawdown = config.get("max_drawdown", 0.10)  # 10% max drawdown
        self.max_open_positions = config.get("max_open_positions", 3)
        self.correlation_threshold = config.get("correlation_threshold", 0.7)
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        self.daily_trades = 0
        self.max_daily_trades = config.get("max_daily_trades", 50)
        
        # Position tracking
        self.open_positions = {}
        self.position_entry_prices = {}
        self.position_entry_times = {}
        
        self.logger = logging.getLogger(__name__)

    def calculate_position_size(self, account_balance: float, volatility: float, 
                              confidence: float = 1.0) -> float:
        """
        Calculate optimal position size using Kelly Criterion and risk limits
        
        Args:
            account_balance: Current account balance
            volatility: Asset pair volatility
            confidence: Confidence level in the signal (0-1)
            
        Returns:
            Optimal position size
        """
        # Base position size from config
        base_size = min(self.max_position_size, account_balance * 0.02)
        
        # Adjust for volatility (reduce size for higher volatility)
        volatility_adjustment = max(0.1, 1.0 - (volatility * 2))
        
        # Adjust for confidence
        confidence_adjustment = max(0.1, confidence)
        
        # Calculate final position size
        position_size = base_size * volatility_adjustment * confidence_adjustment
        
        # Ensure minimum position size
        position_size = max(position_size, 0.001)
        
        self.logger.info(f"Position size calculated: {position_size:.6f} "
                        f"(base: {base_size:.6f}, vol_adj: {volatility_adjustment:.3f}, "
                        f"conf_adj: {confidence_adjustment:.3f})")
        
        return position_size

    def check_risk_limits(self, symbol: str, side: str, size: float, 
                         current_price: float, account_balance: float) -> Tuple[bool, str]:
        """
        Check if a trade violates risk limits
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Position size
            current_price: Current market price
            account_balance: Current account balance
            
        Returns:
            (is_allowed, reason)
        """
        # Check maximum position size
        if size > self.max_position_size:
            return False, f"Position size {size} exceeds maximum {self.max_position_size}"
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss * account_balance:
            return False, f"Daily loss limit exceeded: {self.daily_pnl:.4f}"
        
        # Check maximum drawdown
        if self.current_drawdown > self.max_drawdown:
            return False, f"Maximum drawdown exceeded: {self.current_drawdown:.4f}"
        
        # Check maximum daily trades
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit exceeded: {self.daily_trades}"
        
        # Check maximum open positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Maximum open positions exceeded: {len(self.open_positions)}"
        
        # Check account balance
        required_margin = size * current_price * 0.1  # Assuming 10x leverage
        if required_margin > account_balance * 0.8:  # Don't use more than 80% of balance
            return False, f"Insufficient margin: required {required_margin:.2f}, available {account_balance * 0.8:.2f}"
        
        return True, "Risk checks passed"

    def update_position(self, symbol: str, side: str, size: float, 
                       entry_price: float, timestamp: datetime = None):
        """
        Update position tracking
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Position size
            entry_price: Entry price
            timestamp: Entry timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        position_key = f"{symbol}_{side}"
        
        if position_key in self.open_positions:
            # Add to existing position
            current_size = self.open_positions[position_key]
            current_price = self.position_entry_prices[position_key]
            
            # Calculate weighted average entry price
            total_size = current_size + size
            avg_price = ((current_size * current_price) + (size * entry_price)) / total_size
            
            self.open_positions[position_key] = total_size
            self.position_entry_prices[position_key] = avg_price
        else:
            # New position
            self.open_positions[position_key] = size
            self.position_entry_prices[position_key] = entry_price
            self.position_entry_times[position_key] = timestamp
        
        self.daily_trades += 1
        
        self.logger.info(f"Position updated: {position_key} - Size: {size}, Price: {entry_price}")

    def close_position(self, symbol: str, side: str, exit_price: float, 
                      size: float = None) -> float:
        """
        Close a position and calculate P&L
        
        Args:
            symbol: Trading symbol
            side: Original position side
            exit_price: Exit price
            size: Size to close (None for full position)
            
        Returns:
            Realized P&L
        """
        position_key = f"{symbol}_{side}"
        
        if position_key not in self.open_positions:
            self.logger.warning(f"Attempting to close non-existent position: {position_key}")
            return 0.0
        
        current_size = self.open_positions[position_key]
        entry_price = self.position_entry_prices[position_key]
        
        # Determine size to close
        close_size = size if size is not None else current_size
        close_size = min(close_size, current_size)
        
        # Calculate P&L
        if side == "buy":
            pnl = close_size * (exit_price - entry_price)
        else:  # sell
            pnl = close_size * (entry_price - exit_price)
        
        # Update position
        remaining_size = current_size - close_size
        if remaining_size <= 0.001:  # Close position completely
            del self.open_positions[position_key]
            del self.position_entry_prices[position_key]
            del self.position_entry_times[position_key]
        else:
            self.open_positions[position_key] = remaining_size
        
        # Update P&L tracking
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        self.logger.info(f"Position closed: {position_key} - Size: {close_size}, "
                        f"Entry: {entry_price}, Exit: {exit_price}, P&L: {pnl:.4f}")
        
        return pnl

    def check_stop_loss(self, symbol: str, side: str, current_price: float) -> bool:
        """
        Check if position should be closed due to stop loss
        
        Args:
            symbol: Trading symbol  
            side: Position side
            current_price: Current market price
            
        Returns:
            True if stop loss should be triggered
        """
        position_key = f"{symbol}_{side}"
        
        if position_key not in self.open_positions:
            return False
        
        entry_price = self.position_entry_prices[position_key]
        
        if side == "buy":
            # Long position: stop loss when price drops below entry - stop_loss_pct
            stop_price = entry_price * (1 - self.stop_loss_pct)
            return current_price <= stop_price
        else:  # sell
            # Short position: stop loss when price rises above entry + stop_loss_pct  
            stop_price = entry_price * (1 + self.stop_loss_pct)
            return current_price >= stop_price

    def update_drawdown(self, current_equity: float):
        """
        Update drawdown calculations
        
        Args:
            current_equity: Current account equity
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        self.logger.debug(f"Drawdown updated: {self.current_drawdown:.4f} "
                         f"(Peak: {self.peak_equity:.2f}, Current: {current_equity:.2f})")

    def reset_daily_metrics(self):
        """Reset daily metrics at start of new day"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.logger.info("Daily risk metrics reset")

    def get_risk_summary(self) -> Dict:
        """
        Get current risk summary
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "current_drawdown": self.current_drawdown,
            "daily_trades": self.daily_trades,
            "open_positions": len(self.open_positions),
            "position_details": self.open_positions.copy(),
            "max_daily_loss_remaining": self.max_daily_loss + self.daily_pnl,
            "max_drawdown_remaining": self.max_drawdown - self.current_drawdown
        }

    def calculate_correlation(self, prices1: list, prices2: list) -> float:
        """
        Calculate correlation between two price series
        
        Args:
            prices1: First price series
            prices2: Second price series
            
        Returns:
            Correlation coefficient
        """
        if len(prices1) < 2 or len(prices2) < 2:
            return 0.0
        
        try:
            correlation = np.corrcoef(prices1, prices2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def is_correlation_acceptable(self, prices1: list, prices2: list) -> bool:
        """
        Check if correlation between assets is acceptable for pairs trading
        
        Args:
            prices1: First asset price series
            prices2: Second asset price series
            
        Returns:
            True if correlation is acceptable
        """
        correlation = self.calculate_correlation(prices1, prices2)
        return abs(correlation) >= self.correlation_threshold