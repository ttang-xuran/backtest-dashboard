#!/usr/bin/env python3
"""
Transaction Cost Optimization Service
Minimizes trading costs and prevents unprofitable trades due to fees
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from utils.logger import get_logger
from config.settings import config_manager


@dataclass
class TradingCosts:
    maker_fee: float = 0.0002  # 0.02% for maker orders
    taker_fee: float = 0.0005  # 0.05% for taker orders (market orders)
    funding_rate: float = 0.0001  # 0.01% funding rate per 8 hours
    slippage_estimate: float = 0.0003  # 0.03% average slippage
    spread_cost: float = 0.0002  # 0.02% bid-ask spread cost


@dataclass
class TradeAnalysis:
    estimated_total_cost: float
    min_profit_target: float
    cost_adjusted_position_size: float
    expected_hold_time_hours: float
    cost_per_hour: float
    recommended_action: str
    reasoning: str


class CostOptimizationService:
    def __init__(self, data_service=None):
        self.data_service = data_service
        self.logger = get_logger(__name__)
        self.config = config_manager.config
        
        # Cost structure
        self.costs = TradingCosts()
        
        # Trading statistics for cost estimation
        self.recent_trades = deque(maxlen=50)
        self.slippage_history = deque(maxlen=30)
        self.hold_time_history = deque(maxlen=30)
        
        # Daily cost tracking
        self.daily_costs = deque(maxlen=30)
        self.cost_tracking_date = datetime.now().date()
        self.daily_fee_total = 0.0
        
        # Optimization parameters
        self.min_profit_multiple = 2.0  # Minimum profit should be 2x total costs
        self.max_daily_cost_pct = 0.005  # Max 0.5% of account in daily fees
        self.cost_efficiency_threshold = 0.7  # Trade only if >70% cost efficient
    
    def analyze_trade_costs(self, symbol_pair: str, position_size: float, 
                          current_prices: Dict[str, float], 
                          expected_z_score: float,
                          confidence: float) -> TradeAnalysis:
        """Comprehensive analysis of trade costs vs expected profit"""
        
        # Calculate total transaction costs
        total_cost = self._calculate_total_transaction_cost(position_size, current_prices)
        
        # Estimate expected profit based on Z-score and confidence
        expected_profit = self._estimate_expected_profit(expected_z_score, confidence, position_size, current_prices)
        
        # Calculate minimum profit target
        min_profit_target = total_cost * self.min_profit_multiple
        
        # Estimate hold time
        expected_hold_time = self._estimate_hold_time(abs(expected_z_score), confidence)
        
        # Calculate cost per hour
        cost_per_hour = total_cost / max(expected_hold_time, 0.1)
        
        # Optimize position size for cost efficiency
        optimized_size = self._optimize_position_size(position_size, total_cost, expected_profit)
        
        # Make recommendation
        recommendation, reasoning = self._make_trade_recommendation(
            total_cost, expected_profit, min_profit_target, confidence, expected_hold_time
        )
        
        analysis = TradeAnalysis(
            estimated_total_cost=total_cost,
            min_profit_target=min_profit_target,
            cost_adjusted_position_size=optimized_size,
            expected_hold_time_hours=expected_hold_time,
            cost_per_hour=cost_per_hour,
            recommended_action=recommendation,
            reasoning=reasoning
        )
        
        self.logger.info(f"Trade cost analysis: Cost={total_cost:.6f}, MinProfit={min_profit_target:.6f}, "
                        f"ExpectedProfit={expected_profit:.6f}, Recommendation={recommendation}")
        
        return analysis
    
    def _calculate_total_transaction_cost(self, position_size: float, prices: Dict[str, float]) -> float:
        """Calculate total cost including all fees"""
        
        # Base position value (for BTC/ETH pair)
        btc_value = position_size * prices.get('BTC', 50000)
        eth_value = position_size * prices.get('ETH', 3000)
        total_position_value = btc_value + eth_value
        
        # Trading fees (entry + exit for both assets)
        trading_fees = total_position_value * self.costs.taker_fee * 2  # Round trip
        
        # Spread costs (bid-ask spread impact)
        spread_costs = total_position_value * self.costs.spread_cost * 2  # Entry + exit
        
        # Slippage costs (estimated based on position size)
        slippage_multiplier = min(2.0, 1.0 + (position_size / 0.1))  # Higher slippage for larger positions
        slippage_costs = total_position_value * self.costs.slippage_estimate * slippage_multiplier
        
        # Funding costs (estimated based on expected hold time)
        expected_hold_hours = self._estimate_hold_time(3.0, 0.7)  # Use default estimates
        funding_periods = max(1, expected_hold_hours / 8)  # Funding every 8 hours
        funding_costs = total_position_value * self.costs.funding_rate * funding_periods
        
        total_cost = trading_fees + spread_costs + slippage_costs + funding_costs
        
        self.logger.debug(f"Cost breakdown: Trading={trading_fees:.6f}, Spread={spread_costs:.6f}, "
                         f"Slippage={slippage_costs:.6f}, Funding={funding_costs:.6f}")
        
        return total_cost
    
    def _estimate_expected_profit(self, z_score: float, confidence: float, 
                                position_size: float, prices: Dict[str, float]) -> float:
        """Estimate expected profit based on Z-score and confidence"""
        
        # Base profit expectation from Z-score (mean reversion assumption)
        # Higher Z-score = higher expected reversion = higher profit potential
        base_profit_pct = min(0.02, abs(z_score) * 0.003)  # Max 2% profit expectation
        
        # Adjust for confidence
        confidence_adjusted_profit = base_profit_pct * confidence
        
        # Adjust for historical success rate
        historical_success_rate = self._get_historical_success_rate()
        probability_adjusted_profit = confidence_adjusted_profit * historical_success_rate
        
        # Calculate dollar amount
        total_position_value = position_size * (prices.get('BTC', 50000) + prices.get('ETH', 3000))
        expected_profit = total_position_value * probability_adjusted_profit
        
        return expected_profit
    
    def _estimate_hold_time(self, z_score: float, confidence: float) -> float:
        """Estimate expected holding time in hours"""
        
        # Use historical data if available
        if len(self.hold_time_history) > 5:
            historical_avg = np.mean(list(self.hold_time_history))
            return max(0.5, historical_avg)
        
        # Base hold time estimation
        # Higher Z-score typically means longer reversion time
        base_hold_time = 2.0 + (abs(z_score) - 2.0) * 0.5  # 2-4 hours typically
        
        # Adjust for confidence (lower confidence = exit faster)
        confidence_adjustment = 0.5 + confidence * 0.5  # 0.5x to 1.0x multiplier
        
        estimated_hold_time = base_hold_time * confidence_adjustment
        
        return max(0.5, min(24.0, estimated_hold_time))  # Clamp between 0.5 and 24 hours
    
    def _optimize_position_size(self, original_size: float, total_cost: float, expected_profit: float) -> float:
        """Optimize position size for best cost efficiency"""
        
        if expected_profit <= total_cost:
            return 0.0  # Don't trade if expected profit doesn't cover costs
        
        # Calculate cost efficiency ratio
        efficiency_ratio = expected_profit / total_cost
        
        if efficiency_ratio < self.cost_efficiency_threshold:
            # Reduce position size to improve efficiency
            size_multiplier = max(0.1, efficiency_ratio / self.cost_efficiency_threshold)
            optimized_size = original_size * size_multiplier
        else:
            # Size is already efficient
            optimized_size = original_size
        
        # Ensure we don't exceed daily cost limits
        if self._would_exceed_daily_cost_limit(optimized_size, total_cost):
            max_affordable_size = self._calculate_max_affordable_size()
            optimized_size = min(optimized_size, max_affordable_size)
        
        return max(0.0, optimized_size)
    
    def _make_trade_recommendation(self, total_cost: float, expected_profit: float, 
                                 min_profit_target: float, confidence: float, 
                                 hold_time: float) -> Tuple[str, str]:
        """Make final trade recommendation based on cost analysis"""
        
        # Cost efficiency check
        if expected_profit < total_cost:
            return "SKIP", f"Expected profit ({expected_profit:.6f}) less than costs ({total_cost:.6f})"
        
        # Minimum profit target check
        if expected_profit < min_profit_target:
            return "SKIP", f"Expected profit ({expected_profit:.6f}) below minimum target ({min_profit_target:.6f})"
        
        # Daily cost limit check
        if self._would_exceed_daily_cost_limit_amount(total_cost):
            return "SKIP", f"Would exceed daily cost limit"
        
        # Cost per hour efficiency
        cost_per_hour = total_cost / hold_time
        if cost_per_hour > 0.001:  # More than 0.1% per hour is expensive
            return "REDUCE_SIZE", f"High cost per hour ({cost_per_hour:.6f}), consider smaller position"
        
        # Confidence check
        if confidence < 0.6:
            return "REDUCE_SIZE", f"Low confidence ({confidence:.3f}), trade smaller"
        
        # All checks passed
        profit_ratio = expected_profit / total_cost
        if profit_ratio > 3.0:
            return "TRADE", f"Excellent profit/cost ratio ({profit_ratio:.2f}x)"
        elif profit_ratio > 2.0:
            return "TRADE", f"Good profit/cost ratio ({profit_ratio:.2f}x)"
        else:
            return "TRADE_SMALL", f"Marginal profit/cost ratio ({profit_ratio:.2f}x)"
    
    def _get_historical_success_rate(self) -> float:
        """Get historical success rate for profit estimation"""
        
        if len(self.recent_trades) < 5:
            return 0.6  # Default assumption
        
        successful_trades = sum(1 for trade in self.recent_trades if trade.get('profitable', False))
        return successful_trades / len(self.recent_trades)
    
    def _would_exceed_daily_cost_limit(self, position_size: float, trade_cost: float) -> bool:
        """Check if trade would exceed daily cost limits"""
        return self._would_exceed_daily_cost_limit_amount(trade_cost)
    
    def _would_exceed_daily_cost_limit_amount(self, additional_cost: float) -> bool:
        """Check if additional cost would exceed daily limits"""
        
        # Reset daily tracking if new day
        today = datetime.now().date()
        if today != self.cost_tracking_date:
            self.daily_fee_total = 0.0
            self.cost_tracking_date = today
        
        # Estimate account value (simplified)
        account_value = 100000  # Default - should get from account info
        if self.data_service:
            # Try to get actual account value
            try:
                # This would need to be implemented in data service
                pass
            except:
                pass
        
        daily_cost_limit = account_value * self.max_daily_cost_pct
        projected_daily_cost = self.daily_fee_total + additional_cost
        
        return projected_daily_cost > daily_cost_limit
    
    def _calculate_max_affordable_size(self) -> float:
        """Calculate maximum position size we can afford today"""
        
        # Simplified calculation - in practice would be more sophisticated
        remaining_cost_budget = self._get_remaining_daily_cost_budget()
        
        # Estimate cost per unit of position size
        typical_btc_price = 50000
        typical_eth_price = 3000
        cost_per_unit = (typical_btc_price + typical_eth_price) * (
            self.costs.taker_fee * 2 + 
            self.costs.spread_cost * 2 + 
            self.costs.slippage_estimate
        )
        
        max_size = remaining_cost_budget / cost_per_unit if cost_per_unit > 0 else 0.0
        
        return max(0.0, max_size)
    
    def _get_remaining_daily_cost_budget(self) -> float:
        """Get remaining daily cost budget"""
        
        account_value = 100000  # Simplified
        daily_limit = account_value * self.max_daily_cost_pct
        
        return max(0.0, daily_limit - self.daily_fee_total)
    
    def record_trade_execution(self, position_size: float, actual_costs: Dict[str, float], 
                             execution_time: datetime, slippage: float):
        """Record actual trade execution for cost model improvement"""
        
        total_actual_cost = sum(actual_costs.values())
        
        trade_record = {
            'timestamp': execution_time,
            'position_size': position_size,
            'total_cost': total_actual_cost,
            'slippage': slippage,
            'cost_breakdown': actual_costs.copy()
        }
        
        self.recent_trades.append(trade_record)
        self.slippage_history.append(slippage)
        
        # Update daily cost tracking
        today = datetime.now().date()
        if today == self.cost_tracking_date:
            self.daily_fee_total += total_actual_cost
        else:
            # New day
            self.daily_costs.append(self.daily_fee_total)
            self.daily_fee_total = total_actual_cost
            self.cost_tracking_date = today
        
        # Update cost model with actual data
        self._update_cost_estimates()
        
        self.logger.info(f"Recorded trade execution: Size={position_size:.4f}, "
                        f"Cost={total_actual_cost:.6f}, Slippage={slippage:.6f}")
    
    def record_trade_exit(self, hold_time_hours: float, final_pnl: float, exit_costs: Dict[str, float]):
        """Record trade exit for hold time and profitability tracking"""
        
        self.hold_time_history.append(hold_time_hours)
        
        trade_update = {
            'hold_time': hold_time_hours,
            'final_pnl': final_pnl,
            'profitable': final_pnl > 0,
            'exit_costs': exit_costs.copy()
        }
        
        # Update the most recent trade record
        if self.recent_trades:
            self.recent_trades[-1].update(trade_update)
        
        self.logger.info(f"Recorded trade exit: HoldTime={hold_time_hours:.2f}h, "
                        f"PnL={final_pnl:.6f}, Profitable={final_pnl > 0}")
    
    def _update_cost_estimates(self):
        """Update cost estimates based on actual trading data"""
        
        if len(self.slippage_history) > 5:
            # Update slippage estimate with recent average
            recent_avg_slippage = np.mean(list(self.slippage_history)[-10:])
            self.costs.slippage_estimate = (self.costs.slippage_estimate * 0.7 + 
                                          recent_avg_slippage * 0.3)
        
        if len(self.recent_trades) > 3:
            # Could update other cost estimates here based on actual data
            pass
    
    def get_cost_efficiency_report(self) -> Dict:
        """Generate cost efficiency analysis report"""
        
        if len(self.recent_trades) == 0:
            return {'status': 'insufficient_data'}
        
        recent_trades = list(self.recent_trades)[-20:]  # Last 20 trades
        
        total_costs = sum(trade.get('total_cost', 0) for trade in recent_trades)
        total_pnl = sum(trade.get('final_pnl', 0) for trade in recent_trades 
                       if 'final_pnl' in trade)
        
        profitable_trades = [t for t in recent_trades if t.get('profitable', False)]
        
        cost_efficiency_metrics = {
            'total_trades': len(recent_trades),
            'total_costs': total_costs,
            'total_pnl': total_pnl,
            'net_after_costs': total_pnl - total_costs,
            'cost_ratio': total_costs / max(abs(total_pnl), 0.0001),
            'win_rate': len(profitable_trades) / len(recent_trades),
            'avg_cost_per_trade': total_costs / len(recent_trades),
            'avg_slippage': np.mean(list(self.slippage_history)) if self.slippage_history else 0,
            'avg_hold_time': np.mean(list(self.hold_time_history)) if self.hold_time_history else 0,
            'daily_cost_utilization': self.daily_fee_total / (100000 * self.max_daily_cost_pct),
            'cost_efficiency_score': self._calculate_cost_efficiency_score()
        }
        
        return cost_efficiency_metrics
    
    def _calculate_cost_efficiency_score(self) -> float:
        """Calculate overall cost efficiency score (0-1)"""
        
        if len(self.recent_trades) < 3:
            return 0.5
        
        recent_trades = list(self.recent_trades)[-10:]
        
        # Factor 1: Profit vs Cost ratio
        total_costs = sum(trade.get('total_cost', 0) for trade in recent_trades)
        total_pnl = sum(trade.get('final_pnl', 0) for trade in recent_trades 
                       if 'final_pnl' in trade)
        
        profit_cost_ratio = max(0, total_pnl / max(total_costs, 0.0001))
        profit_score = min(1.0, profit_cost_ratio / 3.0)  # Normalize to 0-1
        
        # Factor 2: Win rate
        win_rate = len([t for t in recent_trades if t.get('profitable', False)]) / len(recent_trades)
        
        # Factor 3: Cost control (lower costs = higher score)
        avg_cost = total_costs / len(recent_trades)
        cost_control_score = max(0, 1.0 - (avg_cost / 0.001))  # Penalize if avg cost > 0.1%
        
        # Combine factors
        efficiency_score = (profit_score * 0.5 + win_rate * 0.3 + cost_control_score * 0.2)
        
        return min(1.0, max(0.0, efficiency_score))
    
    def should_pause_trading_due_to_costs(self) -> Tuple[bool, str]:
        """Determine if trading should be paused due to cost concerns"""
        
        # Check daily cost limit
        if self._would_exceed_daily_cost_limit_amount(0):
            return True, "Daily cost limit reached"
        
        # Check cost efficiency
        if len(self.recent_trades) > 10:
            efficiency_score = self._calculate_cost_efficiency_score()
            if efficiency_score < 0.3:
                return True, f"Poor cost efficiency: {efficiency_score:.3f}"
        
        # Check recent profitability after costs
        if len(self.recent_trades) > 5:
            recent_trades = list(self.recent_trades)[-5:]
            recent_net = sum(trade.get('final_pnl', 0) - trade.get('total_cost', 0) 
                           for trade in recent_trades if 'final_pnl' in trade)
            
            if recent_net < -0.01:  # Lost more than 1% to costs recently
                return True, f"Recent net loss after costs: {recent_net:.4f}"
        
        return False, "Cost metrics acceptable"