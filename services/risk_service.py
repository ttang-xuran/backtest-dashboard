#!/usr/bin/env python3
"""
Risk Management Service
Provides comprehensive risk controls, position sizing, and portfolio protection
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger, get_trade_logger, get_performance_monitor
from config.settings import config_manager


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation: float


@dataclass
class RiskLimit:
    name: str
    current_value: float
    limit_value: float
    risk_level: RiskLevel
    description: str


class RiskService:
    def __init__(self, data_service=None):
        self.data_service = data_service
        self.logger = get_logger(__name__)
        self.trade_logger = get_trade_logger(__name__)
        self.perf_monitor = get_performance_monitor(__name__)
        self.config = config_manager.config
        
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.pnl_history: List[float] = []
        self.returns_history: List[float] = []
        
        self._risk_events: List[Dict[str, Any]] = []
        self._circuit_breaker_triggered = False
        self._circuit_breaker_time: Optional[datetime] = None
        
        self.risk_limits = self._initialize_risk_limits()
    
    def _initialize_risk_limits(self) -> Dict[str, RiskLimit]:
        return {
            'daily_loss': RiskLimit(
                name='Daily Loss Limit',
                current_value=0.0,
                limit_value=self.config.risk.max_daily_loss,
                risk_level=RiskLevel.HIGH,
                description='Maximum allowed daily loss percentage'
            ),
            'position_size': RiskLimit(
                name='Position Size Limit',
                current_value=0.0,
                limit_value=self.config.risk.max_position_size,
                risk_level=RiskLevel.MEDIUM,
                description='Maximum position size as percentage of account'
            ),
            'drawdown': RiskLimit(
                name='Drawdown Limit',
                current_value=0.0,
                limit_value=self.config.risk.drawdown_threshold,
                risk_level=RiskLevel.CRITICAL,
                description='Maximum allowed drawdown from peak'
            ),
            'correlation': RiskLimit(
                name='Correlation Threshold',
                current_value=0.0,
                limit_value=self.config.risk.correlation_threshold,
                risk_level=RiskLevel.LOW,
                description='Minimum correlation required between trading pairs'
            )
        }
    
    def reset_daily_metrics(self) -> None:
        current_date = datetime.now().date()
        if self.last_reset_date != current_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
            self.logger.info("Reset daily risk metrics")
    
    def calculate_position_size(self, account_balance: float, volatility: float, 
                              confidence: float = 1.0, signal_strength: float = 1.0) -> float:
        try:
            self.perf_monitor.start_timing("calculate_position_size")
            
            base_size = self.config.risk.max_position_size
            
            volatility_adjustment = max(0.1, 1.0 - min(volatility * 2, 0.8))
            confidence_adjustment = max(0.2, confidence)
            signal_adjustment = max(0.3, signal_strength)
            
            drawdown_adjustment = max(0.1, 1.0 - (self.current_drawdown / self.config.risk.drawdown_threshold))
            
            daily_loss_adjustment = 1.0
            if self.daily_pnl < 0:
                daily_loss_ratio = abs(self.daily_pnl) / (self.config.risk.max_daily_loss * account_balance)
                daily_loss_adjustment = max(0.1, 1.0 - daily_loss_ratio)
            
            position_size = (base_size * volatility_adjustment * confidence_adjustment * 
                           signal_adjustment * drawdown_adjustment * daily_loss_adjustment)
            
            position_size = max(position_size, 0.001)
            position_size = min(position_size, self.config.risk.max_position_size)
            
            self.perf_monitor.end_timing("calculate_position_size", 
                                       position_size=position_size,
                                       adjustments={
                                           'volatility': volatility_adjustment,
                                           'confidence': confidence_adjustment,
                                           'signal': signal_adjustment,
                                           'drawdown': drawdown_adjustment,
                                           'daily_loss': daily_loss_adjustment
                                       })
            
            self.logger.debug(f"Position size: {position_size:.6f} (adjustments: vol={volatility_adjustment:.3f}, "
                            f"conf={confidence_adjustment:.3f}, sig={signal_adjustment:.3f}, "
                            f"dd={drawdown_adjustment:.3f}, dl={daily_loss_adjustment:.3f})")
            
            return position_size
            
        except Exception as e:
            self.perf_monitor.end_timing("calculate_position_size", error=str(e))
            self.logger.error(f"Error calculating position size: {e}")
            return 0.001
    
    def check_risk_limits(self, symbol: str, side: str, size: float, 
                         current_price: float, account_balance: float) -> Tuple[bool, str]:
        try:
            self.reset_daily_metrics()
            
            if self._circuit_breaker_triggered:
                if self._should_reset_circuit_breaker():
                    self._reset_circuit_breaker()
                else:
                    return False, "Circuit breaker is active"
            
            position_value = size * current_price
            position_pct = position_value / account_balance
            
            if position_pct > self.config.risk.max_position_size:
                self._log_risk_event("position_size_exceeded", symbol, position_pct, 
                                   self.config.risk.max_position_size)
                return False, f"Position size {position_pct:.4f} exceeds limit {self.config.risk.max_position_size}"
            
            daily_loss_limit = self.config.risk.max_daily_loss * account_balance
            if self.daily_pnl < -daily_loss_limit:
                self._log_risk_event("daily_loss_exceeded", symbol, abs(self.daily_pnl), daily_loss_limit)
                self._trigger_circuit_breaker("Daily loss limit exceeded")
                return False, f"Daily loss limit exceeded: {self.daily_pnl:.4f}"
            
            if self.current_drawdown > self.config.risk.drawdown_threshold:
                self._log_risk_event("drawdown_exceeded", symbol, self.current_drawdown, 
                                   self.config.risk.drawdown_threshold)
                self._trigger_circuit_breaker("Maximum drawdown exceeded")
                return False, f"Drawdown limit exceeded: {self.current_drawdown:.4f}"
            
            if self.data_service:
                correlation = self.data_service.get_correlation("BTC", "ETH")
                if correlation is not None and correlation < self.config.risk.correlation_threshold:
                    self._log_risk_event("low_correlation", symbol, correlation, 
                                       self.config.risk.correlation_threshold)
                    return False, f"Correlation too low: {correlation:.4f} < {self.config.risk.correlation_threshold}"
            
            total_exposure = self._calculate_total_exposure(account_balance)
            max_total_exposure = self.config.risk.max_leverage
            if total_exposure > max_total_exposure:
                return False, f"Total exposure {total_exposure:.4f} exceeds limit {max_total_exposure}"
            
            return True, "Risk checks passed"
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False, f"Risk check error: {str(e)}"
    
    def _calculate_total_exposure(self, account_balance: float) -> float:
        total_exposure = 0.0
        for position_data in self.open_positions.values():
            size = position_data.get('size', 0)
            price = position_data.get('current_price', 0)
            exposure = abs(size * price) / account_balance
            total_exposure += exposure
        return total_exposure
    
    def update_position(self, symbol: str, size: float, entry_price: float, 
                       current_price: float, timestamp: Optional[datetime] = None) -> None:
        if timestamp is None:
            timestamp = datetime.now()
        
        self.open_positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'current_price': current_price,
            'unrealized_pnl': (current_price - entry_price) * size if size > 0 else (entry_price - current_price) * abs(size),
            'timestamp': timestamp,
            'holding_time': (timestamp - self.open_positions.get(symbol, {}).get('timestamp', timestamp)).total_seconds() / 60
        }
    
    def close_position(self, symbol: str, exit_price: float, timestamp: Optional[datetime] = None) -> float:
        if symbol not in self.open_positions:
            return 0.0
        
        if timestamp is None:
            timestamp = datetime.now()
        
        position = self.open_positions[symbol]
        size = position['size']
        entry_price = position['entry_price']
        
        if size > 0:  # Long position
            realized_pnl = (exit_price - entry_price) * size
        else:  # Short position
            realized_pnl = (entry_price - exit_price) * abs(size)
        
        self.daily_pnl += realized_pnl
        self.total_pnl += realized_pnl
        
        if self.total_pnl > self.peak_equity:
            self.peak_equity = self.total_pnl
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - self.total_pnl) / max(self.peak_equity, 1.0)
        
        self.pnl_history.append(realized_pnl)
        if len(self.pnl_history) > 0 and len(self.pnl_history) > 1:
            returns = realized_pnl / max(abs(self.pnl_history[-2]), 1.0)
            self.returns_history.append(returns)
        
        holding_time = (timestamp - position['timestamp']).total_seconds() / 60
        
        self.trade_logger.log_portfolio_update(
            total_pnl=self.total_pnl,
            daily_pnl=self.daily_pnl,
            positions={symbol: {
                'realized_pnl': realized_pnl,
                'holding_time_minutes': holding_time,
                'entry_price': entry_price,
                'exit_price': exit_price
            }}
        )
        
        del self.open_positions[symbol]
        self.daily_trades += 1
        
        return realized_pnl
    
    def calculate_var(self, confidence_level: float = 0.05, lookback_days: int = 30) -> float:
        if len(self.returns_history) < lookback_days:
            return 0.0
        
        recent_returns = self.returns_history[-lookback_days:]
        return np.percentile(recent_returns, confidence_level * 100)
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.05, lookback_days: int = 30) -> float:
        if len(self.returns_history) < lookback_days:
            return 0.0
        
        recent_returns = self.returns_history[-lookback_days:]
        var = self.calculate_var(confidence_level, lookback_days)
        
        tail_returns = [r for r in recent_returns if r <= var]
        return np.mean(tail_returns) if tail_returns else 0.0
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02, lookback_days: int = 252) -> float:
        if len(self.returns_history) < lookback_days:
            return 0.0
        
        recent_returns = np.array(self.returns_history[-lookback_days:])
        excess_returns = recent_returns - (risk_free_rate / 252)
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_sortino_ratio(self, target_return: float = 0.0, lookback_days: int = 252) -> float:
        if len(self.returns_history) < lookback_days:
            return 0.0
        
        recent_returns = np.array(self.returns_history[-lookback_days:])
        excess_returns = recent_returns - target_return
        
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    def get_risk_metrics(self) -> RiskMetrics:
        return RiskMetrics(
            var_95=self.calculate_var(0.05),
            var_99=self.calculate_var(0.01),
            expected_shortfall=self.calculate_expected_shortfall(0.05),
            max_drawdown=max([dd for dd in [self.current_drawdown] + 
                            [max(0, (max(self.pnl_history[:i+1]) - pnl) / max(max(self.pnl_history[:i+1]), 1))
                             for i, pnl in enumerate(self.pnl_history) if i > 0]], default=0),
            current_drawdown=self.current_drawdown,
            sharpe_ratio=self.calculate_sharpe_ratio(),
            sortino_ratio=self.calculate_sortino_ratio(),
            beta=self.data_service.get_correlation("BTC", "ETH") if self.data_service else 0.0,
            correlation=self.data_service.get_correlation("BTC", "ETH") if self.data_service else 0.0
        )
    
    def _log_risk_event(self, event_type: str, symbol: str, current_value: float, 
                       threshold: float, action: str = "position_rejected") -> None:
        risk_event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'symbol': symbol,
            'current_value': current_value,
            'threshold': threshold,
            'action': action
        }
        
        self._risk_events.append(risk_event)
        
        self.trade_logger.log_risk_event(
            event_type=event_type,
            symbol=symbol,
            current_value=current_value,
            threshold=threshold,
            action=action
        )
    
    def _trigger_circuit_breaker(self, reason: str) -> None:
        self._circuit_breaker_triggered = True
        self._circuit_breaker_time = datetime.now()
        
        self.logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
        self._log_risk_event("circuit_breaker", "ALL", 0, 0, "halt_trading")
    
    def _should_reset_circuit_breaker(self) -> bool:
        if not self._circuit_breaker_time:
            return True
        
        time_since_trigger = datetime.now() - self._circuit_breaker_time
        return time_since_trigger > timedelta(hours=1)
    
    def _reset_circuit_breaker(self) -> None:
        self._circuit_breaker_triggered = False
        self._circuit_breaker_time = None
        self.logger.info("Circuit breaker reset")
    
    def is_circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_triggered
    
    def get_risk_summary(self) -> Dict[str, Any]:
        risk_metrics = self.get_risk_metrics()
        
        self.risk_limits['daily_loss'].current_value = abs(self.daily_pnl)
        self.risk_limits['drawdown'].current_value = self.current_drawdown
        if self.data_service:
            correlation = self.data_service.get_correlation("BTC", "ETH")
            if correlation is not None:
                self.risk_limits['correlation'].current_value = correlation
        
        return {
            'risk_metrics': {
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'expected_shortfall': risk_metrics.expected_shortfall,
                'max_drawdown': risk_metrics.max_drawdown,
                'current_drawdown': risk_metrics.current_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'sortino_ratio': risk_metrics.sortino_ratio
            },
            'portfolio_metrics': {
                'total_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'peak_equity': self.peak_equity,
                'daily_trades': self.daily_trades,
                'open_positions_count': len(self.open_positions)
            },
            'risk_limits': {
                name: {
                    'current': limit.current_value,
                    'limit': limit.limit_value,
                    'utilization_pct': (limit.current_value / limit.limit_value * 100) if limit.limit_value > 0 else 0,
                    'risk_level': limit.risk_level.value,
                    'status': 'violated' if limit.current_value > limit.limit_value else 'ok'
                } for name, limit in self.risk_limits.items()
            },
            'circuit_breaker': {
                'active': self._circuit_breaker_triggered,
                'triggered_time': self._circuit_breaker_time.isoformat() if self._circuit_breaker_time else None
            },
            'recent_risk_events': [
                {
                    'timestamp': event['timestamp'].isoformat(),
                    'type': event['event_type'],
                    'symbol': event['symbol'],
                    'action': event['action']
                } for event in self._risk_events[-10:]
            ]
        }
    
    def force_reset_daily_limits(self) -> None:
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.logger.warning("Force reset daily risk limits")
    
    def emergency_stop(self, reason: str) -> None:
        self._trigger_circuit_breaker(f"Emergency stop: {reason}")
        self.logger.critical(f"🛑 EMERGENCY STOP ACTIVATED: {reason}")
        
        for symbol in list(self.open_positions.keys()):
            self.logger.critical(f"Marking position {symbol} for emergency exit")
    
    async def health_check(self) -> Dict[str, Any]:
        try:
            risk_summary = self.get_risk_summary()
            
            health_status = 'healthy'
            issues = []
            
            if self._circuit_breaker_triggered:
                health_status = 'unhealthy'
                issues.append('Circuit breaker active')
            
            if self.current_drawdown > self.config.risk.drawdown_threshold * 0.8:
                health_status = 'degraded' if health_status == 'healthy' else health_status
                issues.append('High drawdown')
            
            if abs(self.daily_pnl) > self.config.risk.max_daily_loss * 0.8:
                health_status = 'degraded' if health_status == 'healthy' else health_status
                issues.append('High daily loss')
            
            return {
                'status': health_status,
                'timestamp': datetime.now().isoformat(),
                'issues': issues,
                'risk_summary': risk_summary
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }