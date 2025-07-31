#!/usr/bin/env python3
"""
Enhanced Trading Service with Market Regime Detection and Cost Optimization
Integrates regime detection, cost optimization, and adaptive thresholds
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

from services.trading_service import TradingService, TradeSignal, SignalType, Position
from services.market_regime_service import MarketRegimeService, MarketRegime
from services.cost_optimization_service import CostOptimizationService
from utils.logger import get_logger, get_trade_logger, get_performance_monitor
from utils.error_handler import handle_errors, ErrorCategory, ErrorSeverity
from utils.circuit_breaker import trading_circuit_breaker
from utils.validators import trading_signal_validator


@dataclass
class EnhancedTradeSignal:
    base_signal: TradeSignal
    regime_analysis: Dict[str, Any]
    cost_analysis: Dict[str, Any]
    adaptive_thresholds: Dict[str, float]
    final_recommendation: str
    confidence_adjusted: float
    position_size_adjusted: float
    reasoning: List[str]


class EnhancedTradingService(TradingService):
    def __init__(self, hyperliquid_client=None, data_service=None, risk_service=None):
        super().__init__(hyperliquid_client, data_service)
        
        self.risk_service = risk_service
        self.regime_service = MarketRegimeService(data_service)
        self.cost_service = CostOptimizationService(data_service)
        
        self.logger = get_logger(__name__)
        self.trade_logger = get_trade_logger(__name__)
        self.perf_monitor = get_performance_monitor(__name__)
        
        # Enhanced configuration
        self.min_regime_confidence = 0.6
        self.min_cost_efficiency = 2.0  # 2x cost coverage minimum
        self.max_position_reduction = 0.8  # Don't reduce position by more than 80%
        
        # Trade tracking for regime service feedback
        self.active_trade_start = None
        self.active_trade_entry_signal = None
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0
        
        # Pause trading conditions
        self.trading_paused = False
        self.pause_reason = ""
        self.pause_until = None
    
    @handle_errors(retry_count=1, error_category=ErrorCategory.TRADING, severity=ErrorSeverity.HIGH)
    async def generate_enhanced_signal(self) -> EnhancedTradeSignal:
        """Generate enhanced signal with regime detection and cost optimization"""
        
        try:
            self.perf_monitor.start_timing("enhanced_signal_generation")
            
            # Check if trading should be paused
            should_pause, pause_reason = await self._check_trading_pause_conditions()
            if should_pause:
                return self._create_no_trade_signal("Trading paused", pause_reason)
            
            # Generate base signal
            base_signal = self.generate_signal()
            
            # Get regime analysis
            regime_metrics = self.regime_service.detect_market_regime()
            can_trade, regime_reason = self.regime_service.should_trade_now()
            
            if not can_trade:
                return self._create_no_trade_signal("Regime check failed", regime_reason)
            
            # Get adaptive thresholds
            adaptive_thresholds = self.regime_service.get_adaptive_thresholds()
            
            # Apply adaptive thresholds to signal
            signal_passes_adaptive = self._check_adaptive_thresholds(base_signal, adaptive_thresholds)
            if not signal_passes_adaptive:
                return self._create_no_trade_signal("Adaptive threshold check failed", 
                                                  f"Z-score {base_signal.z_score:.3f} below adaptive threshold {adaptive_thresholds['entry_threshold']:.3f}")
            
            # Cost analysis (only for entry signals)
            cost_analysis = None
            if base_signal.signal_type in [SignalType.LONG_BTC, SignalType.SHORT_BTC]:
                current_prices = await self._get_current_prices()
                if not current_prices:
                    return self._create_no_trade_signal("Price data unavailable", "Cannot get current prices")
                
                cost_analysis = self.cost_service.analyze_trade_costs(
                    "BTC/ETH",
                    self.config.trading.position_size,
                    current_prices,
                    base_signal.z_score,
                    base_signal.confidence
                )
                
                if cost_analysis.recommended_action == "SKIP":
                    return self._create_no_trade_signal("Cost analysis failed", cost_analysis.reasoning)
            
            # Create enhanced signal
            enhanced_signal = self._create_enhanced_signal(
                base_signal, regime_metrics, cost_analysis, adaptive_thresholds
            )
            
            self.perf_monitor.end_timing("enhanced_signal_generation",
                                       signal_type=enhanced_signal.base_signal.signal_type.value,
                                       final_recommendation=enhanced_signal.final_recommendation)
            
            # Log the enhanced signal
            self._log_enhanced_signal(enhanced_signal)
            
            return enhanced_signal
            
        except Exception as e:
            self.perf_monitor.end_timing("enhanced_signal_generation", error=str(e))
            self.logger.error(f"Error generating enhanced signal: {e}")
            return self._create_no_trade_signal("Signal generation error", str(e))
    
    async def _check_trading_pause_conditions(self) -> Tuple[bool, str]:
        """Check various conditions that might require pausing trading"""
        
        # Check if manually paused
        if self.trading_paused:
            if self.pause_until and datetime.now() > self.pause_until:
                self.trading_paused = False
                self.pause_reason = ""
                self.pause_until = None
                self.logger.info("Trading pause period expired, resuming trading")
            else:
                return True, self.pause_reason
        
        # Check cost-based pause conditions
        should_pause_cost, cost_reason = self.cost_service.should_pause_trading_due_to_costs()
        if should_pause_cost:
            return True, f"Cost control: {cost_reason}"
        
        # Check risk-based pause conditions
        if self.risk_service and self.risk_service.is_circuit_breaker_active():
            return True, "Risk circuit breaker active"
        
        # Check recent performance
        if await self._should_pause_due_to_performance():
            return True, "Poor recent performance"
        
        return False, "No pause conditions"
    
    async def _should_pause_due_to_performance(self) -> bool:
        """Check if recent performance warrants pausing trading"""
        
        if len(self.regime_service.trade_outcomes) < 10:
            return False
        
        recent_outcomes = list(self.regime_service.trade_outcomes)[-10:]
        
        # Check win rate
        wins = sum(1 for outcome in recent_outcomes if outcome.get('pnl', 0) > 0)
        win_rate = wins / len(recent_outcomes)
        
        if win_rate < 0.3:  # Less than 30% win rate
            return True
        
        # Check total PnL
        total_pnl = sum(outcome.get('pnl', 0) for outcome in recent_outcomes)
        if total_pnl < -0.02:  # Lost more than 2%
            return True
        
        return False
    
    def _check_adaptive_thresholds(self, signal: TradeSignal, thresholds: Dict[str, float]) -> bool:
        """Check if signal meets adaptive thresholds"""
        
        if signal.signal_type == SignalType.NO_SIGNAL:
            return True  # No thresholds to check
        
        if signal.signal_type == SignalType.EXIT:
            return abs(signal.z_score) <= thresholds.get('exit_threshold', 0.8)
        
        # Entry signal
        return abs(signal.z_score) >= thresholds.get('entry_threshold', 3.0)
    
    async def _get_current_prices(self) -> Optional[Dict[str, float]]:
        """Get current prices for cost analysis"""
        
        if not self.data_service:
            return None
        
        try:
            btc_price = await self.data_service.get_market_price("BTC")
            eth_price = await self.data_service.get_market_price("ETH")
            
            if btc_price and eth_price:
                return {"BTC": btc_price, "ETH": eth_price}
            
        except Exception as e:
            self.logger.error(f"Error getting current prices: {e}")
        
        return None
    
    def _create_enhanced_signal(self, base_signal: TradeSignal, regime_metrics, 
                              cost_analysis, adaptive_thresholds) -> EnhancedTradeSignal:
        """Create enhanced signal with all analysis"""
        
        reasoning = []
        final_recommendation = "TRADE"
        confidence_adjusted = base_signal.confidence
        position_size_adjusted = self.config.trading.position_size
        
        # Regime-based adjustments
        if regime_metrics.market_regime == MarketRegime.CHOPPY:
            confidence_adjusted *= 0.7
            position_size_adjusted *= 0.5
            reasoning.append("Reduced confidence and size due to choppy market")
        elif regime_metrics.market_regime == MarketRegime.TRENDING:
            if not self.contrarian_mode:
                confidence_adjusted *= 0.8
                reasoning.append("Reduced confidence in trending market for mean reversion")
        
        # Cost-based adjustments
        if cost_analysis:
            if cost_analysis.recommended_action == "REDUCE_SIZE":
                position_size_adjusted = cost_analysis.cost_adjusted_position_size
                reasoning.append(f"Position size adjusted for cost efficiency: {cost_analysis.reasoning}")
            elif cost_analysis.recommended_action == "TRADE_SMALL":
                position_size_adjusted *= 0.7
                reasoning.append("Reduced position size due to marginal cost efficiency")
        
        # Correlation strength adjustment
        if abs(regime_metrics.current_correlation) < 0.8:
            correlation_adjustment = abs(regime_metrics.current_correlation) / 0.8
            confidence_adjusted *= correlation_adjustment
            position_size_adjusted *= correlation_adjustment
            reasoning.append(f"Adjusted for correlation strength: {regime_metrics.current_correlation:.3f}")
        
        # Volatility adjustment
        if regime_metrics.volatility_regime == "HIGH":
            position_size_adjusted *= 0.8
            reasoning.append("Reduced size due to high volatility")
        elif regime_metrics.volatility_regime == "EXTREME_HIGH":
            final_recommendation = "SKIP"
            reasoning.append("Skipping due to extreme volatility")
        
        # Final validation
        if position_size_adjusted < 0.001:  # Too small to trade
            final_recommendation = "SKIP"
            reasoning.append("Position size too small after adjustments")
        
        if confidence_adjusted < 0.3:  # Too low confidence
            final_recommendation = "SKIP"
            reasoning.append("Confidence too low after adjustments")
        
        return EnhancedTradeSignal(
            base_signal=base_signal,
            regime_analysis={
                'market_regime': regime_metrics.market_regime.value,
                'correlation_regime': regime_metrics.correlation_regime.value,
                'current_correlation': regime_metrics.current_correlation,
                'confidence': regime_metrics.confidence,
                'tradeable': regime_metrics.tradeable
            },
            cost_analysis={
                'total_cost': cost_analysis.estimated_total_cost if cost_analysis else 0,
                'min_profit_target': cost_analysis.min_profit_target if cost_analysis else 0,
                'cost_per_hour': cost_analysis.cost_per_hour if cost_analysis else 0,
                'recommendation': cost_analysis.recommended_action if cost_analysis else "N/A"
            },
            adaptive_thresholds=adaptive_thresholds,
            final_recommendation=final_recommendation,
            confidence_adjusted=confidence_adjusted,
            position_size_adjusted=position_size_adjusted,
            reasoning=reasoning
        )
    
    def _create_no_trade_signal(self, primary_reason: str, secondary_reason: str) -> EnhancedTradeSignal:
        """Create a no-trade signal with reasoning"""
        
        no_signal = TradeSignal(
            SignalType.NO_SIGNAL, 0.0, 0.0, datetime.now(), 
            {'reason': primary_reason}
        )
        
        return EnhancedTradeSignal(
            base_signal=no_signal,
            regime_analysis={},
            cost_analysis={},
            adaptive_thresholds={},
            final_recommendation="SKIP",
            confidence_adjusted=0.0,
            position_size_adjusted=0.0,
            reasoning=[primary_reason, secondary_reason]
        )
    
    def _log_enhanced_signal(self, signal: EnhancedTradeSignal):
        """Log enhanced signal details"""
        
        self.trade_logger.log_trade_signal(
            symbol_pair="BTC/ETH",
            signal_type=signal.base_signal.signal_type.value,
            z_score=signal.base_signal.z_score,
            price_data={},
            confidence=signal.confidence_adjusted,
            metadata={
                'original_confidence': signal.base_signal.confidence,
                'regime': signal.regime_analysis.get('market_regime', 'unknown'),
                'correlation': signal.regime_analysis.get('current_correlation', 0),
                'adaptive_entry_threshold': signal.adaptive_thresholds.get('entry_threshold', 0),
                'position_size_adjusted': signal.position_size_adjusted,
                'cost_analysis': signal.cost_analysis,
                'final_recommendation': signal.final_recommendation,
                'reasoning': signal.reasoning
            }
        )
    
    @trading_circuit_breaker
    async def execute_enhanced_signal(self, enhanced_signal: EnhancedTradeSignal) -> bool:
        """Execute enhanced signal with all safeguards"""
        
        if enhanced_signal.final_recommendation == "SKIP":
            self.logger.info(f"Skipping trade: {'; '.join(enhanced_signal.reasoning)}")
            return False
        
        if enhanced_signal.base_signal.signal_type == SignalType.NO_SIGNAL:
            return False
        
        try:
            # Store signal for tracking
            if enhanced_signal.base_signal.signal_type in [SignalType.LONG_BTC, SignalType.SHORT_BTC]:
                self.active_trade_entry_signal = enhanced_signal
                self.active_trade_start = datetime.now()
                self.max_favorable_excursion = 0.0
                self.max_adverse_excursion = 0.0
            
            # Override position size with adjusted size
            original_position_size = self.config.trading.position_size
            self.config.trading.position_size = enhanced_signal.position_size_adjusted
            
            # Execute the trade
            success = await self.execute_signal(enhanced_signal.base_signal)
            
            # Restore original position size
            self.config.trading.position_size = original_position_size
            
            if success:
                self.logger.info(f"Successfully executed enhanced signal: {enhanced_signal.base_signal.signal_type.value}")
                
                # Record cost service execution
                if enhanced_signal.cost_analysis and 'total_cost' in enhanced_signal.cost_analysis:
                    current_prices = await self._get_current_prices()
                    if current_prices:
                        self.cost_service.record_trade_execution(
                            enhanced_signal.position_size_adjusted,
                            {'total_cost': enhanced_signal.cost_analysis['total_cost']},
                            datetime.now(),
                            0.0005  # Estimated slippage
                        )
            else:
                self.logger.error(f"Failed to execute enhanced signal: {enhanced_signal.base_signal.signal_type.value}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing enhanced signal: {e}")
            return False
    
    async def _execute_enhanced_exit(self, enhanced_signal: EnhancedTradeSignal) -> bool:
        """Execute exit with tracking for regime service"""
        
        if not self.is_trading:
            return False
        
        entry_time = self.active_trade_start or self.entry_time
        hold_time = (datetime.now() - entry_time).total_seconds() / 3600 if entry_time else 0
        
        # Calculate current P&L for tracking
        current_pnl = await self._calculate_current_unrealized_pnl()
        
        # Update excursion tracking
        if current_pnl > self.max_favorable_excursion:
            self.max_favorable_excursion = current_pnl
        if current_pnl < self.max_adverse_excursion:
            self.max_adverse_excursion = current_pnl
        
        # Execute exit
        success = await self._execute_exit()
        
        if success and self.active_trade_entry_signal:
            # Record trade outcome for regime service
            self.regime_service.record_trade_outcome(
                entry_signal={
                    'z_score': self.active_trade_entry_signal.base_signal.z_score,
                    'confidence': self.active_trade_entry_signal.confidence_adjusted,
                    'regime': self.active_trade_entry_signal.regime_analysis.get('market_regime', 'unknown')
                },
                exit_result={
                    'success': True,
                    'pnl': current_pnl,
                    'hold_time_minutes': hold_time * 60,
                    'max_favorable_excursion': self.max_favorable_excursion,
                    'max_adverse_excursion': self.max_adverse_excursion,
                    'exit_reason': 'signal_exit'
                }
            )
            
            # Record for cost service
            self.cost_service.record_trade_exit(
                hold_time, current_pnl,
                {'exit_cost': enhanced_signal.cost_analysis.get('total_cost', 0) * 0.5}  # Estimate exit cost
            )
            
            # Reset tracking
            self.active_trade_entry_signal = None
            self.active_trade_start = None
        
        return success
    
    async def _calculate_current_unrealized_pnl(self) -> float:
        """Calculate current unrealized P&L"""
        
        if not self.is_trading or not self.positions:
            return 0.0
        
        total_pnl = 0.0
        
        try:
            for symbol, position in self.positions.items():
                current_price = await self.data_service.get_market_price(symbol)
                if current_price:
                    if position.size > 0:  # Long position
                        pnl = (current_price - position.entry_price) * position.size
                    else:  # Short position  
                        pnl = (position.entry_price - current_price) * abs(position.size)
                    
                    total_pnl += pnl
        
        except Exception as e:
            self.logger.error(f"Error calculating unrealized PnL: {e}")
        
        return total_pnl
    
    async def force_exit_with_tracking(self, reason: str) -> bool:
        """Force exit with proper tracking"""
        
        current_pnl = await self._calculate_current_unrealized_pnl()
        hold_time = 0
        
        if self.active_trade_start:
            hold_time = (datetime.now() - self.active_trade_start).total_seconds() / 3600
        
        success = await self.force_exit_all_positions(reason)
        
        if success and self.active_trade_entry_signal:
            # Record forced exit
            self.regime_service.record_trade_outcome(
                entry_signal={
                    'z_score': self.active_trade_entry_signal.base_signal.z_score,
                    'confidence': self.active_trade_entry_signal.confidence_adjusted
                },
                exit_result={
                    'success': False,
                    'pnl': current_pnl,
                    'hold_time_minutes': hold_time * 60,
                    'max_favorable_excursion': self.max_favorable_excursion,
                    'max_adverse_excursion': self.max_adverse_excursion,
                    'exit_reason': reason
                }
            )
            
            self.cost_service.record_trade_exit(hold_time, current_pnl, {'exit_cost': 0.001})
            
            # Reset tracking
            self.active_trade_entry_signal = None
            self.active_trade_start = None
        
        return success
    
    def pause_trading(self, reason: str, duration_hours: float = 1.0):
        """Manually pause trading for specified duration"""
        
        self.trading_paused = True
        self.pause_reason = reason
        self.pause_until = datetime.now() + timedelta(hours=duration_hours)
        
        self.logger.warning(f"Trading paused for {duration_hours:.1f} hours: {reason}")
    
    def resume_trading(self):
        """Manually resume trading"""
        
        self.trading_paused = False
        self.pause_reason = ""
        self.pause_until = None
        
        self.logger.info("Trading manually resumed")
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced trading status"""
        
        base_status = self.get_trading_stats()
        regime_summary = self.regime_service.get_regime_summary()
        cost_report = self.cost_service.get_cost_efficiency_report()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'trading_paused': self.trading_paused,
            'pause_reason': self.pause_reason,
            'pause_until': self.pause_until.isoformat() if self.pause_until else None,
            'base_trading_stats': base_status,
            'regime_analysis': regime_summary,
            'cost_efficiency': cost_report,
            'current_trade_tracking': {
                'has_active_trade': self.active_trade_start is not None,
                'trade_duration_hours': (datetime.now() - self.active_trade_start).total_seconds() / 3600 
                                       if self.active_trade_start else 0,
                'max_favorable_excursion': self.max_favorable_excursion,
                'max_adverse_excursion': self.max_adverse_excursion
            }
        }