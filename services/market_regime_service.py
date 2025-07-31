#!/usr/bin/env python3
"""
Market Regime Detection Service  
Identifies market conditions to determine when statistical arbitrage is viable
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from collections import deque

from utils.logger import get_logger, get_performance_monitor
from utils.error_handler import handle_errors, ErrorCategory, ErrorSeverity


class MarketRegime(Enum):
    MEAN_REVERTING = "mean_reverting"
    TRENDING = "trending" 
    CHOPPY = "choppy"
    UNCERTAIN = "uncertain"


class CorrelationRegime(Enum):
    STABLE_POSITIVE = "stable_positive"
    STABLE_NEGATIVE = "stable_negative"
    UNSTABLE = "unstable"
    BREAKDOWN = "breakdown"


@dataclass
class RegimeMetrics:
    market_regime: MarketRegime
    correlation_regime: CorrelationRegime
    current_correlation: float
    correlation_stability: float
    hurst_btc: float
    hurst_eth: float
    volatility_regime: str
    tradeable: bool
    confidence: float
    regime_duration_hours: float


@dataclass 
class SignalQuality:
    recent_success_rate: float
    avg_favorable_excursion: float
    avg_hold_time: float
    avg_slippage: float
    win_rate: float
    profit_factor: float
    recent_trades_count: int


class MarketRegimeService:
    def __init__(self, data_service=None):
        self.data_service = data_service
        self.logger = get_logger(__name__)
        self.perf_monitor = get_performance_monitor(__name__)
        
        # Regime tracking
        self.current_regime = MarketRegime.UNCERTAIN
        self.regime_start_time = datetime.now()
        self.regime_history = deque(maxlen=100)
        
        # Correlation tracking  
        self.correlation_history = deque(maxlen=100)
        self.rolling_correlation_window = 14  # 14 periods
        
        # Signal quality tracking
        self.signal_history = deque(maxlen=50)
        self.trade_outcomes = deque(maxlen=30)
        
        # Configuration
        self.min_data_points = 30
        self.regime_change_threshold = 0.3
        self.correlation_stability_threshold = 0.2
        self.min_correlation_strength = 0.7
        
        # Volatility percentile tracking
        self.volatility_history = deque(maxlen=252)  # ~1 year of daily data
    
    @handle_errors(retry_count=1, error_category=ErrorCategory.CALCULATION, severity=ErrorSeverity.MEDIUM)
    def detect_market_regime(self, lookback_periods: int = 50) -> RegimeMetrics:
        """Comprehensive market regime detection"""
        
        if not self.data_service:
            return self._create_uncertain_regime()
        
        try:
            self.perf_monitor.start_timing("regime_detection")
            
            # Get data
            btc_returns = self.data_service.get_returns_series("BTC", lookback_periods)
            eth_returns = self.data_service.get_returns_series("ETH", lookback_periods)
            
            if len(btc_returns) < self.min_data_points or len(eth_returns) < self.min_data_points:
                self.logger.warning(f"Insufficient data for regime detection: BTC={len(btc_returns)}, ETH={len(eth_returns)}")
                return self._create_uncertain_regime()
            
            # Calculate regime components
            current_correlation = self._calculate_current_correlation(btc_returns, eth_returns)
            correlation_stability = self._calculate_correlation_stability(lookback_periods)
            correlation_regime = self._classify_correlation_regime(current_correlation, correlation_stability)
            
            hurst_btc = self._estimate_hurst_exponent(btc_returns)
            hurst_eth = self._estimate_hurst_exponent(eth_returns)
            market_regime = self._classify_market_regime(hurst_btc, hurst_eth, correlation_stability)
            
            volatility_regime = self._assess_volatility_regime(btc_returns, eth_returns)
            
            # Determine if market is tradeable
            tradeable, confidence = self._assess_tradeability(
                market_regime, correlation_regime, current_correlation, 
                correlation_stability, volatility_regime
            )
            
            regime_duration = self._calculate_regime_duration()
            
            metrics = RegimeMetrics(
                market_regime=market_regime,
                correlation_regime=correlation_regime,
                current_correlation=current_correlation,
                correlation_stability=correlation_stability,
                hurst_btc=hurst_btc,
                hurst_eth=hurst_eth,
                volatility_regime=volatility_regime,
                tradeable=tradeable,
                confidence=confidence,
                regime_duration_hours=regime_duration
            )
            
            self._update_regime_history(metrics)
            
            self.perf_monitor.end_timing("regime_detection", 
                                       regime=market_regime.value,
                                       tradeable=tradeable,
                                       confidence=confidence)
            
            self.logger.info(f"Regime detected: {market_regime.value} | "
                           f"Correlation: {correlation_regime.value} ({current_correlation:.3f}) | "
                           f"Tradeable: {tradeable} | Confidence: {confidence:.3f}")
            
            return metrics
            
        except Exception as e:
            self.perf_monitor.end_timing("regime_detection", error=str(e))
            self.logger.error(f"Error in regime detection: {e}")
            return self._create_uncertain_regime()
    
    def _calculate_current_correlation(self, btc_returns: np.ndarray, eth_returns: np.ndarray) -> float:
        """Calculate current correlation between BTC and ETH"""
        if len(btc_returns) < 2 or len(eth_returns) < 2:
            return 0.0
        
        correlation_matrix = np.corrcoef(btc_returns, eth_returns)
        correlation = correlation_matrix[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        self.correlation_history.append({
            'timestamp': datetime.now(),
            'correlation': correlation
        })
        
        return correlation
    
    def _calculate_correlation_stability(self, lookback_periods: int) -> float:
        """Calculate how stable the correlation has been"""
        if not self.data_service:
            return 0.0
        
        try:
            # Calculate rolling correlations
            rolling_correlations = []
            window_size = max(10, lookback_periods // 5)  # Use 1/5 of lookback as window
            
            for i in range(window_size, lookback_periods):
                btc_window = self.data_service.get_returns_series("BTC", i)[-window_size:]
                eth_window = self.data_service.get_returns_series("ETH", i)[-window_size:]
                
                if len(btc_window) == window_size and len(eth_window) == window_size:
                    corr = np.corrcoef(btc_window, eth_window)[0, 1]
                    if not np.isnan(corr):
                        rolling_correlations.append(corr)
            
            if len(rolling_correlations) < 3:
                return 0.0
            
            # Stability is inverse of standard deviation
            stability = 1.0 - min(1.0, np.std(rolling_correlations) * 2)
            return max(0.0, stability)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation stability: {e}")
            return 0.0
    
    def _classify_correlation_regime(self, correlation: float, stability: float) -> CorrelationRegime:
        """Classify the correlation regime"""
        
        if stability < 0.5:  # Very unstable
            if abs(correlation) < 0.3:
                return CorrelationRegime.BREAKDOWN
            else:
                return CorrelationRegime.UNSTABLE
        
        if correlation > 0.7 and stability > 0.7:
            return CorrelationRegime.STABLE_POSITIVE
        elif correlation < -0.7 and stability > 0.7:
            return CorrelationRegime.STABLE_NEGATIVE
        else:
            return CorrelationRegime.UNSTABLE
    
    def _estimate_hurst_exponent(self, returns: np.ndarray) -> float:
        """Estimate Hurst exponent to determine trend vs mean reversion"""
        if len(returns) < 20:
            return 0.5  # Neutral
        
        try:
            # Simple R/S analysis
            cumulative = np.cumsum(returns - np.mean(returns))
            
            lags = np.arange(2, min(len(returns)//4, 50))
            rs_values = []
            
            for lag in lags:
                # Split series into non-overlapping segments
                n_segments = len(cumulative) // lag
                if n_segments < 2:
                    continue
                
                rs_segment = []
                for i in range(n_segments):
                    start_idx = i * lag
                    end_idx = (i + 1) * lag
                    segment = cumulative[start_idx:end_idx]
                    
                    if len(segment) == lag:
                        range_val = np.max(segment) - np.min(segment)
                        std_val = np.std(returns[start_idx:end_idx])
                        
                        if std_val > 0:
                            rs_segment.append(range_val / std_val)
                
                if rs_segment:
                    rs_values.append(np.mean(rs_segment))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Fit log(R/S) vs log(lag) to get Hurst exponent
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Remove any inf or nan values
            valid_idx = np.isfinite(log_lags) & np.isfinite(log_rs)
            if np.sum(valid_idx) < 3:
                return 0.5
            
            hurst = np.polyfit(log_lags[valid_idx], log_rs[valid_idx], 1)[0]
            
            # Clamp to reasonable range
            return np.clip(hurst, 0.1, 0.9)
            
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5
    
    def _classify_market_regime(self, hurst_btc: float, hurst_eth: float, correlation_stability: float) -> MarketRegime:
        """Classify market regime based on Hurst exponents and correlation"""
        
        avg_hurst = (hurst_btc + hurst_eth) / 2
        
        # If correlation is very unstable, market is choppy regardless of Hurst
        if correlation_stability < 0.4:
            return MarketRegime.CHOPPY
        
        # Mean reverting: Hurst < 0.5, stable correlation
        if avg_hurst < 0.45 and correlation_stability > 0.7:
            return MarketRegime.MEAN_REVERTING
        
        # Trending: Hurst > 0.55, some correlation stability
        elif avg_hurst > 0.55 and correlation_stability > 0.5:
            return MarketRegime.TRENDING
        
        # Choppy: mixed signals or unstable conditions
        elif correlation_stability < 0.6 or abs(avg_hurst - 0.5) < 0.1:
            return MarketRegime.CHOPPY
        
        else:
            return MarketRegime.UNCERTAIN
    
    def _assess_volatility_regime(self, btc_returns: np.ndarray, eth_returns: np.ndarray) -> str:
        """Assess current volatility regime"""
        
        if len(btc_returns) < 10:
            return "INSUFFICIENT_DATA"
        
        # Calculate current volatility (annualized)
        btc_vol = np.std(btc_returns) * np.sqrt(252 * 24)  # Assuming hourly data
        eth_vol = np.std(eth_returns) * np.sqrt(252 * 24)
        avg_vol = (btc_vol + eth_vol) / 2
        
        self.volatility_history.append(avg_vol)
        
        if len(self.volatility_history) < 20:
            return "NORMAL"
        
        # Calculate percentile
        vol_percentile = np.percentile(list(self.volatility_history), 
                                     [(avg_vol <= v) * 100 for v in self.volatility_history].count(True))
        
        if vol_percentile > 90:
            return "EXTREME_HIGH"
        elif vol_percentile > 75:
            return "HIGH"
        elif vol_percentile < 25:
            return "LOW"
        else:
            return "NORMAL"
    
    def _assess_tradeability(self, market_regime: MarketRegime, correlation_regime: CorrelationRegime,
                           current_correlation: float, correlation_stability: float, 
                           volatility_regime: str) -> Tuple[bool, float]:
        """Determine if market conditions are suitable for statistical arbitrage"""
        
        confidence = 0.0
        tradeable = False
        
        # Base scoring
        regime_scores = {
            MarketRegime.MEAN_REVERTING: 0.8,
            MarketRegime.TRENDING: 0.3,  # Can work with contrarian mode  
            MarketRegime.CHOPPY: 0.1,   # Very difficult
            MarketRegime.UNCERTAIN: 0.2
        }
        
        correlation_scores = {
            CorrelationRegime.STABLE_POSITIVE: 0.9,
            CorrelationRegime.STABLE_NEGATIVE: 0.9,
            CorrelationRegime.UNSTABLE: 0.3,
            CorrelationRegime.BREAKDOWN: 0.1
        }
        
        volatility_scores = {
            "LOW": 0.9,
            "NORMAL": 0.8, 
            "HIGH": 0.5,
            "EXTREME_HIGH": 0.2,
            "INSUFFICIENT_DATA": 0.5
        }
        
        # Calculate base confidence
        confidence = (
            regime_scores.get(market_regime, 0.2) * 0.4 +
            correlation_scores.get(correlation_regime, 0.2) * 0.4 +
            volatility_scores.get(volatility_regime, 0.5) * 0.2
        )
        
        # Additional filters
        if abs(current_correlation) < self.min_correlation_strength:
            confidence *= 0.5
        
        if correlation_stability < self.correlation_stability_threshold:
            confidence *= 0.3
        
        # Signal quality adjustment
        signal_quality = self._get_recent_signal_quality()
        if signal_quality.recent_success_rate < 0.4:
            confidence *= 0.5
        elif signal_quality.recent_success_rate > 0.7:
            confidence *= 1.2
        
        # Recent regime stability bonus
        if self._is_regime_stable():
            confidence *= 1.1
        
        confidence = min(1.0, confidence)
        tradeable = confidence > 0.5
        
        return tradeable, confidence
    
    def _get_recent_signal_quality(self) -> SignalQuality:
        """Analyze recent signal performance"""
        
        if len(self.trade_outcomes) < 5:
            return SignalQuality(
                recent_success_rate=0.5,
                avg_favorable_excursion=0.0,
                avg_hold_time=60.0,
                avg_slippage=0.001,
                win_rate=0.5,
                profit_factor=1.0,
                recent_trades_count=0
            )
        
        recent_trades = list(self.trade_outcomes)[-10:]  # Last 10 trades
        
        success_count = sum(1 for trade in recent_trades if trade.get('success', False))
        success_rate = success_count / len(recent_trades)
        
        profitable_trades = [t for t in recent_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in recent_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(profitable_trades) / len(recent_trades)
        
        total_profits = sum(t.get('pnl', 0) for t in profitable_trades)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
        
        profit_factor = (total_profits / total_losses) if total_losses > 0 else 1.0
        
        avg_favorable_excursion = np.mean([t.get('max_favorable_excursion', 0) for t in recent_trades])
        avg_hold_time = np.mean([t.get('hold_time_minutes', 60) for t in recent_trades])
        avg_slippage = np.mean([t.get('slippage', 0.001) for t in recent_trades])
        
        return SignalQuality(
            recent_success_rate=success_rate,
            avg_favorable_excursion=avg_favorable_excursion,
            avg_hold_time=avg_hold_time,
            avg_slippage=avg_slippage,
            win_rate=win_rate,
            profit_factor=profit_factor,
            recent_trades_count=len(recent_trades)
        )
    
    def _is_regime_stable(self) -> bool:
        """Check if current regime has been stable recently"""
        if len(self.regime_history) < 3:
            return False
        
        recent_regimes = [r.market_regime for r in list(self.regime_history)[-5:]]
        
        # Check if regime has been consistent
        most_common_regime = max(set(recent_regimes), key=recent_regimes.count)
        consistency = recent_regimes.count(most_common_regime) / len(recent_regimes)
        
        return consistency >= 0.8
    
    def _calculate_regime_duration(self) -> float:
        """Calculate how long current regime has been active"""
        duration = datetime.now() - self.regime_start_time
        return duration.total_seconds() / 3600  # Return hours
    
    def _update_regime_history(self, metrics: RegimeMetrics):
        """Update regime tracking history"""
        
        # Check if regime changed
        if metrics.market_regime != self.current_regime:
            self.logger.info(f"Regime change detected: {self.current_regime.value} -> {metrics.market_regime.value}")
            self.current_regime = metrics.market_regime
            self.regime_start_time = datetime.now()
        
        self.regime_history.append(metrics)
    
    def _create_uncertain_regime(self) -> RegimeMetrics:
        """Create default uncertain regime metrics"""
        return RegimeMetrics(
            market_regime=MarketRegime.UNCERTAIN,
            correlation_regime=CorrelationRegime.UNSTABLE,
            current_correlation=0.0,
            correlation_stability=0.0,
            hurst_btc=0.5,
            hurst_eth=0.5,
            volatility_regime="INSUFFICIENT_DATA",
            tradeable=False,
            confidence=0.0,
            regime_duration_hours=0.0
        )
    
    def record_trade_outcome(self, entry_signal: Dict, exit_result: Dict):
        """Record trade outcome for signal quality analysis"""
        
        outcome = {
            'timestamp': datetime.now(),
            'entry_z_score': entry_signal.get('z_score', 0),
            'entry_confidence': entry_signal.get('confidence', 0.5),
            'success': exit_result.get('success', False),
            'pnl': exit_result.get('pnl', 0),
            'hold_time_minutes': exit_result.get('hold_time_minutes', 60),
            'max_favorable_excursion': exit_result.get('max_favorable_excursion', 0),
            'max_adverse_excursion': exit_result.get('max_adverse_excursion', 0),
            'slippage': exit_result.get('slippage', 0.001),
            'exit_reason': exit_result.get('exit_reason', 'unknown')
        }
        
        self.trade_outcomes.append(outcome)
        
        self.logger.info(f"Recorded trade outcome: PnL={outcome['pnl']:.4f}, "
                        f"Success={outcome['success']}, Reason={outcome['exit_reason']}")
    
    def get_adaptive_thresholds(self, base_entry: float = 3.0, base_exit: float = 0.8) -> Dict[str, float]:
        """Get adaptive Z-score thresholds based on current regime"""
        
        regime_metrics = self.detect_market_regime()
        
        # Base multipliers for different regimes
        regime_multipliers = {
            MarketRegime.MEAN_REVERTING: {"entry": 0.8, "exit": 1.0},    # More aggressive
            MarketRegime.TRENDING: {"entry": 1.5, "exit": 1.3},          # Much more conservative
            MarketRegime.CHOPPY: {"entry": 2.0, "exit": 0.6},            # Very conservative entry, quick exit
            MarketRegime.UNCERTAIN: {"entry": 999, "exit": 0.3}          # Don't trade
        }
        
        multiplier = regime_multipliers.get(regime_metrics.market_regime, {"entry": 999, "exit": 0.3})
        
        adaptive_entry = base_entry * multiplier["entry"]
        adaptive_exit = base_exit * multiplier["exit"]
        
        # Additional adjustments based on confidence and signal quality
        if regime_metrics.confidence < 0.6:
            adaptive_entry *= 1.5  # Be more conservative
        
        signal_quality = self._get_recent_signal_quality()
        if signal_quality.recent_success_rate < 0.5:
            adaptive_entry *= 1.3
        
        return {
            "entry_threshold": min(adaptive_entry, 999),  # Cap at 999 (effectively no trading)
            "exit_threshold": adaptive_exit,
            "regime": regime_metrics.market_regime.value,
            "confidence": regime_metrics.confidence,
            "reasoning": f"Regime: {regime_metrics.market_regime.value}, Confidence: {regime_metrics.confidence:.3f}"
        }
    
    def should_trade_now(self) -> Tuple[bool, str]:
        """Comprehensive check if trading should be allowed right now"""
        
        regime_metrics = self.detect_market_regime()
        
        if not regime_metrics.tradeable:
            return False, f"Market regime not suitable: {regime_metrics.market_regime.value}"
        
        if regime_metrics.confidence < 0.5:
            return False, f"Low confidence: {regime_metrics.confidence:.3f}"
        
        if abs(regime_metrics.current_correlation) < self.min_correlation_strength:
            return False, f"Correlation too weak: {regime_metrics.current_correlation:.3f}"
        
        if regime_metrics.correlation_stability < self.correlation_stability_threshold:
            return False, f"Correlation unstable: {regime_metrics.correlation_stability:.3f}"
        
        if regime_metrics.volatility_regime == "EXTREME_HIGH":
            return False, "Volatility too high for safe trading"
        
        signal_quality = self._get_recent_signal_quality()
        if signal_quality.recent_success_rate < 0.4 and signal_quality.recent_trades_count > 5:
            return False, f"Recent signal quality poor: {signal_quality.recent_success_rate:.3f}"
        
        return True, f"Conditions favorable - {regime_metrics.market_regime.value} regime"
    
    def get_regime_summary(self) -> Dict:
        """Get comprehensive regime analysis summary"""
        
        regime_metrics = self.detect_market_regime()
        signal_quality = self._get_recent_signal_quality()
        thresholds = self.get_adaptive_thresholds()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'market_regime': regime_metrics.market_regime.value,
            'correlation_regime': regime_metrics.correlation_regime.value,
            'current_correlation': regime_metrics.current_correlation,
            'correlation_stability': regime_metrics.correlation_stability,
            'hurst_exponents': {
                'btc': regime_metrics.hurst_btc,
                'eth': regime_metrics.hurst_eth
            },
            'volatility_regime': regime_metrics.volatility_regime,
            'tradeable': regime_metrics.tradeable,
            'confidence': regime_metrics.confidence,
            'regime_duration_hours': regime_metrics.regime_duration_hours,
            'adaptive_thresholds': thresholds,
            'signal_quality': {
                'recent_success_rate': signal_quality.recent_success_rate,
                'win_rate': signal_quality.win_rate,
                'profit_factor': signal_quality.profit_factor,
                'avg_hold_time': signal_quality.avg_hold_time,
                'recent_trades_count': signal_quality.recent_trades_count
            },
            'recommendation': 'TRADE' if regime_metrics.tradeable else 'PAUSE'
        }