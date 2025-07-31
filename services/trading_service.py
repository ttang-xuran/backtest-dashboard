#!/usr/bin/env python3
"""
Trading Service
Handles all trading logic, order execution, and position management
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import json

from utils.logger import get_logger, get_trade_logger, get_performance_monitor
from config.settings import config_manager


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class SignalType(Enum):
    LONG_BTC = "long_btc"
    SHORT_BTC = "short_btc"
    EXIT = "exit"
    NO_SIGNAL = "no_signal"


@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    side: str
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class TradeSignal:
    signal_type: SignalType
    z_score: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]


class TradingService:
    def __init__(self, hyperliquid_client=None, data_service=None):
        self.hl = hyperliquid_client
        self.data_service = data_service
        self.logger = get_logger(__name__)
        self.trade_logger = get_trade_logger(__name__)
        self.perf_monitor = get_performance_monitor(__name__)
        self.config = config_manager.config
        
        self.positions: Dict[str, Position] = {}
        self.is_trading = False
        self.entry_time: Optional[datetime] = None
        self.entry_prices: Dict[str, float] = {}
        
        self.daily_trades_count = 0
        self.last_trade_date = datetime.now().date()
        self.beta_eth_btc = 1.0
        self.min_beta_periods = 10
        
        self.contrarian_mode = self.config.trading.__dict__.get('contrarian_mode', True)
        self.daily_trade_limit = self.config.trading.__dict__.get('daily_trade_limit', 20)
        self.min_holding_minutes = self.config.trading.__dict__.get('min_holding_minutes', 5)
        self.max_holding_minutes = self.config.trading.__dict__.get('max_holding_minutes', 120)
        
        self._trading_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0
        }
    
    def can_trade_today(self) -> bool:
        current_date = datetime.now().date()
        
        if self.last_trade_date != current_date:
            self.daily_trades_count = 0
            self.last_trade_date = current_date
            self._trading_stats['daily_pnl'] = 0.0
        
        if self.daily_trades_count >= self.daily_trade_limit:
            self.logger.warning(f"Daily trade limit reached: {self.daily_trades_count}/{self.daily_trade_limit}")
            return False
        
        return True
    
    def update_beta(self) -> None:
        if not self.data_service:
            return
        
        btc_returns = self.data_service.get_returns_series("BTC")
        eth_returns = self.data_service.get_returns_series("ETH")
        
        if len(btc_returns) < self.min_beta_periods or len(eth_returns) < self.min_beta_periods:
            return
        
        try:
            covariance = np.cov(eth_returns, btc_returns)[0, 1]
            btc_variance = np.var(btc_returns)
            
            if btc_variance > 0:
                self.beta_eth_btc = covariance / btc_variance
                self.logger.debug(f"Updated beta ETH/BTC: {self.beta_eth_btc:.4f}")
            else:
                self.beta_eth_btc = 1.0
                
        except Exception as e:
            self.logger.error(f"Error calculating beta: {e}")
            self.beta_eth_btc = 1.0
    
    def get_position_sizes(self, base_position_size: float) -> Tuple[float, float]:
        btc_size = base_position_size
        eth_size = base_position_size * abs(self.beta_eth_btc)
        return btc_size, eth_size
    
    def generate_signal(self) -> TradeSignal:
        if not self.data_service:
            return TradeSignal(SignalType.NO_SIGNAL, 0.0, 0.0, datetime.now(), {})
        
        try:
            z_score = self.data_service.calculate_z_score("BTC", "ETH")
            
            if z_score is None:
                return TradeSignal(SignalType.NO_SIGNAL, 0.0, 0.0, datetime.now(), 
                                 {'reason': 'insufficient_data'})
            
            correlation = self.data_service.get_correlation("BTC", "ETH")
            btc_volatility = self.data_service.get_volatility("BTC")
            eth_volatility = self.data_service.get_volatility("ETH")
            
            confidence = self._calculate_signal_confidence(z_score, correlation, btc_volatility, eth_volatility)
            
            metadata = {
                'z_score': z_score,
                'correlation': correlation,
                'btc_volatility': btc_volatility,
                'eth_volatility': eth_volatility,
                'beta': self.beta_eth_btc,
                'contrarian_mode': self.contrarian_mode
            }
            
            if self.is_trading:
                if abs(z_score) <= self.config.trading.z_score_exit:
                    return TradeSignal(SignalType.EXIT, z_score, confidence, datetime.now(), metadata)
            else:
                if not self.can_trade_today():
                    return TradeSignal(SignalType.NO_SIGNAL, z_score, confidence, datetime.now(), 
                                     {**metadata, 'reason': 'daily_limit_reached'})
                
                if abs(z_score) >= self.config.trading.z_score_entry and confidence > 0.5:
                    if self.contrarian_mode:
                        signal_type = SignalType.LONG_BTC if z_score > 0 else SignalType.SHORT_BTC
                    else:
                        signal_type = SignalType.SHORT_BTC if z_score > 0 else SignalType.LONG_BTC
                    
                    return TradeSignal(signal_type, z_score, confidence, datetime.now(), metadata)
            
            return TradeSignal(SignalType.NO_SIGNAL, z_score, confidence, datetime.now(), metadata)
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return TradeSignal(SignalType.NO_SIGNAL, 0.0, 0.0, datetime.now(), 
                             {'error': str(e)})
    
    def _calculate_signal_confidence(self, z_score: float, correlation: Optional[float], 
                                   btc_vol: Optional[float], eth_vol: Optional[float]) -> float:
        confidence = 0.0
        
        confidence += min(abs(z_score) / 5.0, 1.0) * 0.4
        
        if correlation is not None:
            confidence += max(0, correlation - 0.5) * 0.3
        
        if btc_vol is not None and eth_vol is not None:
            vol_ratio = min(btc_vol, eth_vol) / max(btc_vol, eth_vol)
            confidence += vol_ratio * 0.2
        
        if self.data_service:
            data_quality = self.data_service.get_data_quality_report()
            confidence += (data_quality['success_rate'] / 100.0) * 0.1
        
        return min(confidence, 1.0)
    
    def check_holding_time_limits(self) -> Tuple[bool, str]:
        if not self.is_trading or self.entry_time is None:
            return False, ""
        
        holding_minutes = (datetime.now() - self.entry_time).total_seconds() / 60
        
        if holding_minutes < self.min_holding_minutes:
            return False, ""
        
        if holding_minutes > self.max_holding_minutes:
            return True, f"Maximum holding time reached: {holding_minutes:.1f} minutes"
        
        return False, ""
    
    def check_stop_loss(self) -> Tuple[bool, str]:
        if not self.is_trading or not self.positions:
            return False, ""
        
        try:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_position_value = sum(abs(pos.size * pos.entry_price) for pos in self.positions.values())
            
            if total_position_value == 0:
                return False, ""
            
            pnl_percentage = total_unrealized_pnl / total_position_value
            
            if pnl_percentage < -self.config.trading.stop_loss_pct:
                return True, f"Stop loss triggered: {pnl_percentage*100:.2f}% < -{self.config.trading.stop_loss_pct*100:.1f}%"
        
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
        
        return False, ""
    
    async def place_order(self, symbol: str, side: OrderSide, size: float, 
                         order_type: OrderType = OrderType.MARKET, 
                         price: Optional[float] = None) -> Dict[str, Any]:
        try:
            self.perf_monitor.start_timing(f"place_order_{symbol}")
            
            if order_type == OrderType.MARKET:
                result = await self.hl.market_order(symbol, side.value, size)
            else:
                if price is None:
                    current_price = await self.data_service.get_market_price(symbol)
                    if not current_price:
                        raise ValueError(f"Could not get current price for {symbol}")
                    price = current_price
                
                result = await self.hl.limit_order(symbol, side.value, size, price)
            
            self.perf_monitor.end_timing(f"place_order_{symbol}", 
                                       symbol=symbol, side=side.value, size=size)
            
            if result and 'order_id' in result:
                self.trade_logger.log_trade_execution(
                    symbol=symbol,
                    side=side.value,
                    size=size,
                    price=price or 0.0,
                    order_id=result['order_id']
                )
                
                self.daily_trades_count += 1
                self._trading_stats['total_trades'] += 1
            
            return result
            
        except Exception as e:
            self.perf_monitor.end_timing(f"place_order_{symbol}", error=str(e))
            self.logger.error(f"Error placing {order_type.value} {side.value} order for {symbol}: {e}")
            return {'error': str(e)}
    
    async def execute_signal(self, signal: TradeSignal) -> bool:
        if signal.signal_type == SignalType.NO_SIGNAL:
            return False
        
        try:
            self.trade_logger.log_trade_signal(
                symbol_pair="BTC/ETH",
                signal_type=signal.signal_type.value,
                z_score=signal.z_score,
                price_data={
                    'btc_price': self.data_service.price_history["BTC"][-1] if self.data_service else 0,
                    'eth_price': self.data_service.price_history["ETH"][-1] if self.data_service else 0
                },
                confidence=signal.confidence,
                metadata=signal.metadata
            )
            
            if signal.signal_type == SignalType.EXIT:
                return await self._execute_exit()
            else:
                return await self._execute_entry(signal)
                
        except Exception as e:
            self.logger.error(f"Error executing signal {signal.signal_type}: {e}")
            return False
    
    async def _execute_entry(self, signal: TradeSignal) -> bool:
        if self.is_trading:
            self.logger.warning("Already in a trade, cannot enter new position")
            return False
        
        try:
            self.update_beta()
            btc_size, eth_size = self.get_position_sizes(self.config.trading.position_size)
            
            btc_price = await self.data_service.get_market_price("BTC") if self.data_service else None
            eth_price = await self.data_service.get_market_price("ETH") if self.data_service else None
            
            if not btc_price or not eth_price:
                self.logger.error("Could not get current prices for entry")
                return False
            
            orders = []
            
            if signal.signal_type == SignalType.LONG_BTC:
                orders.append(("BTC", OrderSide.BUY, btc_size))
                orders.append(("ETH", OrderSide.SELL, eth_size))
            else:  # SHORT_BTC
                orders.append(("BTC", OrderSide.SELL, btc_size))
                orders.append(("ETH", OrderSide.BUY, eth_size))
            
            executed_orders = []
            for symbol, side, size in orders:
                result = await self.place_order(symbol, side, size)
                if 'error' not in result:
                    executed_orders.append((symbol, side, size, result))
                else:
                    self.logger.error(f"Failed to execute {side.value} order for {symbol}: {result['error']}")
                    await self._rollback_orders(executed_orders)
                    return False
            
            self.positions["BTC"] = Position(
                symbol="BTC",
                size=btc_size if signal.signal_type == SignalType.LONG_BTC else -btc_size,
                entry_price=btc_price,
                entry_time=datetime.now(),
                side="long" if signal.signal_type == SignalType.LONG_BTC else "short"
            )
            
            self.positions["ETH"] = Position(
                symbol="ETH",
                size=-eth_size if signal.signal_type == SignalType.LONG_BTC else eth_size,
                entry_price=eth_price,
                entry_time=datetime.now(),
                side="short" if signal.signal_type == SignalType.LONG_BTC else "long"
            )
            
            self.is_trading = True
            self.entry_time = datetime.now()
            self.entry_prices = {"BTC": btc_price, "ETH": eth_price}
            
            self.logger.info(f"Successfully entered {signal.signal_type.value} position")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing entry: {e}")
            return False
    
    async def _execute_exit(self) -> bool:
        if not self.is_trading or not self.positions:
            return False
        
        try:
            exit_orders = []
            for symbol, position in self.positions.items():
                if position.size > 0:
                    side = OrderSide.SELL
                else:
                    side = OrderSide.BUY
                
                result = await self.place_order(symbol, side, abs(position.size))
                if 'error' not in result:
                    exit_orders.append((symbol, result))
                else:
                    self.logger.error(f"Failed to exit position for {symbol}: {result['error']}")
            
            await self._calculate_realized_pnl()
            
            self.positions.clear()
            self.is_trading = False
            self.entry_time = None
            self.entry_prices.clear()
            
            self.logger.info("Successfully exited all positions")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing exit: {e}")
            return False
    
    async def _rollback_orders(self, executed_orders: List[Tuple]) -> None:
        for symbol, side, size, result in executed_orders:
            try:
                opposite_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
                await self.place_order(symbol, opposite_side, size)
                self.logger.info(f"Rolled back order for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to rollback order for {symbol}: {e}")
    
    async def _calculate_realized_pnl(self) -> None:
        if not self.data_service:
            return
        
        try:
            total_pnl = 0.0
            
            for symbol, position in self.positions.items():
                current_price = await self.data_service.get_market_price(symbol)
                if current_price:
                    if position.size > 0:  # Long position
                        pnl = (current_price - position.entry_price) * position.size
                    else:  # Short position
                        pnl = (position.entry_price - current_price) * abs(position.size)
                    
                    position.realized_pnl = pnl
                    total_pnl += pnl
            
            self._trading_stats['total_pnl'] += total_pnl
            self._trading_stats['daily_pnl'] += total_pnl
            
            if total_pnl > 0:
                self._trading_stats['winning_trades'] += 1
            else:
                self._trading_stats['losing_trades'] += 1
            
            self.trade_logger.log_portfolio_update(
                total_pnl=self._trading_stats['total_pnl'],
                daily_pnl=self._trading_stats['daily_pnl'],
                positions={symbol: {
                    'size': pos.size,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                } for symbol, pos in self.positions.items()}
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating realized PnL: {e}")
    
    async def update_unrealized_pnl(self) -> None:
        if not self.is_trading or not self.data_service:
            return
        
        try:
            for symbol, position in self.positions.items():
                current_price = await self.data_service.get_market_price(symbol)
                if current_price:
                    if position.size > 0:  # Long position
                        unrealized_pnl = (current_price - position.entry_price) * position.size
                    else:  # Short position
                        unrealized_pnl = (position.entry_price - current_price) * abs(position.size)
                    
                    position.unrealized_pnl = unrealized_pnl
                    
        except Exception as e:
            self.logger.error(f"Error updating unrealized PnL: {e}")
    
    def get_trading_stats(self) -> Dict[str, Any]:
        total_trades = self._trading_stats['winning_trades'] + self._trading_stats['losing_trades']
        win_rate = (self._trading_stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
        
        return {
            **self._trading_stats,
            'win_rate': win_rate,
            'daily_trades_remaining': self.daily_trade_limit - self.daily_trades_count,
            'is_trading': self.is_trading,
            'current_positions': {
                symbol: {
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'holding_time_minutes': (datetime.now() - pos.entry_time).total_seconds() / 60
                } for symbol, pos in self.positions.items()
            }
        }
    
    async def force_exit_all_positions(self, reason: str = "Manual exit") -> bool:
        if not self.is_trading:
            return True
        
        self.logger.warning(f"Force exiting all positions: {reason}")
        result = await self._execute_exit()
        
        if result:
            self.trade_logger.log_risk_event(
                event_type="force_exit",
                symbol="ALL",
                current_value=0,
                threshold=0,
                action="exit_all_positions",
                reason=reason
            )
        
        return result