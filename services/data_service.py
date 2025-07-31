#!/usr/bin/env python3
"""
Market Data Service
Handles all market data collection, processing, and storage
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import time

from utils.logger import get_logger, get_performance_monitor
from config.settings import config_manager


class MarketDataError(Exception):
    pass


class DataService:
    def __init__(self, hyperliquid_client=None):
        self.hl = hyperliquid_client
        self.logger = get_logger(__name__)
        self.perf_monitor = get_performance_monitor(__name__)
        self.config = config_manager.config
        
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.trading.lookback_period * 2))
        self.returns_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.trading.lookback_period))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.trading.lookback_period))
        
        self.last_update_time: Dict[str, datetime] = {}
        self.update_interval = 60
        
        self._data_quality_stats = {
            'successful_updates': 0,
            'failed_updates': 0,
            'last_successful_update': None,
            'consecutive_failures': 0
        }
    
    async def get_market_price(self, symbol: str, retry_count: int = 0) -> Optional[float]:
        try:
            self.perf_monitor.start_timing(f"get_market_price_{symbol}")
            
            tickers = await self.hl.fetch_tickers()
            market_symbol = f"{symbol}/USD:USD"
            
            if market_symbol in tickers:
                price = float(tickers[market_symbol]['close'])
                self.perf_monitor.end_timing(f"get_market_price_{symbol}", price=price)
                return price
            
            markets = await self.hl.fetch_markets()
            for market in markets:
                if market['base'] == symbol:
                    price = float(market['info']['markPx'])
                    self.perf_monitor.end_timing(f"get_market_price_{symbol}", price=price)
                    return price
            
            self.logger.warning(f"Symbol {symbol} not found in market data")
            return None
            
        except Exception as e:
            self.perf_monitor.end_timing(f"get_market_price_{symbol}", error=str(e))
            
            if retry_count < self.config.api.max_retries:
                self.logger.warning(f"Retrying get_market_price for {symbol} (attempt {retry_count + 1})")
                await asyncio.sleep(self.config.api.retry_delay * (retry_count + 1))
                return await self.get_market_price(symbol, retry_count + 1)
            
            self.logger.error(f"Failed to get price for {symbol} after {retry_count + 1} attempts: {e}")
            self._data_quality_stats['failed_updates'] += 1
            self._data_quality_stats['consecutive_failures'] += 1
            return None
    
    async def get_market_data_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        try:
            self.perf_monitor.start_timing("get_market_data_batch")
            
            tickers = await self.hl.fetch_tickers()
            markets = await self.hl.fetch_markets()
            
            market_data = {}
            
            for symbol in symbols:
                market_symbol = f"{symbol}/USD:USD"
                data = {}
                
                if market_symbol in tickers:
                    ticker = tickers[market_symbol]
                    data.update({
                        'price': float(ticker['close']),
                        'bid': float(ticker['bid']) if ticker['bid'] else None,
                        'ask': float(ticker['ask']) if ticker['ask'] else None,
                        'volume': float(ticker['baseVolume']) if ticker['baseVolume'] else None,
                        'timestamp': datetime.now(),
                        'spread': (float(ticker['ask']) - float(ticker['bid'])) if ticker['ask'] and ticker['bid'] else None
                    })
                
                for market in markets:
                    if market['base'] == symbol:
                        data.update({
                            'mark_price': float(market['info']['markPx']),
                            'funding_rate': float(market['info'].get('funding', 0)),
                            'open_interest': float(market['info'].get('openInterest', 0))
                        })
                        break
                
                if data:
                    market_data[symbol] = data
                else:
                    self.logger.warning(f"No data found for symbol {symbol}")
            
            self.perf_monitor.end_timing("get_market_data_batch", symbols_count=len(symbols))
            self._data_quality_stats['successful_updates'] += 1
            self._data_quality_stats['consecutive_failures'] = 0
            self._data_quality_stats['last_successful_update'] = datetime.now()
            
            return market_data
            
        except Exception as e:
            self.perf_monitor.end_timing("get_market_data_batch", error=str(e))
            self.logger.error(f"Failed to get batch market data: {e}")
            self._data_quality_stats['failed_updates'] += 1
            self._data_quality_stats['consecutive_failures'] += 1
            return {}
    
    async def update_price_history(self, symbols: List[str]) -> Dict[str, bool]:
        results = {}
        
        try:
            market_data = await self.get_market_data_batch(symbols)
            
            for symbol in symbols:
                if symbol not in market_data:
                    results[symbol] = False
                    continue
                
                data = market_data[symbol]
                current_price = data.get('price') or data.get('mark_price')
                current_volume = data.get('volume', 0)
                
                if current_price is None:
                    results[symbol] = False
                    continue
                
                if len(self.price_history[symbol]) > 0:
                    previous_price = self.price_history[symbol][-1]
                    if previous_price > 0:
                        return_rate = (current_price - previous_price) / previous_price
                        self.returns_history[symbol].append(return_rate)
                
                self.price_history[symbol].append(current_price)
                self.volume_history[symbol].append(current_volume)
                self.last_update_time[symbol] = datetime.now()
                
                results[symbol] = True
                
                self.logger.debug(f"Updated {symbol}: Price={current_price:.4f}, Volume={current_volume:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error updating price history: {e}")
            return {symbol: False for symbol in symbols}
    
    def get_returns_series(self, symbol: str, lookback: Optional[int] = None) -> np.ndarray:
        lookback = lookback or self.config.trading.lookback_period
        
        if symbol not in self.returns_history or len(self.returns_history[symbol]) < lookback:
            return np.array([])
        
        return np.array(list(self.returns_history[symbol])[-lookback:])
    
    def get_price_series(self, symbol: str, lookback: Optional[int] = None) -> np.ndarray:
        lookback = lookback or self.config.trading.lookback_period
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < lookback:
            return np.array([])
        
        return np.array(list(self.price_history[symbol])[-lookback:])
    
    def get_volume_series(self, symbol: str, lookback: Optional[int] = None) -> np.ndarray:
        lookback = lookback or self.config.trading.lookback_period
        
        if symbol not in self.volume_history or len(self.volume_history[symbol]) < lookback:
            return np.array([])
        
        return np.array(list(self.volume_history[symbol])[-lookback:])
    
    def calculate_spread_ratio(self, symbol1: str, symbol2: str) -> Optional[float]:
        if (len(self.price_history[symbol1]) == 0 or 
            len(self.price_history[symbol2]) == 0):
            return None
        
        price1 = self.price_history[symbol1][-1]
        price2 = self.price_history[symbol2][-1]
        
        if price2 == 0:
            return None
        
        return price1 / price2
    
    def calculate_z_score(self, symbol1: str, symbol2: str, lookback: Optional[int] = None) -> Optional[float]:
        lookback = lookback or self.config.trading.lookback_period
        
        returns1 = self.get_returns_series(symbol1, lookback)
        returns2 = self.get_returns_series(symbol2, lookback)
        
        if len(returns1) < lookback or len(returns2) < lookback:
            self.logger.debug(f"Insufficient data for Z-score calculation: {symbol1}={len(returns1)}, {symbol2}={len(returns2)}")
            return None
        
        spread = returns1 - returns2
        
        if len(spread) < 2:
            return None
        
        mean_spread = np.mean(spread)
        std_spread = np.std(spread, ddof=1)
        
        if std_spread == 0:
            return None
        
        current_spread = spread[-1]
        z_score = (current_spread - mean_spread) / std_spread
        
        return z_score
    
    def get_correlation(self, symbol1: str, symbol2: str, lookback: Optional[int] = None) -> Optional[float]:
        lookback = lookback or self.config.trading.lookback_period
        
        returns1 = self.get_returns_series(symbol1, lookback)
        returns2 = self.get_returns_series(symbol2, lookback)
        
        if len(returns1) < lookback or len(returns2) < lookback:
            return None
        
        correlation_matrix = np.corrcoef(returns1, returns2)
        return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else None
    
    def get_volatility(self, symbol: str, lookback: Optional[int] = None) -> Optional[float]:
        lookback = lookback or self.config.trading.lookback_period
        
        returns = self.get_returns_series(symbol, lookback)
        
        if len(returns) < lookback:
            return None
        
        return np.std(returns, ddof=1) * np.sqrt(252)
    
    def is_data_stale(self, symbol: str, max_age_minutes: int = 5) -> bool:
        if symbol not in self.last_update_time:
            return True
        
        age = datetime.now() - self.last_update_time[symbol]
        return age.total_seconds() > (max_age_minutes * 60)
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        total_updates = self._data_quality_stats['successful_updates'] + self._data_quality_stats['failed_updates']
        success_rate = (self._data_quality_stats['successful_updates'] / total_updates * 100) if total_updates > 0 else 0
        
        return {
            'success_rate': success_rate,
            'total_updates': total_updates,
            'successful_updates': self._data_quality_stats['successful_updates'],
            'failed_updates': self._data_quality_stats['failed_updates'],
            'consecutive_failures': self._data_quality_stats['consecutive_failures'],
            'last_successful_update': self._data_quality_stats['last_successful_update'],
            'data_staleness': {
                symbol: self.is_data_stale(symbol) 
                for symbol in self.last_update_time.keys()
            }
        }
    
    def clear_history(self, symbol: Optional[str] = None):
        if symbol:
            self.price_history[symbol].clear()
            self.returns_history[symbol].clear()
            self.volume_history[symbol].clear()
            if symbol in self.last_update_time:
                del self.last_update_time[symbol]
        else:
            self.price_history.clear()
            self.returns_history.clear()
            self.volume_history.clear()
            self.last_update_time.clear()
        
        self.logger.info(f"Cleared history for {'all symbols' if not symbol else symbol}")
    
    async def health_check(self) -> Dict[str, Any]:
        try:
            test_symbols = ["BTC", "ETH"]
            market_data = await self.get_market_data_batch(test_symbols)
            
            health_status = {
                'status': 'healthy' if len(market_data) == len(test_symbols) else 'degraded',
                'timestamp': datetime.now().isoformat(),
                'api_connection': len(market_data) > 0,
                'data_quality': self.get_data_quality_report(),
                'symbols_available': list(market_data.keys()),
                'consecutive_failures': self._data_quality_stats['consecutive_failures']
            }
            
            if self._data_quality_stats['consecutive_failures'] > 5:
                health_status['status'] = 'unhealthy'
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'api_connection': False
            }