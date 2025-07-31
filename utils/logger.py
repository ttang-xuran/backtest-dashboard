#!/usr/bin/env python3
"""
Centralized Logging System with Monitoring Support
"""

import logging
import logging.handlers
import sys
import os
from typing import Optional, Dict, Any
from datetime import datetime
import json
from pathlib import Path

class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

class TradingLogger:
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.setup_logging()
    
    def setup_logging(self, 
                     log_level: str = "INFO",
                     log_file: str = "stat_arb_bot.log",
                     max_file_size: int = 10485760,
                     backup_count: int = 5,
                     structured: bool = False) -> None:
        
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file_path = log_dir / log_file
        
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
        
        if structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        
        error_file_path = log_dir / f"error_{log_file}"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
        
        logging.getLogger("hyperliquid").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]

class TradeLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_trade_signal(self, symbol_pair: str, signal_type: str, z_score: float, 
                        price_data: Dict[str, float], **kwargs) -> None:
        extra_data = {
            'event_type': 'trade_signal',
            'symbol_pair': symbol_pair,
            'signal_type': signal_type,
            'z_score': z_score,
            'price_data': price_data,
            **kwargs
        }
        
        self.logger.info(
            f"Trade Signal: {signal_type} for {symbol_pair} (Z-Score: {z_score:.3f})",
            extra={'extra_data': extra_data}
        )
    
    def log_trade_execution(self, symbol: str, side: str, size: float, 
                           price: float, order_id: str, **kwargs) -> None:
        extra_data = {
            'event_type': 'trade_execution',
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'order_id': order_id,
            **kwargs
        }
        
        self.logger.info(
            f"Trade Executed: {side} {size} {symbol} @ {price} (Order: {order_id})",
            extra={'extra_data': extra_data}
        )
    
    def log_risk_event(self, event_type: str, symbol: str, current_value: float, 
                      threshold: float, action: str, **kwargs) -> None:
        extra_data = {
            'event_type': 'risk_event',
            'risk_type': event_type,
            'symbol': symbol,
            'current_value': current_value,
            'threshold': threshold,
            'action': action,
            **kwargs
        }
        
        self.logger.warning(
            f"Risk Event: {event_type} for {symbol} - {current_value} vs {threshold} -> {action}",
            extra={'extra_data': extra_data}
        )
    
    def log_portfolio_update(self, total_pnl: float, positions: Dict[str, Any], 
                           daily_pnl: float, **kwargs) -> None:
        extra_data = {
            'event_type': 'portfolio_update',
            'total_pnl': total_pnl,
            'daily_pnl': daily_pnl,
            'positions': positions,
            **kwargs
        }
        
        self.logger.info(
            f"Portfolio Update: Total PnL: {total_pnl:.4f}, Daily PnL: {daily_pnl:.4f}",
            extra={'extra_data': extra_data}
        )
    
    def log_market_data_update(self, symbol: str, price: float, volume: float, 
                              timestamp: datetime, **kwargs) -> None:
        extra_data = {
            'event_type': 'market_data',
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'timestamp': timestamp.isoformat(),
            **kwargs
        }
        
        self.logger.debug(
            f"Market Data: {symbol} @ {price} (Vol: {volume})",
            extra={'extra_data': extra_data}
        )

class PerformanceMonitor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times: Dict[str, float] = {}
    
    def start_timing(self, operation: str) -> None:
        import time
        self.start_times[operation] = time.time()
    
    def end_timing(self, operation: str, **kwargs) -> float:
        import time
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        extra_data = {
            'event_type': 'performance',
            'operation': operation,
            'duration': duration,
            **kwargs
        }
        
        level = logging.WARNING if duration > 5.0 else logging.INFO
        self.logger.log(
            level,
            f"Performance: {operation} took {duration:.3f}s",
            extra={'extra_data': extra_data}
        )
        
        return duration
    
    def log_memory_usage(self) -> None:
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            extra_data = {
                'event_type': 'memory_usage',
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
            }
            
            self.logger.info(
                f"Memory Usage: RSS: {extra_data['rss_mb']:.1f}MB, CPU: {extra_data['cpu_percent']:.1f}%",
                extra={'extra_data': extra_data}
            )
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")

def get_logger(name: str) -> logging.Logger:
    trading_logger = TradingLogger()
    return trading_logger.get_logger(name)

def get_trade_logger(name: str) -> TradeLogger:
    logger = get_logger(name)
    return TradeLogger(logger)

def get_performance_monitor(name: str) -> PerformanceMonitor:
    logger = get_logger(name)
    return PerformanceMonitor(logger)