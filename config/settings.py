#!/usr/bin/env python3
"""
Centralized Configuration Management System
Handles all configuration loading with environment variable support
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TradingConfig:
    position_size: float = 0.20
    z_score_entry: float = 3.0
    z_score_exit: float = 0.8
    lookback_period: int = 30
    max_positions: int = 2
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10

@dataclass
class RiskConfig:
    max_daily_loss: float = 0.02
    max_position_size: float = 0.25
    max_leverage: float = 1.0
    drawdown_threshold: float = 0.15
    correlation_threshold: float = 0.8

@dataclass
class APIConfig:
    hyperliquid_url: str = "https://api.hyperliquid.xyz"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_per_second: int = 10

@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "stat_arb_bot.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5

@dataclass
class Configuration:
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)  
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    credentials_path: str = ""
    environment: str = "development"

class ConfigManager:
    def __init__(self, config_file: Optional[str] = None):
        self.config = Configuration()
        self.logger = self._setup_logger()
        
        if config_file:
            self.load_from_file(config_file)
        
        self._load_from_environment()
        self._load_credentials_path()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_from_file(self, config_file: str) -> None:
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                self.logger.warning(f"Config file {config_file} not found, using defaults")
                return
            
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            self._update_config_from_dict(data)
            self.logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading config file {config_file}: {e}")
            self.logger.info("Using default configuration")
    
    def _update_config_from_dict(self, data: Dict[str, Any]) -> None:
        if 'trading' in data:
            for key, value in data['trading'].items():
                if hasattr(self.config.trading, key):
                    setattr(self.config.trading, key, value)
        
        if 'risk' in data:
            for key, value in data['risk'].items():
                if hasattr(self.config.risk, key):
                    setattr(self.config.risk, key, value)
        
        if 'api' in data:
            for key, value in data['api'].items():
                if hasattr(self.config.api, key):
                    setattr(self.config.api, key, value)
        
        if 'logging' in data:
            for key, value in data['logging'].items():
                if hasattr(self.config.logging, key):
                    setattr(self.config.logging, key, value)
        
        for key in ['credentials_path', 'environment']:
            if key in data:
                setattr(self.config, key, data[key])
    
    def _load_from_environment(self) -> None:
        env_mappings = {
            'POSITION_SIZE': ('trading', 'position_size', float),
            'Z_SCORE_ENTRY': ('trading', 'z_score_entry', float),
            'Z_SCORE_EXIT': ('trading', 'z_score_exit', float),
            'LOOKBACK_PERIOD': ('trading', 'lookback_period', int),
            'MAX_DAILY_LOSS': ('risk', 'max_daily_loss', float),
            'MAX_POSITION_SIZE': ('risk', 'max_position_size', float),
            'LOG_LEVEL': ('logging', 'level', str),
            'ENVIRONMENT': ('', 'environment', str),
        }
        
        for env_var, (section, attr, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = type_func(value)
                    if section:
                        setattr(getattr(self.config, section), attr, converted_value)
                    else:
                        setattr(self.config, attr, converted_value)
                    self.logger.debug(f"Set {attr} from environment: {converted_value}")
                except ValueError as e:
                    self.logger.warning(f"Invalid value for {env_var}: {value}, error: {e}")
    
    def _load_credentials_path(self) -> None:
        credentials_path = os.getenv('HYPERLIQUID_CREDENTIALS_PATH')
        if credentials_path:
            self.config.credentials_path = credentials_path
        elif not self.config.credentials_path:
            default_path = "/mnt/c/Users/16473/Desktop/Trading/hyperliquid/trade_api.json"
            if Path(default_path).exists():
                self.config.credentials_path = default_path
                self.logger.info(f"Using default credentials path: {default_path}")
            else:
                self.logger.warning("No credentials path found. Set HYPERLIQUID_CREDENTIALS_PATH environment variable")
    
    def get_credentials(self) -> Dict[str, Any]:
        if not self.config.credentials_path:
            raise ValueError("No credentials path configured")
        
        try:
            with open(self.config.credentials_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Credentials file not found: {self.config.credentials_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in credentials file: {e}")
    
    def validate_config(self) -> bool:
        try:
            assert 0 < self.config.trading.position_size <= 1.0, "Position size must be between 0 and 1"
            assert self.config.trading.z_score_entry > 0, "Z-score entry must be positive"
            assert self.config.trading.z_score_exit >= 0, "Z-score exit must be non-negative"
            assert self.config.trading.lookback_period > 0, "Lookback period must be positive"
            assert 0 < self.config.risk.max_daily_loss <= 1.0, "Max daily loss must be between 0 and 1"
            assert 0 < self.config.risk.max_position_size <= 1.0, "Max position size must be between 0 and 1"
            
            self.logger.info("Configuration validation successful")
            return True
            
        except AssertionError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trading': {
                'position_size': self.config.trading.position_size,
                'z_score_entry': self.config.trading.z_score_entry,
                'z_score_exit': self.config.trading.z_score_exit,
                'lookback_period': self.config.trading.lookback_period,
                'max_positions': self.config.trading.max_positions,
                'stop_loss_pct': self.config.trading.stop_loss_pct,
                'take_profit_pct': self.config.trading.take_profit_pct,
            },
            'risk': {
                'max_daily_loss': self.config.risk.max_daily_loss,
                'max_position_size': self.config.risk.max_position_size,
                'max_leverage': self.config.risk.max_leverage,
                'drawdown_threshold': self.config.risk.drawdown_threshold,
                'correlation_threshold': self.config.risk.correlation_threshold,
            },
            'api': {
                'hyperliquid_url': self.config.api.hyperliquid_url,
                'timeout': self.config.api.timeout,
                'max_retries': self.config.api.max_retries,
                'retry_delay': self.config.api.retry_delay,
                'rate_limit_per_second': self.config.api.rate_limit_per_second,
            },
            'logging': {
                'level': self.config.logging.level,
                'format': self.config.logging.format,
                'file_path': self.config.logging.file_path,
                'max_file_size': self.config.logging.max_file_size,
                'backup_count': self.config.logging.backup_count,
            },
            'credentials_path': self.config.credentials_path,
            'environment': self.config.environment,
        }

config_manager = ConfigManager()