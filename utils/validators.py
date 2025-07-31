#!/usr/bin/env python3
"""
Data Validation Layers
Comprehensive validation system for trading data, API responses, and configurations
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json

from utils.logger import get_logger
from utils.error_handler import ValidationError, DataQualityError, error_tracker, ErrorSeverity, ErrorCategory


class ValidationLevel(Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Any = None
    metadata: Dict[str, Any] = None


class BaseValidator:
    def __init__(self, name: str, level: ValidationLevel = ValidationLevel.MODERATE):
        self.name = name
        self.level = level
        self.logger = get_logger(f"{__name__}.{name}")
    
    def validate(self, data: Any) -> ValidationResult:
        raise NotImplementedError("Subclasses must implement validate method")
    
    def _log_validation_result(self, result: ValidationResult, data_type: str = "unknown"):
        if not result.is_valid:
            self.logger.error(f"Validation failed for {self.name}: {result.errors}")
            error_tracker.record_error(
                f"validation_{self.name}",
                ValidationError(f"Validation failed: {result.errors}"),
                ErrorSeverity.MEDIUM,
                ErrorCategory.DATA,
                {'validator': self.name, 'data_type': data_type, 'errors': result.errors}
            )
        elif result.warnings:
            self.logger.warning(f"Validation warnings for {self.name}: {result.warnings}")


class NumericValidator(BaseValidator):
    def __init__(self, name: str = "numeric", 
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None,
                 allow_nan: bool = False,
                 allow_inf: bool = False,
                 level: ValidationLevel = ValidationLevel.MODERATE):
        super().__init__(name, level)
        self.min_value = min_value
        self.max_value = max_value
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf
    
    def validate(self, data: Union[float, int, np.ndarray, List]) -> ValidationResult:
        errors = []
        warnings = []
        cleaned_data = data
        
        try:
            if isinstance(data, (list, np.ndarray)):
                arr = np.array(data)
                
                # Check for NaN values
                if np.isnan(arr).any() and not self.allow_nan:
                    nan_count = np.isnan(arr).sum()
                    if self.level == ValidationLevel.STRICT:
                        errors.append(f"Contains {nan_count} NaN values")
                    else:
                        warnings.append(f"Contains {nan_count} NaN values")
                        cleaned_data = arr[~np.isnan(arr)] if self.level == ValidationLevel.LENIENT else arr
                
                # Check for infinite values
                if np.isinf(arr).any() and not self.allow_inf:
                    inf_count = np.isinf(arr).sum()
                    if self.level == ValidationLevel.STRICT:
                        errors.append(f"Contains {inf_count} infinite values")
                    else:
                        warnings.append(f"Contains {inf_count} infinite values")
                        if self.level == ValidationLevel.LENIENT:
                            cleaned_data = arr[~np.isinf(arr)]
                
                # Check range for array
                valid_data = arr[~np.isnan(arr) & ~np.isinf(arr)]
                if len(valid_data) > 0:
                    if self.min_value is not None and valid_data.min() < self.min_value:
                        errors.append(f"Minimum value {valid_data.min()} below threshold {self.min_value}")
                    
                    if self.max_value is not None and valid_data.max() > self.max_value:
                        errors.append(f"Maximum value {valid_data.max()} above threshold {self.max_value}")
            
            else:
                # Single value validation
                if pd.isna(data) and not self.allow_nan:
                    errors.append("Value is NaN")
                
                if np.isinf(data) and not self.allow_inf:
                    errors.append("Value is infinite")
                
                if not pd.isna(data) and not np.isinf(data):
                    if self.min_value is not None and data < self.min_value:
                        errors.append(f"Value {data} below minimum {self.min_value}")
                    
                    if self.max_value is not None and data > self.max_value:
                        errors.append(f"Value {data} above maximum {self.max_value}")
        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned_data,
            metadata={'validator_type': 'numeric', 'original_type': type(data).__name__}
        )
        
        self._log_validation_result(result, "numeric")
        return result


class PriceValidator(BaseValidator):
    def __init__(self, name: str = "price",
                 min_price: float = 0.0001,
                 max_price: float = 1000000,
                 max_price_change_pct: float = 0.5,  # 50% max change
                 level: ValidationLevel = ValidationLevel.MODERATE):
        super().__init__(name, level)
        self.min_price = min_price
        self.max_price = max_price
        self.max_price_change_pct = max_price_change_pct
        self.last_price: Optional[float] = None
    
    def validate(self, price: float, symbol: str = "unknown") -> ValidationResult:
        errors = []
        warnings = []
        
        # Basic numeric validation
        if pd.isna(price) or price <= 0:
            errors.append(f"Invalid price: {price}")
            return ValidationResult(False, errors, warnings)
        
        # Range validation
        if price < self.min_price:
            errors.append(f"Price {price} below minimum {self.min_price}")
        
        if price > self.max_price:
            errors.append(f"Price {price} above maximum {self.max_price}")
        
        # Price change validation
        if self.last_price is not None:
            price_change_pct = abs(price - self.last_price) / self.last_price
            
            if price_change_pct > self.max_price_change_pct:
                if self.level == ValidationLevel.STRICT:
                    errors.append(f"Price change {price_change_pct:.2%} exceeds limit {self.max_price_change_pct:.2%}")
                else:
                    warnings.append(f"Large price change detected: {price_change_pct:.2%}")
        
        self.last_price = price
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=price,
            metadata={
                'symbol': symbol,
                'price_change_pct': abs(price - self.last_price) / self.last_price if self.last_price else 0
            }
        )
        
        self._log_validation_result(result, f"price_{symbol}")
        return result


class MarketDataValidator(BaseValidator):
    def __init__(self, name: str = "market_data", level: ValidationLevel = ValidationLevel.MODERATE):
        super().__init__(name, level)
        self.price_validator = PriceValidator(level=level)
        self.required_fields = ['price', 'timestamp']
        
    def validate(self, market_data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []
        cleaned_data = market_data.copy()
        
        # Check required fields
        for field in self.required_fields:
            if field not in market_data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(False, errors, warnings)
        
        # Validate price
        symbol = market_data.get('symbol', 'unknown')
        price_result = self.price_validator.validate(market_data['price'], symbol)
        
        if not price_result.is_valid:
            errors.extend([f"Price validation: {e}" for e in price_result.errors])
        warnings.extend([f"Price validation: {w}" for w in price_result.warnings])
        
        # Validate timestamp
        timestamp = market_data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    cleaned_data['timestamp'] = timestamp
                except ValueError:
                    errors.append(f"Invalid timestamp format: {timestamp}")
            
            elif isinstance(timestamp, datetime):
                # Check if timestamp is reasonable (not too old or in future)
                now = datetime.now()
                if timestamp > now + timedelta(minutes=5):
                    warnings.append("Timestamp is in the future")
                elif timestamp < now - timedelta(hours=24):
                    warnings.append("Timestamp is more than 24 hours old")
        
        # Validate optional fields
        if 'volume' in market_data:
            volume = market_data['volume']
            if volume < 0:
                errors.append(f"Invalid volume: {volume}")
            elif volume == 0:
                warnings.append("Zero volume detected")
        
        if 'bid' in market_data and 'ask' in market_data:
            bid, ask = market_data['bid'], market_data['ask']
            if bid >= ask:
                errors.append(f"Invalid bid/ask spread: bid={bid}, ask={ask}")
            
            spread_pct = (ask - bid) / bid if bid > 0 else float('inf')
            if spread_pct > 0.1:  # 10% spread seems excessive
                warnings.append(f"Wide bid/ask spread: {spread_pct:.2%}")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned_data,
            metadata={'symbol': symbol, 'data_fields': list(market_data.keys())}
        )
        
        self._log_validation_result(result, f"market_data_{symbol}")
        return result


class TradingSignalValidator(BaseValidator):
    def __init__(self, name: str = "trading_signal", level: ValidationLevel = ValidationLevel.MODERATE):
        super().__init__(name, level)
        self.valid_signal_types = ['long_btc', 'short_btc', 'exit', 'no_signal']
        
    def validate(self, signal_data: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []
        
        # Check signal type
        signal_type = signal_data.get('signal_type')
        if not signal_type:
            errors.append("Missing signal_type")
        elif signal_type not in self.valid_signal_types:
            errors.append(f"Invalid signal_type: {signal_type}")
        
        # Validate Z-score
        z_score = signal_data.get('z_score')
        if z_score is not None:
            if pd.isna(z_score) or abs(z_score) > 10:  # Extreme Z-score
                if abs(z_score) > 10:
                    warnings.append(f"Extreme Z-score detected: {z_score}")
                if pd.isna(z_score):
                    errors.append("Z-score is NaN")
        
        # Validate confidence
        confidence = signal_data.get('confidence')
        if confidence is not None:
            if not (0 <= confidence <= 1):
                errors.append(f"Confidence must be between 0 and 1: {confidence}")
            elif confidence < 0.3:
                warnings.append(f"Low confidence signal: {confidence}")
        
        # Validate timestamp
        timestamp = signal_data.get('timestamp')
        if timestamp and isinstance(timestamp, datetime):
            age_seconds = (datetime.now() - timestamp).total_seconds()
            if age_seconds > 300:  # 5 minutes old
                warnings.append(f"Signal is {age_seconds:.0f} seconds old")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=signal_data,
            metadata={'signal_type': signal_type, 'z_score': z_score, 'confidence': confidence}
        )
        
        self._log_validation_result(result, "trading_signal")
        return result


class ConfigValidator(BaseValidator):
    def __init__(self, name: str = "config", level: ValidationLevel = ValidationLevel.STRICT):
        super().__init__(name, level)
    
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []
        
        # Validate trading parameters
        trading_config = config.get('trading', {})
        
        position_size = trading_config.get('position_size')
        if position_size is not None:
            if not (0 < position_size <= 1.0):
                errors.append(f"position_size must be between 0 and 1: {position_size}")
        
        z_score_entry = trading_config.get('z_score_entry')
        if z_score_entry is not None:
            if z_score_entry <= 0:
                errors.append(f"z_score_entry must be positive: {z_score_entry}")
            elif z_score_entry < 1.5:
                warnings.append(f"Low z_score_entry may cause overtrading: {z_score_entry}")
        
        z_score_exit = trading_config.get('z_score_exit')
        if z_score_exit is not None and z_score_entry is not None:
            if z_score_exit >= z_score_entry:
                errors.append(f"z_score_exit ({z_score_exit}) must be less than z_score_entry ({z_score_entry})")
        
        # Validate risk parameters
        risk_config = config.get('risk', {})
        
        max_daily_loss = risk_config.get('max_daily_loss')
        if max_daily_loss is not None:
            if not (0 < max_daily_loss <= 1.0):
                errors.append(f"max_daily_loss must be between 0 and 1: {max_daily_loss}")
        
        drawdown_threshold = risk_config.get('drawdown_threshold')
        if drawdown_threshold is not None:
            if not (0 < drawdown_threshold <= 1.0):
                errors.append(f"drawdown_threshold must be between 0 and 1: {drawdown_threshold}")
        
        # Validate API configuration
        api_config = config.get('api', {})
        
        timeout = api_config.get('timeout')
        if timeout is not None:
            if timeout <= 0 or timeout > 300:  # 5 minutes max
                errors.append(f"API timeout must be between 0 and 300 seconds: {timeout}")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=config,
            metadata={'config_sections': list(config.keys())}
        )
        
        self._log_validation_result(result, "configuration")
        return result


class DataQualityChecker:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.quality_thresholds = {
            'completeness': 0.95,  # 95% of expected data
            'consistency': 0.98,   # 98% consistent values
            'timeliness': 300,     # 5 minutes max age
            'accuracy': 0.95       # 95% accurate values
        }
    
    def check_completeness(self, data: Dict[str, Any], expected_fields: List[str]) -> float:
        """Check what percentage of expected fields are present and non-null"""
        present_fields = sum(1 for field in expected_fields if field in data and data[field] is not None)
        return present_fields / len(expected_fields) if expected_fields else 1.0
    
    def check_consistency(self, current_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> float:
        """Check consistency with historical patterns"""
        if not historical_data:
            return 1.0
        
        consistency_score = 1.0
        
        # Check price consistency
        if 'price' in current_data:
            current_price = current_data['price']
            recent_prices = [d.get('price') for d in historical_data[-10:] if d.get('price')]
            
            if recent_prices:
                avg_price = np.mean(recent_prices)
                price_deviation = abs(current_price - avg_price) / avg_price
                
                # Penalize large deviations
                if price_deviation > 0.1:  # 10% deviation
                    consistency_score *= (1 - min(price_deviation, 0.5))
        
        return consistency_score
    
    def check_timeliness(self, data: Dict[str, Any]) -> float:
        """Check if data is recent enough"""
        timestamp = data.get('timestamp')
        if not timestamp:
            return 0.5  # No timestamp is moderately bad
        
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                return 0.0
        
        age_seconds = (datetime.now() - timestamp).total_seconds()
        
        if age_seconds <= self.quality_thresholds['timeliness']:
            return 1.0
        else:
            # Exponential decay based on age
            return max(0.0, np.exp(-age_seconds / self.quality_thresholds['timeliness']))
    
    def check_accuracy(self, data: Dict[str, Any]) -> float:
        """Check data accuracy based on various heuristics"""
        accuracy_score = 1.0
        
        # Check for reasonable price values
        if 'price' in data:
            price = data['price']
            if price <= 0 or price > 1000000:  # Unreasonable price
                accuracy_score *= 0.5
        
        # Check for reasonable volume
        if 'volume' in data:
            volume = data['volume']
            if volume < 0:  # Negative volume is impossible
                accuracy_score *= 0.3
        
        # Check bid/ask spread
        if 'bid' in data and 'ask' in data:
            bid, ask = data['bid'], data['ask']
            if bid >= ask:  # Invalid spread
                accuracy_score *= 0.2
            else:
                spread_pct = (ask - bid) / bid
                if spread_pct > 0.2:  # 20% spread seems inaccurate
                    accuracy_score *= 0.7
        
        return accuracy_score
    
    def assess_data_quality(self, data: Dict[str, Any], 
                          expected_fields: List[str] = None,
                          historical_data: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """Comprehensive data quality assessment"""
        expected_fields = expected_fields or ['price', 'timestamp']
        historical_data = historical_data or []
        
        quality_scores = {
            'completeness': self.check_completeness(data, expected_fields),
            'consistency': self.check_consistency(data, historical_data),
            'timeliness': self.check_timeliness(data),
            'accuracy': self.check_accuracy(data)
        }
        
        # Overall quality score (weighted average)
        weights = {'completeness': 0.3, 'consistency': 0.2, 'timeliness': 0.3, 'accuracy': 0.2}
        overall_score = sum(quality_scores[metric] * weights[metric] for metric in quality_scores)
        quality_scores['overall'] = overall_score
        
        # Log quality issues
        for metric, score in quality_scores.items():
            if metric != 'overall' and score < self.quality_thresholds.get(metric, 0.9):
                self.logger.warning(f"Data quality issue - {metric}: {score:.3f}")
                
                error_tracker.record_error(
                    f"data_quality_{metric}",
                    DataQualityError(f"Low {metric} score: {score:.3f}"),
                    ErrorSeverity.MEDIUM,
                    ErrorCategory.DATA,
                    {'metric': metric, 'score': score, 'threshold': self.quality_thresholds.get(metric)}
                )
        
        return quality_scores


class ValidationPipeline:
    def __init__(self, name: str = "default"):
        self.name = name
        self.validators: List[BaseValidator] = []
        self.logger = get_logger(f"{__name__}.{name}")
        self.data_quality_checker = DataQualityChecker()
    
    def add_validator(self, validator: BaseValidator):
        self.validators.append(validator)
        self.logger.info(f"Added validator: {validator.name}")
    
    def validate(self, data: Any, check_quality: bool = True) -> ValidationResult:
        all_errors = []
        all_warnings = []
        cleaned_data = data
        all_metadata = {}
        
        # Run all validators
        for validator in self.validators:
            result = validator.validate(cleaned_data)
            
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            
            if result.cleaned_data is not None:
                cleaned_data = result.cleaned_data
            
            if result.metadata:
                all_metadata.update(result.metadata)
        
        # Run data quality check if requested
        if check_quality and isinstance(data, dict):
            quality_scores = self.data_quality_checker.assess_data_quality(data)
            all_metadata['quality_scores'] = quality_scores
            
            if quality_scores['overall'] < 0.8:
                all_warnings.append(f"Low overall data quality: {quality_scores['overall']:.3f}")
        
        final_result = ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            cleaned_data=cleaned_data,
            metadata=all_metadata
        )
        
        if not final_result.is_valid:
            self.logger.error(f"Validation pipeline {self.name} failed: {all_errors}")
        elif all_warnings:
            self.logger.warning(f"Validation pipeline {self.name} warnings: {all_warnings}")
        
        return final_result


# Predefined validation pipelines
def create_market_data_pipeline(level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationPipeline:
    pipeline = ValidationPipeline("market_data")
    pipeline.add_validator(MarketDataValidator(level=level))
    return pipeline


def create_trading_signal_pipeline(level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationPipeline:
    pipeline = ValidationPipeline("trading_signal")
    pipeline.add_validator(TradingSignalValidator(level=level))
    return pipeline


def create_config_pipeline(level: ValidationLevel = ValidationLevel.STRICT) -> ValidationPipeline:
    pipeline = ValidationPipeline("configuration")
    pipeline.add_validator(ConfigValidator(level=level))
    return pipeline


# Global validators for easy access
market_data_validator = create_market_data_pipeline()
trading_signal_validator = create_trading_signal_pipeline()
config_validator = create_config_pipeline()