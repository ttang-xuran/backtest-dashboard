#!/usr/bin/env python3
"""
Comprehensive Error Handling and Recovery System
"""

import asyncio
import functools
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Union, List
from enum import Enum
import logging

from utils.logger import get_logger


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    NETWORK = "network"
    API = "api"
    DATA = "data"
    CALCULATION = "calculation"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"


class RecoveryAction(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    CIRCUIT_BREAK = "circuit_break"


class ErrorTracker:
    def __init__(self):
        self.error_history: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self.last_error_time: Dict[str, datetime] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.logger = get_logger(__name__)
    
    def record_error(self, error_type: str, error: Exception, 
                    severity: ErrorSeverity, category: ErrorCategory,
                    context: Dict[str, Any] = None) -> None:
        error_record = {
            'timestamp': datetime.now(),
            'error_type': error_type,
            'error_message': str(error),
            'severity': severity.value,
            'category': category.value,
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_record)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_error_time[error_type] = datetime.now()
        
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
        
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        self.logger.log(log_level, f"Error recorded: {error_type} - {error}", 
                       extra={'extra_data': error_record})
    
    def get_error_rate(self, error_type: str, time_window_minutes: int = 60) -> float:
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_errors = [
            e for e in self.error_history 
            if e['error_type'] == error_type and e['timestamp'] > cutoff_time
        ]
        return len(recent_errors) / time_window_minutes
    
    def should_circuit_break(self, error_type: str, threshold: int = 5, 
                           time_window_minutes: int = 10) -> bool:
        error_rate = self.get_error_rate(error_type, time_window_minutes)
        return error_rate >= threshold
    
    def get_error_summary(self) -> Dict[str, Any]:
        now = datetime.now()
        recent_errors = [e for e in self.error_history if now - e['timestamp'] < timedelta(hours=24)]
        
        error_by_category = {}
        error_by_severity = {}
        
        for error in recent_errors:
            category = error['category']
            severity = error['severity']
            
            error_by_category[category] = error_by_category.get(category, 0) + 1
            error_by_severity[severity] = error_by_severity.get(severity, 0) + 1
        
        return {
            'total_errors_24h': len(recent_errors),
            'error_by_category': error_by_category,
            'error_by_severity': error_by_severity,
            'most_frequent_errors': sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }


error_tracker = ErrorTracker()


def handle_errors(retry_count: int = 3, 
                 retry_delay: float = 1.0,
                 fallback_value: Any = None,
                 error_category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 circuit_breaker_threshold: int = 5):
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            func_name = f"{func.__module__}.{func.__name__}"
            
            for attempt in range(retry_count + 1):
                try:
                    return await func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    error_type = f"{func_name}_{type(e).__name__}"
                    
                    context = {
                        'function': func_name,
                        'attempt': attempt + 1,
                        'max_attempts': retry_count + 1,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                    
                    error_tracker.record_error(error_type, e, severity, error_category, context)
                    
                    if error_tracker.should_circuit_break(error_type, circuit_breaker_threshold):
                        error_tracker.logger.critical(f"Circuit breaker triggered for {func_name}")
                        if fallback_value is not None:
                            return fallback_value
                        raise CircuitBreakerError(f"Circuit breaker active for {func_name}")
                    
                    if attempt < retry_count:
                        delay = retry_delay * (2 ** attempt)  # Exponential backoff
                        error_tracker.logger.warning(f"Retrying {func_name} in {delay}s (attempt {attempt + 1}/{retry_count + 1})")
                        await asyncio.sleep(delay)
                    else:
                        if fallback_value is not None:
                            error_tracker.logger.warning(f"Using fallback value for {func_name}")
                            return fallback_value
            
            error_tracker.logger.error(f"All retry attempts failed for {func_name}")
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            func_name = f"{func.__module__}.{func.__name__}"
            
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    error_type = f"{func_name}_{type(e).__name__}"
                    
                    context = {
                        'function': func_name,
                        'attempt': attempt + 1,
                        'max_attempts': retry_count + 1,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                    
                    error_tracker.record_error(error_type, e, severity, error_category, context)
                    
                    if error_tracker.should_circuit_break(error_type, circuit_breaker_threshold):
                        error_tracker.logger.critical(f"Circuit breaker triggered for {func_name}")
                        if fallback_value is not None:
                            return fallback_value
                        raise CircuitBreakerError(f"Circuit breaker active for {func_name}")
                    
                    if attempt < retry_count:
                        delay = retry_delay * (2 ** attempt)
                        error_tracker.logger.warning(f"Retrying {func_name} in {delay}s (attempt {attempt + 1}/{retry_count + 1})")
                        import time
                        time.sleep(delay)
                    else:
                        if fallback_value is not None:
                            error_tracker.logger.warning(f"Using fallback value for {func_name}")
                            return fallback_value
            
            error_tracker.logger.error(f"All retry attempts failed for {func_name}")
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class CircuitBreakerError(Exception):
    pass


class ValidationError(Exception):
    pass


class DataQualityError(Exception):
    pass


class TradingError(Exception):
    pass


class RiskLimitError(Exception):
    pass


def validate_input(validation_func: Callable, error_message: str = "Validation failed"):
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                if not validation_func(*args, **kwargs):
                    raise ValidationError(error_message)
                return await func(*args, **kwargs)
            except ValidationError:
                raise
            except Exception as e:
                error_tracker.record_error(
                    f"validation_{func.__name__}",
                    e,
                    ErrorSeverity.MEDIUM,
                    ErrorCategory.DATA,
                    {'function': func.__name__, 'validation_error': error_message}
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                if not validation_func(*args, **kwargs):
                    raise ValidationError(error_message)
                return func(*args, **kwargs)
            except ValidationError:
                raise
            except Exception as e:
                error_tracker.record_error(
                    f"validation_{func.__name__}",
                    e,
                    ErrorSeverity.MEDIUM,
                    ErrorCategory.DATA,
                    {'function': func.__name__, 'validation_error': error_message}
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class ErrorRecoveryManager:
    def __init__(self):
        self.recovery_strategies: Dict[str, Callable] = {}
        self.logger = get_logger(__name__)
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type}")
    
    async def attempt_recovery(self, error_type: str, error: Exception, 
                             context: Dict[str, Any] = None) -> Any:
        if error_type not in self.recovery_strategies:
            self.logger.warning(f"No recovery strategy found for {error_type}")
            return None
        
        try:
            recovery_func = self.recovery_strategies[error_type]
            if asyncio.iscoroutinefunction(recovery_func):
                result = await recovery_func(error, context)
            else:
                result = recovery_func(error, context)
            
            self.logger.info(f"Successfully recovered from {error_type}")
            return result
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy failed for {error_type}: {recovery_error}")
            error_tracker.record_error(
                f"recovery_failure_{error_type}",
                recovery_error,
                ErrorSeverity.HIGH,
                ErrorCategory.SYSTEM,
                {'original_error': str(error), 'context': context}
            )
            return None


recovery_manager = ErrorRecoveryManager()


async def network_recovery_strategy(error: Exception, context: Dict[str, Any] = None) -> Optional[Any]:
    logger = get_logger(__name__)
    logger.info("Attempting network recovery...")
    
    await asyncio.sleep(5)
    
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('https://httpbin.org/status/200', timeout=10) as response:
                if response.status == 200:
                    logger.info("Network connectivity restored")
                    return True
    except Exception:
        pass
    
    return False


def api_fallback_strategy(error: Exception, context: Dict[str, Any] = None) -> Optional[Any]:
    logger = get_logger(__name__)
    logger.info("Using API fallback strategy...")
    
    if context and 'fallback_data' in context:
        return context['fallback_data']
    
    return None


recovery_manager.register_recovery_strategy('network_error', network_recovery_strategy)
recovery_manager.register_recovery_strategy('api_error', api_fallback_strategy)


def safe_execute(func: Callable, *args, default_value: Any = None, 
                log_errors: bool = True, **kwargs) -> Any:
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger = get_logger(__name__)
            logger.error(f"Safe execution failed for {func.__name__}: {e}")
            
            error_tracker.record_error(
                f"safe_execute_{func.__name__}",
                e,
                ErrorSeverity.LOW,
                ErrorCategory.SYSTEM,
                {'function': func.__name__, 'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
            )
        
        return default_value


async def safe_execute_async(func: Callable, *args, default_value: Any = None, 
                           log_errors: bool = True, **kwargs) -> Any:
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger = get_logger(__name__)
            logger.error(f"Safe async execution failed for {func.__name__}: {e}")
            
            error_tracker.record_error(
                f"safe_execute_async_{func.__name__}",
                e,
                ErrorSeverity.LOW,
                ErrorCategory.SYSTEM,
                {'function': func.__name__, 'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
            )
        
        return default_value


class HealthMonitor:
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.health_status: Dict[str, bool] = {}
        self.logger = get_logger(__name__)
    
    def register_health_check(self, name: str, check_func: Callable):
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'error_summary': error_tracker.get_error_summary()
        }
        
        failed_checks = []
        
        for name, check_func in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                health_report['checks'][name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'last_check': datetime.now().isoformat(),
                    'result': result
                }
                
                self.health_status[name] = result
                self.last_health_check[name] = datetime.now()
                
                if not result:
                    failed_checks.append(name)
                    
            except Exception as e:
                health_report['checks'][name] = {
                    'status': 'error',
                    'last_check': datetime.now().isoformat(),
                    'error': str(e)
                }
                failed_checks.append(name)
                self.health_status[name] = False
        
        if failed_checks:
            health_report['overall_status'] = 'degraded' if len(failed_checks) < len(self.health_checks) / 2 else 'unhealthy'
            health_report['failed_checks'] = failed_checks
        
        return health_report


health_monitor = HealthMonitor()