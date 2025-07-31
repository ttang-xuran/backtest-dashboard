#!/usr/bin/env python3
"""
Circuit Breaker Implementation for API Calls
Provides fault tolerance and prevents cascade failures
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Union
from enum import Enum
import functools

from utils.logger import get_logger
from utils.error_handler import error_tracker, ErrorSeverity, ErrorCategory


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker active, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerConfig:
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Union[Exception, tuple] = Exception,
                 name: Optional[str] = None):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "default"


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        self.total_requests = 0
        self.logger = get_logger(f"{__name__}.{config.name}")
        
        self._state_change_listeners = []
    
    def add_state_change_listener(self, listener: Callable[[CircuitState, CircuitState], None]):
        self._state_change_listeners.append(listener)
    
    def _change_state(self, new_state: CircuitState):
        old_state = self.state
        self.state = new_state
        
        self.logger.info(f"Circuit breaker state changed: {old_state.value} -> {new_state.value}")
        
        for listener in self._state_change_listeners:
            try:
                listener(old_state, new_state)
            except Exception as e:
                self.logger.error(f"Error in state change listener: {e}")
    
    def _should_attempt_reset(self) -> bool:
        return (self.state == CircuitState.OPEN and 
                self.last_failure_time and
                datetime.now() - self.last_failure_time >= timedelta(seconds=self.config.recovery_timeout))
    
    def _record_success(self):
        self.failure_count = 0
        self.success_count += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self._change_state(CircuitState.CLOSED)
            self.logger.info(f"Circuit breaker recovered for {self.config.name}")
    
    def _record_failure(self, exception: Exception):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        error_tracker.record_error(
            f"circuit_breaker_{self.config.name}",
            exception,
            ErrorSeverity.HIGH,
            ErrorCategory.API,
            {
                'circuit_breaker': self.config.name,
                'failure_count': self.failure_count,
                'state': self.state.value
            }
        )
        
        if self.state == CircuitState.HALF_OPEN:
            self._change_state(CircuitState.OPEN)
            self.logger.warning(f"Circuit breaker re-opened for {self.config.name}")
        
        elif (self.state == CircuitState.CLOSED and 
              self.failure_count >= self.config.failure_threshold):
            
            self._change_state(CircuitState.OPEN)
            self.logger.error(f"Circuit breaker tripped for {self.config.name} "
                            f"(failures: {self.failure_count}/{self.config.failure_threshold})")
    
    def can_execute(self) -> bool:
        self.total_requests += 1
        
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._change_state(CircuitState.HALF_OPEN)
                self.logger.info(f"Circuit breaker attempting reset for {self.config.name}")
                return True
            return False
        
        # HALF_OPEN state - allow limited requests to test recovery
        return True
    
    async def call(self, func: Callable, *args, **kwargs):
        if not self.can_execute():
            raise CircuitBreakerOpenError(f"Circuit breaker is open for {self.config.name}")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._record_success()
            return result
            
        except self.config.expected_exception as e:
            self._record_failure(e)
            raise
        except Exception as e:
            # Unexpected exceptions don't count as failures for circuit breaking
            self.logger.warning(f"Unexpected exception in circuit breaker {self.config.name}: {e}")
            raise
    
    def get_state(self) -> CircuitState:
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        uptime_pct = ((self.success_count / self.total_requests) * 100) if self.total_requests > 0 else 100
        
        return {
            'name': self.config.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_requests': self.total_requests,
            'uptime_percentage': uptime_pct,
            'failure_threshold': self.config.failure_threshold,
            'recovery_timeout': self.config.recovery_timeout,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'time_until_retry': max(0, self.config.recovery_timeout - 
                                   (datetime.now() - self.last_failure_time).total_seconds()
                                   ) if self.last_failure_time and self.state == CircuitState.OPEN else 0
        }
    
    def reset(self):
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.logger.info(f"Circuit breaker manually reset for {self.config.name}")
        
        for listener in self._state_change_listeners:
            try:
                listener(old_state, self.state)
            except Exception as e:
                self.logger.error(f"Error in state change listener: {e}")


class CircuitBreakerOpenError(Exception):
    pass


class CircuitBreakerRegistry:
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self.logger = get_logger(__name__)
    
    def create_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        config.name = name
        breaker = CircuitBreaker(config)
        self._breakers[name] = breaker
        
        # Add logging listener
        def log_state_change(old_state: CircuitState, new_state: CircuitState):
            self.logger.info(f"Circuit breaker '{name}' changed state: {old_state.value} -> {new_state.value}")
        
        breaker.add_state_change_listener(log_state_change)
        
        self.logger.info(f"Created circuit breaker: {name}")
        return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        return self._breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    def reset_all(self):
        for breaker in self._breakers.values():
            breaker.reset()
        self.logger.info("Reset all circuit breakers")
    
    def get_health_summary(self) -> Dict[str, Any]:
        stats = self.get_all_stats()
        
        total_breakers = len(stats)
        open_breakers = sum(1 for s in stats.values() if s['state'] == 'open')
        half_open_breakers = sum(1 for s in stats.values() if s['state'] == 'half_open')
        
        overall_health = 'healthy'
        if open_breakers > 0:
            overall_health = 'degraded' if open_breakers < total_breakers * 0.5 else 'unhealthy'
        elif half_open_breakers > 0:
            overall_health = 'recovering'
        
        return {
            'overall_health': overall_health,
            'total_breakers': total_breakers,
            'open_breakers': open_breakers,
            'half_open_breakers': half_open_breakers,
            'closed_breakers': total_breakers - open_breakers - half_open_breakers,
            'stats': stats
        }


# Global registry
circuit_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str = None,
                  failure_threshold: int = 5,
                  recovery_timeout: int = 60,
                  expected_exception: Union[Exception, tuple] = Exception):
    def decorator(func: Callable):
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=breaker_name
        )
        
        breaker = circuit_registry.create_breaker(breaker_name, config)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(breaker.call(func, *args, **kwargs))
        
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper._circuit_breaker = breaker  # Allow access to breaker instance
        
        return wrapper
    
    return decorator


# Specific circuit breakers for different API types
def hyperliquid_circuit_breaker(func: Callable):
    return circuit_breaker(
        name=f"hyperliquid_{func.__name__}",
        failure_threshold=3,
        recovery_timeout=30,
        expected_exception=(ConnectionError, TimeoutError, Exception)
    )(func)


def market_data_circuit_breaker(func: Callable):
    return circuit_breaker(
        name=f"market_data_{func.__name__}",
        failure_threshold=5,
        recovery_timeout=10,
        expected_exception=(ConnectionError, TimeoutError, ValueError)
    )(func)


def trading_circuit_breaker(func: Callable):
    return circuit_breaker(
        name=f"trading_{func.__name__}",
        failure_threshold=2,
        recovery_timeout=60,
        expected_exception=Exception
    )(func)


class AdaptiveCircuitBreaker(CircuitBreaker):
    def __init__(self, config: CircuitBreakerConfig):
        super().__init__(config)
        self.response_times = []
        self.max_response_time = 5.0  # seconds
        self.slow_request_threshold = 0.8  # 80% of max response time
        
    async def call(self, func: Callable, *args, **kwargs):
        if not self.can_execute():
            raise CircuitBreakerOpenError(f"Circuit breaker is open for {self.config.name}")
        
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Keep only recent response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-50:]
            
            # Check if response time is consistently slow
            if (response_time > self.max_response_time or 
                (len(self.response_times) >= 10 and 
                 sum(self.response_times[-10:]) / 10 > self.slow_request_threshold * self.max_response_time)):
                
                self.logger.warning(f"Slow response detected for {self.config.name}: {response_time:.2f}s")
                # Count slow responses as partial failures
                self.failure_count += 0.5
            
            self._record_success()
            return result
            
        except self.config.expected_exception as e:
            self._record_failure(e)
            raise
        except Exception as e:
            self.logger.warning(f"Unexpected exception in adaptive circuit breaker {self.config.name}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        
        if self.response_times:
            stats.update({
                'avg_response_time': sum(self.response_times) / len(self.response_times),
                'max_response_time_recorded': max(self.response_times),
                'min_response_time_recorded': min(self.response_times),
                'slow_requests': sum(1 for rt in self.response_times if rt > self.slow_request_threshold * self.max_response_time)
            })
        
        return stats


def adaptive_circuit_breaker(name: str = None,
                           failure_threshold: int = 5,
                           recovery_timeout: int = 60,
                           expected_exception: Union[Exception, tuple] = Exception):
    def decorator(func: Callable):
        breaker_name = name or f"adaptive_{func.__module__}.{func.__name__}"
        
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=breaker_name
        )
        
        breaker = AdaptiveCircuitBreaker(config)
        circuit_registry._breakers[breaker_name] = breaker
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(breaker.call(func, *args, **kwargs))
        
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper._circuit_breaker = breaker
        
        return wrapper
    
    return decorator