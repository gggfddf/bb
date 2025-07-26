"""
Circuit Breaker Implementation for Trading System

This module provides circuit breaker pattern implementation for preventing
cascading failures and providing fault tolerance in the trading system.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import functools
import hashlib

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker state enumeration"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service is recovered

class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception,
                 monitor_interval: int = 10,
                 success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.monitor_interval = monitor_interval
        self.success_threshold = success_threshold

@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    current_failures: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    state_changes: int
    total_downtime: float  # seconds

class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = datetime.utcnow()
        
        # Statistics
        self.stats = CircuitBreakerStats(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            current_failures=0,
            last_failure_time=None,
            last_success_time=None,
            state_changes=0,
            total_downtime=0.0
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks
        self.on_open_callbacks: List[Callable] = []
        self.on_close_callbacks: List[Callable] = []
        self.on_half_open_callbacks: List[Callable] = []
        
        logger.info(f"Circuit breaker initialized: {name}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original function exception
        """
        if not self.can_execute():
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
            
        except self.config.expected_exception as e:
            self.on_failure(e)
            raise
    
    def can_execute(self) -> bool:
        """
        Check if the circuit breaker allows execution.
        
        Returns:
            True if execution is allowed
        """
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
        
        return False
    
    def on_success(self):
        """Handle successful execution"""
        with self._lock:
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.last_success_time = datetime.utcnow()
            
            if self.state == CircuitState.CLOSED:
                self.failure_count = 0
            elif self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
    
    def on_failure(self, exception: Exception):
        """Handle failed execution"""
        with self._lock:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            self.stats.current_failures += 1
            self.stats.last_failure_time = datetime.utcnow()
            
            if self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def force_open(self):
        """Force the circuit breaker to open state"""
        with self._lock:
            if self.state != CircuitState.OPEN:
                self._transition_to_open()
    
    def force_close(self):
        """Force the circuit breaker to closed state"""
        with self._lock:
            if self.state != CircuitState.CLOSED:
                self._transition_to_closed()
    
    def reset(self):
        """Reset the circuit breaker to initial state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_state_change = datetime.utcnow()
            self.stats.current_failures = 0
            self.stats.state_changes += 1
            
            logger.info(f"Circuit breaker {self.name} reset")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        return self.state
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics"""
        with self._lock:
            return CircuitBreakerStats(
                total_requests=self.stats.total_requests,
                successful_requests=self.stats.successful_requests,
                failed_requests=self.stats.failed_requests,
                current_failures=self.stats.current_failures,
                last_failure_time=self.stats.last_failure_time,
                last_success_time=self.stats.last_success_time,
                state_changes=self.stats.state_changes,
                total_downtime=self.stats.total_downtime
            )
    
    def add_on_open_callback(self, callback: Callable):
        """Add callback for when circuit opens"""
        self.on_open_callbacks.append(callback)
    
    def add_on_close_callback(self, callback: Callable):
        """Add callback for when circuit closes"""
        self.on_close_callbacks.append(callback)
    
    def add_on_half_open_callback(self, callback: Callable):
        """Add callback for when circuit goes half-open"""
        self.on_half_open_callbacks.append(callback)
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.last_failure_time:
            return False
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout
    
    def _transition_to_open(self):
        """Transition circuit breaker to open state"""
        if self.state != CircuitState.OPEN:
            old_state = self.state
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.utcnow()
            self.stats.state_changes += 1
            
            # Calculate downtime
            if old_state == CircuitState.CLOSED:
                self.stats.total_downtime += (datetime.utcnow() - self.last_state_change).total_seconds()
            
            logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
            
            # Trigger callbacks
            for callback in self.on_open_callbacks:
                try:
                    callback(self.name, self.failure_count)
                except Exception as e:
                    logger.error(f"Error in circuit breaker open callback: {e}")
    
    def _transition_to_closed(self):
        """Transition circuit breaker to closed state"""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_state_change = datetime.utcnow()
            self.stats.state_changes += 1
            
            logger.info(f"Circuit breaker {self.name} closed")
            
            # Trigger callbacks
            for callback in self.on_close_callbacks:
                try:
                    callback(self.name)
                except Exception as e:
                    logger.error(f"Error in circuit breaker close callback: {e}")
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state"""
        if self.state != CircuitState.HALF_OPEN:
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            self.last_state_change = datetime.utcnow()
            self.stats.state_changes += 1
            
            logger.info(f"Circuit breaker {self.name} half-open")
            
            # Trigger callbacks
            for callback in self.on_half_open_callbacks:
                try:
                    callback(self.name)
                except Exception as e:
                    logger.error(f"Error in circuit breaker half-open callback: {e}")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        
        logger.info("Circuit breaker manager initialized")
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """
        Create a new circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
            
        Returns:
            Created circuit breaker
        """
        with self._lock:
            if name in self.circuit_breakers:
                logger.warning(f"Circuit breaker {name} already exists")
                return self.circuit_breakers[name]
            
            circuit_breaker = CircuitBreaker(name, config)
            self.circuit_breakers[name] = circuit_breaker
            
            logger.info(f"Circuit breaker created: {name}")
            return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker by name.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            Circuit breaker if found
        """
        return self.circuit_breakers.get(name)
    
    def remove_circuit_breaker(self, name: str) -> bool:
        """
        Remove a circuit breaker.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            True if circuit breaker was removed
        """
        with self._lock:
            if name in self.circuit_breakers:
                del self.circuit_breakers[name]
                logger.info(f"Circuit breaker removed: {name}")
                return True
            return False
    
    def get_all_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers"""
        return self.circuit_breakers.copy()
    
    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all circuit breakers.
        
        Returns:
            Circuit breaker statistics
        """
        stats = {}
        
        for name, cb in self.circuit_breakers.items():
            stats[name] = {
                'state': cb.get_state().value,
                'stats': asdict(cb.get_stats())
            }
        
        return stats
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for cb in self.circuit_breakers.values():
                cb.reset()
            
            logger.info("All circuit breakers reset")


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for applying circuit breaker pattern to functions.
    
    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
        
    Returns:
        Decorated function
    """
    def decorator(func):
        if config is None:
            default_config = CircuitBreakerConfig()
        else:
            default_config = config
        
        cb = get_circuit_breaker_manager().create_circuit_breaker(name, default_config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


# Global instance management
_circuit_breaker_manager_instance = None

def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get or create circuit breaker manager instance"""
    global _circuit_breaker_manager_instance
    
    if _circuit_breaker_manager_instance is None:
        _circuit_breaker_manager_instance = CircuitBreakerManager()
    
    return _circuit_breaker_manager_instance


def init_circuit_breaker_manager() -> CircuitBreakerManager:
    """Initialize circuit breaker manager"""
    global _circuit_breaker_manager_instance
    
    _circuit_breaker_manager_instance = CircuitBreakerManager()
    
    return _circuit_breaker_manager_instance


# Example usage:
# @circuit_breaker("database_connection", CircuitBreakerConfig(failure_threshold=3))
# def database_operation():
#     # Database operation code
#     pass