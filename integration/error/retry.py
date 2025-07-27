"""
Retry Mechanisms for Trading System

This module provides comprehensive retry mechanisms for handling transient
failures and implementing retry policies in the trading system.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import functools
import random
import hashlib

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    """Retry strategy enumeration"""
    FIXED = "fixed"           # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"         # Linear backoff
    RANDOM = "random"         # Random delay
    FIBONACCI = "fibonacci"   # Fibonacci backoff

class RetryStatus(Enum):
    """Retry status enumeration"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    timeout: Optional[float] = None  # seconds
    retry_on_exceptions: List[Type[Exception]] = None
    retry_on_result: Optional[Callable] = None

@dataclass
class RetryAttempt:
    """Retry attempt information"""
    attempt_number: int
    start_time: datetime
    end_time: Optional[datetime]
    delay: float
    exception: Optional[Exception]
    result: Optional[Any]
    duration: Optional[float]

@dataclass
class RetryResult:
    """Retry result"""
    success: bool
    attempts: List[RetryAttempt]
    final_result: Optional[Any]
    final_exception: Optional[Exception]
    total_duration: float
    total_attempts: int

class RetryManager:
    """
    Retry manager for handling retry operations.
    """
    
    def __init__(self):
        self.retry_configs: Dict[str, RetryConfig] = {}
        self.retry_history: Dict[str, List[RetryResult]] = {}
        self._lock = threading.RLock()
        
        logger.info("Retry manager initialized")
    
    def register_retry_config(self, name: str, config: RetryConfig) -> bool:
        """
        Register a retry configuration.
        
        Args:
            name: Configuration name
            config: Retry configuration
            
        Returns:
            True if configuration was registered successfully
        """
        try:
            self.retry_configs[name] = config
            logger.info(f"Retry config registered: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register retry config: {e}")
            return False
    
    def retry(self, name: str, func: Callable, *args, **kwargs) -> RetryResult:
        """
        Execute a function with retry logic.
        
        Args:
            name: Retry configuration name
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Retry result
        """
        config = self.retry_configs.get(name)
        if not config:
            raise ValueError(f"Retry config not found: {name}")
        
        return self._execute_with_retry(func, config, *args, **kwargs)
    
    def retry_with_config(self, config: RetryConfig, func: Callable, *args, **kwargs) -> RetryResult:
        """
        Execute a function with specific retry configuration.
        
        Args:
            config: Retry configuration
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Retry result
        """
        return self._execute_with_retry(func, config, *args, **kwargs)
    
    def get_retry_history(self, name: Optional[str] = None) -> Dict[str, List[RetryResult]]:
        """
        Get retry history.
        
        Args:
            name: Optional configuration name filter
            
        Returns:
            Retry history
        """
        if name:
            return {name: self.retry_history.get(name, [])}
        return self.retry_history.copy()
    
    def clear_retry_history(self, name: Optional[str] = None):
        """
        Clear retry history.
        
        Args:
            name: Optional configuration name filter
        """
        with self._lock:
            if name:
                if name in self.retry_history:
                    del self.retry_history[name]
            else:
                self.retry_history.clear()
    
    def _execute_with_retry(self, func: Callable, config: RetryConfig, *args, **kwargs) -> RetryResult:
        """Execute function with retry logic"""
        attempts = []
        start_time = datetime.utcnow()
        
        for attempt_num in range(config.max_attempts):
            attempt_start = datetime.utcnow()
            
            try:
                # Calculate delay for this attempt
                delay = self._calculate_delay(config, attempt_num)
                
                # Execute function
                if config.timeout:
                    result = self._execute_with_timeout(func, config.timeout, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Check if result should trigger retry
                if config.retry_on_result and config.retry_on_result(result):
                    raise RetryableResultError(f"Result triggered retry: {result}")
                
                # Success
                attempt = RetryAttempt(
                    attempt_number=attempt_num + 1,
                    start_time=attempt_start,
                    end_time=datetime.utcnow(),
                    delay=delay,
                    exception=None,
                    result=result,
                    duration=(datetime.utcnow() - attempt_start).total_seconds()
                )
                attempts.append(attempt)
                
                # Store in history
                self._store_retry_result(func.__name__, RetryResult(
                    success=True,
                    attempts=attempts,
                    final_result=result,
                    final_exception=None,
                    total_duration=(datetime.utcnow() - start_time).total_seconds(),
                    total_attempts=attempt_num + 1
                ))
                
                return RetryResult(
                    success=True,
                    attempts=attempts,
                    final_result=result,
                    final_exception=None,
                    total_duration=(datetime.utcnow() - start_time).total_seconds(),
                    total_attempts=attempt_num + 1
                )
                
            except Exception as e:
                # Check if exception should trigger retry
                if not self._should_retry_on_exception(e, config):
                    raise
                
                attempt = RetryAttempt(
                    attempt_number=attempt_num + 1,
                    start_time=attempt_start,
                    end_time=datetime.utcnow(),
                    delay=delay,
                    exception=e,
                    result=None,
                    duration=(datetime.utcnow() - attempt_start).total_seconds()
                )
                attempts.append(attempt)
                
                # If this is the last attempt, return failure
                if attempt_num == config.max_attempts - 1:
                    result = RetryResult(
                        success=False,
                        attempts=attempts,
                        final_result=None,
                        final_exception=e,
                        total_duration=(datetime.utcnow() - start_time).total_seconds(),
                        total_attempts=config.max_attempts
                    )
                    
                    # Store in history
                    self._store_retry_result(func.__name__, result)
                    
                    return result
                
                # Wait before next attempt
                if delay > 0:
                    time.sleep(delay)
        
        # Should not reach here
        raise RuntimeError("Unexpected end of retry loop")
    
    def _calculate_delay(self, config: RetryConfig, attempt_num: int) -> float:
        """Calculate delay for retry attempt"""
        if attempt_num == 0:
            return 0  # No delay for first attempt
        
        if config.strategy == RetryStrategy.FIXED:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.base_delay * (2 ** (attempt_num - 1))
        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.base_delay * attempt_num
        elif config.strategy == RetryStrategy.RANDOM:
            delay = random.uniform(0, config.base_delay * (2 ** (attempt_num - 1)))
        elif config.strategy == RetryStrategy.FIBONACCI:
            delay = config.base_delay * self._fibonacci(attempt_num)
        else:
            delay = config.base_delay
        
        # Apply jitter
        if config.jitter:
            jitter_factor = 0.1  # 10% jitter
            jitter = random.uniform(-jitter_factor * delay, jitter_factor * delay)
            delay += jitter
        
        # Cap at max delay
        return min(delay, config.max_delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number"""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)
    
    def _should_retry_on_exception(self, exception: Exception, config: RetryConfig) -> bool:
        """Check if exception should trigger retry"""
        if not config.retry_on_exceptions:
            return True  # Retry on all exceptions if not specified
        
        return any(isinstance(exception, exc_type) for exc_type in config.retry_on_exceptions)
    
    def _execute_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Execute function with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution timed out after {timeout} seconds")
        
        # Set up timeout handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _store_retry_result(self, func_name: str, result: RetryResult):
        """Store retry result in history"""
        with self._lock:
            if func_name not in self.retry_history:
                self.retry_history[func_name] = []
            
            self.retry_history[func_name].append(result)
            
            # Keep only last 100 results per function
            if len(self.retry_history[func_name]) > 100:
                self.retry_history[func_name] = self.retry_history[func_name][-100:]


class RetryableResultError(Exception):
    """Exception raised when result should trigger retry"""
    pass


def retry(name: str, config: Optional[RetryConfig] = None):
    """
    Decorator for applying retry logic to functions.
    
    Args:
        name: Retry configuration name
        config: Optional retry configuration (if not registered)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        retry_manager = get_retry_manager()
        
        # Register config if provided
        if config:
            retry_manager.register_retry_config(name, config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = retry_manager.retry(name, func, *args, **kwargs)
                if result.success:
                    return result.final_result
                else:
                    raise result.final_exception
            except Exception as e:
                if isinstance(e, RetryableResultError):
                    raise
                raise e
        
        return wrapper
    
    return decorator


def retry_with_config(config: RetryConfig):
    """
    Decorator for applying retry logic with specific configuration.
    
    Args:
        config: Retry configuration
        
    Returns:
        Decorated function
    """
    def decorator(func):
        retry_manager = get_retry_manager()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = retry_manager.retry_with_config(config, func, *args, **kwargs)
                if result.success:
                    return result.final_result
                else:
                    raise result.final_exception
            except Exception as e:
                if isinstance(e, RetryableResultError):
                    raise
                raise e
        
        return wrapper
    
    return decorator


# Global instance management
_retry_manager_instance = None

def get_retry_manager() -> RetryManager:
    """Get or create retry manager instance"""
    global _retry_manager_instance
    
    if _retry_manager_instance is None:
        _retry_manager_instance = RetryManager()
    
    return _retry_manager_instance


def init_retry_manager() -> RetryManager:
    """Initialize retry manager"""
    global _retry_manager_instance
    
    _retry_manager_instance = RetryManager()
    
    return _retry_manager_instance


# Example usage:
# @retry("database_operation", RetryConfig(max_attempts=3, base_delay=1.0))
# def database_operation():
#     # Database operation code
#     pass

# @retry_with_config(RetryConfig(max_attempts=5, strategy=RetryStrategy.EXPONENTIAL))
# def api_call():
#     # API call code
#     pass