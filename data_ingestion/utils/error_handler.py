"""
Error handler utility for comprehensive error handling and retry mechanisms.
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Error types for categorization."""
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    DATABASE = "database"
    PARSING = "parsing"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class ErrorHandler:
    """
    Comprehensive error handler with retry mechanisms and circuit breaker pattern.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0
    ):
        """
        Initialize the error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            circuit_breaker_threshold: Number of failures before opening circuit
            circuit_breaker_timeout: Time to wait before half-opening circuit
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        # Circuit breaker state
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        
        # Error tracking
        self.error_history = []
        self.error_counts = {}
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.retried_requests = 0
    
    def retry_with_backoff(
        self,
        retry_exceptions: tuple = (Exception,),
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None
    ):
        """
        Decorator for retry logic with exponential backoff.
        
        Args:
            retry_exceptions: Tuple of exceptions to retry on
            max_retries: Override max retries for this function
            base_delay: Override base delay for this function
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._retry_async(
                    func, *args, **kwargs,
                    retry_exceptions=retry_exceptions,
                    max_retries=max_retries or self.max_retries,
                    base_delay=base_delay or self.base_delay
                )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._retry_sync(
                    func, *args, **kwargs,
                    retry_exceptions=retry_exceptions,
                    max_retries=max_retries or self.max_retries,
                    base_delay=base_delay or self.base_delay
                )
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _retry_async(
        self,
        func: Callable,
        *args,
        retry_exceptions: tuple = (Exception,),
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs
    ):
        """Async retry implementation."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Check circuit breaker
                if self.circuit_breaker_state == CircuitBreakerState.OPEN:
                    if self._should_half_open():
                        self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                        logger.info("Circuit breaker half-opening")
                    else:
                        raise Exception("Circuit breaker is open")
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Record success
                self._record_success()
                return result
                
            except retry_exceptions as e:
                last_exception = e
                self._record_failure(e)
                
                if attempt < max_retries:
                    delay = self._calculate_delay(attempt, base_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
        
        raise last_exception
    
    def _retry_sync(
        self,
        func: Callable,
        *args,
        retry_exceptions: tuple = (Exception,),
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs
    ):
        """Sync retry implementation."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Check circuit breaker
                if self.circuit_breaker_state == CircuitBreakerState.OPEN:
                    if self._should_half_open():
                        self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                        logger.info("Circuit breaker half-opening")
                    else:
                        raise Exception("Circuit breaker is open")
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Record success
                self._record_success()
                return result
                
            except retry_exceptions as e:
                last_exception = e
                self._record_failure(e)
                
                if attempt < max_retries:
                    delay = self._calculate_delay(attempt, base_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate delay for exponential backoff."""
        delay = base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25%)
            jitter = random.uniform(0.75, 1.25)
            delay *= jitter
        
        return delay
    
    def _record_success(self):
        """Record a successful request."""
        self.successful_requests += 1
        self.success_count += 1
        self.failure_count = 0
        
        if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= 3:  # Require 3 successes to close circuit
                self.circuit_breaker_state = CircuitBreakerState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker closed")
    
    def _record_failure(self, exception: Exception):
        """Record a failed request."""
        self.failed_requests += 1
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = datetime.utcnow()
        
        # Categorize error
        error_type = self._categorize_error(exception)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to error history
        self.error_history.append({
            'timestamp': datetime.utcnow(),
            'error_type': error_type,
            'error_message': str(exception),
            'failure_count': self.failure_count
        })
        
        # Keep only recent errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        # Check circuit breaker
        if self.failure_count >= self.circuit_breaker_threshold:
            self.circuit_breaker_state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _should_half_open(self) -> bool:
        """Check if circuit breaker should half-open."""
        if self.last_failure_time is None:
            return False
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.circuit_breaker_timeout
    
    def _categorize_error(self, exception: Exception) -> ErrorType:
        """Categorize an exception by type."""
        error_message = str(exception).lower()
        
        if any(word in error_message for word in ['timeout', 'timed out']):
            return ErrorType.TIMEOUT
        elif any(word in error_message for word in ['rate limit', 'too many requests', '429']):
            return ErrorType.RATE_LIMIT
        elif any(word in error_message for word in ['unauthorized', '401', 'authentication']):
            return ErrorType.AUTHENTICATION
        elif any(word in error_message for word in ['forbidden', '403', 'authorization']):
            return ErrorType.AUTHORIZATION
        elif any(word in error_message for word in ['validation', 'invalid']):
            return ErrorType.VALIDATION
        elif any(word in error_message for word in ['database', 'sql', 'connection']):
            return ErrorType.DATABASE
        elif any(word in error_message for word in ['parse', 'json', 'xml']):
            return ErrorType.PARSING
        elif any(word in error_message for word in ['network', 'connection', 'dns']):
            return ErrorType.NETWORK
        else:
            return ErrorType.UNKNOWN
    
    def get_error_severity(self, error_type: ErrorType) -> ErrorSeverity:
        """Get severity level for an error type."""
        severity_map = {
            ErrorType.NETWORK: ErrorSeverity.MEDIUM,
            ErrorType.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorType.RATE_LIMIT: ErrorSeverity.LOW,
            ErrorType.AUTHENTICATION: ErrorSeverity.HIGH,
            ErrorType.AUTHORIZATION: ErrorSeverity.HIGH,
            ErrorType.VALIDATION: ErrorSeverity.MEDIUM,
            ErrorType.DATABASE: ErrorSeverity.HIGH,
            ErrorType.PARSING: ErrorSeverity.MEDIUM,
            ErrorType.UNKNOWN: ErrorSeverity.MEDIUM
        }
        return severity_map.get(error_type, ErrorSeverity.MEDIUM)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error handler statistics."""
        total_requests = self.successful_requests + self.failed_requests
        success_rate = (self.successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate,
            'retried_requests': self.retried_requests,
            'circuit_breaker_state': self.circuit_breaker_state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'error_counts': {error_type.value: count for error_type, count in self.error_counts.items()},
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def reset_stats(self):
        """Reset error handler statistics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.retried_requests = 0
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("Error handler statistics reset")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker to closed state."""
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker reset to closed state")
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_errors = [
            error for error in self.error_history
            if error['timestamp'] >= cutoff_time
        ]
        
        if not recent_errors:
            return {
                'period_hours': hours,
                'total_errors': 0,
                'error_types': {},
                'most_common_error': None,
                'error_trend': 'stable'
            }
        
        # Count errors by type
        error_type_counts = {}
        for error in recent_errors:
            error_type = error['error_type'].value
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        # Find most common error
        most_common_error = max(error_type_counts.items(), key=lambda x: x[1])[0] if error_type_counts else None
        
        # Calculate error trend (simplified)
        first_half = [e for e in recent_errors if e['timestamp'] < cutoff_time + timedelta(hours=hours/2)]
        second_half = [e for e in recent_errors if e['timestamp'] >= cutoff_time + timedelta(hours=hours/2)]
        
        if len(first_half) > len(second_half):
            trend = 'decreasing'
        elif len(first_half) < len(second_half):
            trend = 'increasing'
        else:
            trend = 'stable'
        
        return {
            'period_hours': hours,
            'total_errors': len(recent_errors),
            'error_types': error_type_counts,
            'most_common_error': most_common_error,
            'error_trend': trend,
            'recent_errors': recent_errors[-5:]  # Last 5 errors
        }


class RetryableError(Exception):
    """Exception that can be retried."""
    pass


class NonRetryableError(Exception):
    """Exception that should not be retried."""
    pass


class RateLimitError(RetryableError):
    """Rate limit exceeded error."""
    pass


class TimeoutError(RetryableError):
    """Timeout error."""
    pass


class NetworkError(RetryableError):
    """Network-related error."""
    pass


class ValidationError(NonRetryableError):
    """Data validation error."""
    pass


class AuthenticationError(NonRetryableError):
    """Authentication error."""
    pass


# Example usage
async def test_error_handler():
    """Test the error handler functionality."""
    error_handler = ErrorHandler(
        max_retries=3,
        base_delay=1.0,
        circuit_breaker_threshold=3
    )
    
    # Test function that fails
    @error_handler.retry_with_backoff(retry_exceptions=(Exception,))
    async def failing_function(attempt: int):
        if attempt < 2:
            raise Exception(f"Simulated failure {attempt}")
        return "Success!"
    
    try:
        result = await failing_function(0)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Final error: {e}")
    
    # Get stats
    stats = error_handler.get_stats()
    print(f"Error handler stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_error_handler())