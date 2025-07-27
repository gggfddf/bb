"""
Error handler utility for comprehensive error handling and retry mechanisms.
"""

import asyncio
import random
import logging
import json
from typing import Optional, Callable, Dict, Any, List
from enum import Enum
from functools import wraps
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import traceback
import structlog

logger = structlog.get_logger()

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorEvent:
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    source: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    retry_count: int = 0
    resolved: bool = False

class ErrorHandler:
    def __init__(self, 
                 max_retries: int = 3, 
                 base_delay: float = 1.0, 
                 circuit_breaker_threshold: int = 5, 
                 circuit_breaker_timeout: float = 60.0,
                 enable_alerting: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.enable_alerting = enable_alerting
        
        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        
        # Error tracking
        self.error_events: List[ErrorEvent] = []
        self.error_stats: Dict[str, int] = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            ErrorSeverity.LOW: 10,
            ErrorSeverity.MEDIUM: 5,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 1
        }

    def retry_with_backoff(self, 
                          retry_exceptions: tuple = (Exception,), 
                          max_retries: Optional[int] = None, 
                          base_delay: Optional[float] = None,
                          jitter: bool = True):
        """
        Decorator for retry logic with exponential backoff and jitter
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                retries = max_retries or self.max_retries
                delay = base_delay or self.base_delay
                
                for attempt in range(retries + 1):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                    except retry_exceptions as e:
                        if attempt == retries:
                            await self._handle_final_failure(e, func.__name__, args, kwargs)
                            raise
                        
                        wait_time = delay * (2 ** attempt)
                        if jitter:
                            wait_time += random.uniform(0, 0.1 * wait_time)
                        
                        await self._handle_retry(e, attempt, wait_time, func.__name__)
                        await asyncio.sleep(wait_time)
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                retries = max_retries or self.max_retries
                delay = base_delay or self.base_delay
                
                for attempt in range(retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except retry_exceptions as e:
                        if attempt == retries:
                            self._handle_final_failure_sync(e, func.__name__, args, kwargs)
                            raise
                        
                        wait_time = delay * (2 ** attempt)
                        if jitter:
                            wait_time += random.uniform(0, 0.1 * wait_time)
                        
                        self._handle_retry_sync(e, attempt, wait_time, func.__name__)
                        time.sleep(wait_time)
                
                return None
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator

    async def _handle_retry(self, exception: Exception, attempt: int, wait_time: float, func_name: str):
        """Handle retry attempt with logging and metrics"""
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_type=type(exception).__name__,
            error_message=str(exception),
            severity=self._determine_severity(exception),
            source=func_name,
            context={
                "attempt": attempt + 1,
                "wait_time": wait_time,
                "function": func_name
            },
            stack_trace=traceback.format_exc(),
            retry_count=attempt
        )
        
        self._record_error_event(error_event)
        logger.warning(
            "Retry attempt",
            attempt=attempt + 1,
            function=func_name,
            error=str(exception),
            wait_time=wait_time
        )

    def _handle_retry_sync(self, exception: Exception, attempt: int, wait_time: float, func_name: str):
        """Handle retry attempt for synchronous functions"""
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_type=type(exception).__name__,
            error_message=str(exception),
            severity=self._determine_severity(exception),
            source=func_name,
            context={
                "attempt": attempt + 1,
                "wait_time": wait_time,
                "function": func_name
            },
            stack_trace=traceback.format_exc(),
            retry_count=attempt
        )
        
        self._record_error_event(error_event)
        logger.warning(
            "Retry attempt",
            attempt=attempt + 1,
            function=func_name,
            error=str(exception),
            wait_time=wait_time
        )

    async def _handle_final_failure(self, exception: Exception, func_name: str, args: tuple, kwargs: dict):
        """Handle final failure after all retries exhausted"""
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_type=type(exception).__name__,
            error_message=str(exception),
            severity=self._determine_severity(exception),
            source=func_name,
            context={
                "function": func_name,
                "args": str(args),
                "kwargs": str(kwargs)
            },
            stack_trace=traceback.format_exc(),
            retry_count=self.max_retries
        )
        
        self._record_error_event(error_event)
        self._record_failure(exception)
        
        logger.error(
            "Final failure after retries",
            function=func_name,
            error=str(exception),
            retry_count=self.max_retries
        )
        
        if self.enable_alerting:
            await self._send_alert(error_event)

    def _handle_final_failure_sync(self, exception: Exception, func_name: str, args: tuple, kwargs: dict):
        """Handle final failure for synchronous functions"""
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_type=type(exception).__name__,
            error_message=str(exception),
            severity=self._determine_severity(exception),
            source=func_name,
            context={
                "function": func_name,
                "args": str(args),
                "kwargs": str(kwargs)
            },
            stack_trace=traceback.format_exc(),
            retry_count=self.max_retries
        )
        
        self._record_error_event(error_event)
        self._record_failure(exception)
        
        logger.error(
            "Final failure after retries",
            function=func_name,
            error=str(exception),
            retry_count=self.max_retries
        )

    def _record_failure(self, exception: Exception):
        """Record failure for circuit breaker logic"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.circuit_breaker_threshold:
            self.circuit_breaker_state = CircuitBreakerState.OPEN
            logger.error(
                "Circuit breaker opened",
                failure_count=self.failure_count,
                threshold=self.circuit_breaker_threshold
            )

    def _record_error_event(self, error_event: ErrorEvent):
        """Record error event for tracking and analysis"""
        self.error_events.append(error_event)
        
        # Update error statistics
        error_type = error_event.error_type
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1
        
        # Check if we should send an alert
        if self._should_send_alert(error_event):
            asyncio.create_task(self._send_alert(error_event))

    def _determine_severity(self, exception: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type and message"""
        error_message = str(exception).lower()
        
        # Critical errors
        if any(keyword in error_message for keyword in ['database', 'connection', 'timeout', 'authentication']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_message for keyword in ['api', 'rate limit', 'quota', 'validation']):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if any(keyword in error_message for keyword in ['parsing', 'format', 'data']):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW

    def _should_send_alert(self, error_event: ErrorEvent) -> bool:
        """Determine if an alert should be sent based on severity and frequency"""
        severity = error_event.severity
        threshold = self.alert_thresholds.get(severity, 1)
        
        # Count recent errors of this severity
        recent_errors = [
            e for e in self.error_events 
            if e.severity == severity and 
            e.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        return len(recent_errors) >= threshold

    async def _send_alert(self, error_event: ErrorEvent):
        """Send alert for critical errors"""
        alert_data = {
            "timestamp": error_event.timestamp.isoformat(),
            "severity": error_event.severity.value,
            "error_type": error_event.error_type,
            "error_message": error_event.error_message,
            "source": error_event.source,
            "context": error_event.context
        }
        
        logger.error(
            "Sending alert",
            alert_data=alert_data
        )
        
        # TODO: Integrate with actual alerting system (email, Slack, etc.)
        # For now, just log the alert

    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        if self.circuit_breaker_state == CircuitBreakerState.CLOSED:
            return True
        
        if self.circuit_breaker_state == CircuitBreakerState.OPEN:
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.circuit_breaker_timeout):
                self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        
        # HALF_OPEN state - allow one attempt
        return True

    def record_success(self):
        """Record successful execution to reset circuit breaker"""
        if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            self.circuit_breaker_state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            logger.info("Circuit breaker reset to closed state")

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error statistics"""
        return {
            "total_errors": len(self.error_events),
            "error_stats": self.error_stats,
            "circuit_breaker_state": self.circuit_breaker_state.value,
            "failure_count": self.failure_count,
            "recent_errors": [
                asdict(e) for e in self.error_events[-10:]  # Last 10 errors
            ]
        }

    def clear_old_errors(self, hours: int = 24):
        """Clear old error events to prevent memory bloat"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        self.error_events = [
            e for e in self.error_events 
            if e.timestamp > cutoff_time
        ]
        logger.info(f"Cleared error events older than {hours} hours")

# Global error handler instance
global_error_handler = ErrorHandler()

# Convenience functions for common error handling patterns
async def handle_data_collection_error(func: Callable, *args, **kwargs):
    """Handle errors specifically for data collection operations"""
    try:
        if not global_error_handler.can_execute():
            raise Exception("Circuit breaker is open")
        
        result = await func(*args, **kwargs)
        global_error_handler.record_success()
        return result
    except Exception as e:
        await global_error_handler._handle_final_failure(e, func.__name__, args, kwargs)
        raise

def handle_validation_error(data: Dict[str, Any], validator_func: Callable) -> Dict[str, Any]:
    """Handle errors specifically for data validation operations"""
    try:
        return validator_func(data)
    except Exception as e:
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_type=type(e).__name__,
            error_message=str(e),
            severity=ErrorSeverity.MEDIUM,
            source="data_validation",
            context={"data_keys": list(data.keys())},
            stack_trace=traceback.format_exc()
        )
        global_error_handler._record_error_event(error_event)
        raise