#!/usr/bin/env python3
"""
Error Handling and Recovery Module

Implements comprehensive error handling and recovery for the trading system:
- Error classification and categorization
- Error recovery mechanisms
- Circuit breaker patterns
- Retry mechanisms
- Error reporting and analytics
- Fault tolerance systems

Features:
- Complete error classification and categorization system
- Advanced error recovery mechanisms with fallback strategies
- Circuit breaker pattern implementation for fault tolerance
- Configurable retry mechanisms with exponential backoff
- Comprehensive error reporting and analytics
- Fault tolerance and resilience patterns
- High-performance error handling and recovery
"""

import asyncio
import time
import threading
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
import functools
import inspect

logger = structlog.get_logger()

class ErrorSeverity(Enum):
    """Error severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error category enumeration."""
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    EXTERNAL_SERVICE = "external_service"
    TIMEOUT = "timeout"
    RESOURCE = "resource"

class ErrorRecoveryStrategy(Enum):
    """Error recovery strategy enumeration."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    DEGRADE = "degrade"
    FAIL_FAST = "fail_fast"
    IGNORE = "ignore"

class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class ErrorInfo:
    """Error information structure."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    error_category: ErrorCategory
    severity: ErrorSeverity
    component: str
    operation: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryAction:
    """Recovery action structure."""
    action_id: str
    error_id: str
    strategy: ErrorRecoveryStrategy
    timestamp: datetime
    success: bool
    duration: float
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorHandlerConfig:
    """Error handler configuration."""
    enable_error_tracking: bool = True
    enable_recovery: bool = True
    enable_circuit_breaker: bool = True
    enable_retry: bool = True
    max_error_history: int = 10000
    error_retention_days: int = 30
    default_retry_attempts: int = 3
    default_retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    enable_error_analytics: bool = True
    error_reporting_interval: float = 300.0  # seconds

class ErrorClassifier:
    """Error classification system."""
    
    def __init__(self):
        """Initialize error classifier."""
        self.error_patterns = {
            ErrorCategory.NETWORK: [
                "ConnectionError", "TimeoutError", "socket.error", "requests.exceptions",
                "urllib.error", "aiohttp.ClientError", "websockets.exceptions"
            ],
            ErrorCategory.DATABASE: [
                "DatabaseError", "OperationalError", "IntegrityError", "psycopg2",
                "sqlite3", "mysql.connector", "redis.ConnectionError"
            ],
            ErrorCategory.AUTHENTICATION: [
                "AuthenticationError", "UnauthorizedError", "PermissionDenied",
                "InvalidCredentials", "TokenExpired", "JWTError"
            ],
            ErrorCategory.AUTHORIZATION: [
                "AuthorizationError", "ForbiddenError", "AccessDenied",
                "InsufficientPermissions", "RoleRequired"
            ],
            ErrorCategory.VALIDATION: [
                "ValidationError", "ValueError", "TypeError", "AttributeError",
                "KeyError", "IndexError", "AssertionError"
            ],
            ErrorCategory.BUSINESS_LOGIC: [
                "BusinessLogicError", "InvalidStateError", "ConstraintViolation",
                "InsufficientFunds", "OrderRejected", "StrategyError"
            ],
            ErrorCategory.SYSTEM: [
                "SystemError", "OSError", "MemoryError", "ResourceError",
                "FileNotFoundError", "ProcessError"
            ],
            ErrorCategory.EXTERNAL_SERVICE: [
                "ExternalServiceError", "APIError", "ServiceUnavailable",
                "RateLimitExceeded", "QuotaExceeded"
            ],
            ErrorCategory.TIMEOUT: [
                "TimeoutError", "asyncio.TimeoutError", "concurrent.futures.TimeoutError"
            ],
            ErrorCategory.RESOURCE: [
                "ResourceError", "MemoryError", "DiskSpaceError", "ConnectionPoolError"
            ]
        }
        
        self.severity_patterns = {
            ErrorSeverity.CRITICAL: [
                "SystemError", "MemoryError", "DatabaseError", "AuthenticationError"
            ],
            ErrorSeverity.HIGH: [
                "AuthorizationError", "BusinessLogicError", "ExternalServiceError"
            ],
            ErrorSeverity.MEDIUM: [
                "ValidationError", "NetworkError", "TimeoutError"
            ],
            ErrorSeverity.LOW: [
                "Warning", "DeprecationWarning", "UserWarning"
            ]
        }
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an error.
        
        Args:
            error: The exception to classify
            context: Additional context information
            
        Returns:
            Tuple of (category, severity)
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Determine category
        category = self._determine_category(error_type, error_message, context)
        
        # Determine severity
        severity = self._determine_severity(error_type, error_message, context)
        
        return category, severity
    
    def _determine_category(self, error_type: str, error_message: str, context: Dict[str, Any] = None) -> ErrorCategory:
        """Determine error category."""
        error_message_lower = error_message.lower()
        
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.lower() in error_type.lower() or pattern.lower() in error_message_lower:
                    return category
        
        # Default category based on context
        if context:
            if 'network' in context.get('operation', '').lower():
                return ErrorCategory.NETWORK
            elif 'database' in context.get('operation', '').lower():
                return ErrorCategory.DATABASE
            elif 'auth' in context.get('operation', '').lower():
                return ErrorCategory.AUTHENTICATION
        
        return ErrorCategory.SYSTEM
    
    def _determine_severity(self, error_type: str, error_message: str, context: Dict[str, Any] = None) -> ErrorSeverity:
        """Determine error severity."""
        error_message_lower = error_message.lower()
        
        for severity, patterns in self.severity_patterns.items():
            for pattern in patterns:
                if pattern.lower() in error_type.lower() or pattern.lower() in error_message_lower:
                    return severity
        
        # Default severity based on context
        if context:
            if context.get('critical_operation', False):
                return ErrorSeverity.HIGH
            elif context.get('user_facing', False):
                return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.MEDIUM

class RetryMechanism:
    """Retry mechanism with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        """
        Initialize retry mechanism.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            backoff_factor: Exponential backoff factor
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    async def execute_with_retry(self, func: Callable, *args, 
                               retryable_errors: List[Type[Exception]] = None,
                               **kwargs) -> Any:
        """
        Execute function with retry mechanism.
        
        Args:
            func: Function to execute
            *args: Function arguments
            retryable_errors: List of retryable exception types
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        retryable_errors = retryable_errors or [Exception]
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            except tuple(retryable_errors) as e:
                last_exception = e
                
                if attempt < self.max_attempts - 1:
                    delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
                    logger.warning("Retry attempt failed", 
                                 attempt=attempt + 1,
                                 max_attempts=self.max_attempts,
                                 delay=delay,
                                 error=str(e))
                    await asyncio.sleep(delay)
                else:
                    logger.error("All retry attempts failed", 
                               max_attempts=self.max_attempts,
                               error=str(e))
        
        raise last_exception

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that indicates failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self._lock = threading.RLock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError if circuit is open
            Original exception if function fails
        """
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to half-open")
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker reset to closed")
    
    def _on_failure(self):
        """Handle failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning("Circuit breaker opened", 
                             failure_count=self.failure_count,
                             threshold=self.failure_threshold)

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class ErrorRecoveryManager:
    """Error recovery management system."""
    
    def __init__(self, config: ErrorHandlerConfig):
        """
        Initialize error recovery manager.
        
        Args:
            config: Error handler configuration
        """
        self.config = config
        self.recovery_strategies = {}
        self.fallback_handlers = {}
        self._lock = threading.RLock()
    
    def register_recovery_strategy(self, error_category: ErrorCategory, 
                                 strategy: ErrorRecoveryStrategy, 
                                 handler: Callable):
        """Register recovery strategy for error category."""
        with self._lock:
            self.recovery_strategies[error_category] = {
                'strategy': strategy,
                'handler': handler
            }
            logger.info("Recovery strategy registered", 
                       category=error_category.value,
                       strategy=strategy.value)
    
    def register_fallback_handler(self, operation: str, handler: Callable):
        """Register fallback handler for operation."""
        with self._lock:
            self.fallback_handlers[operation] = handler
            logger.info("Fallback handler registered", operation=operation)
    
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> RecoveryAction:
        """
        Handle error with recovery strategy.
        
        Args:
            error: The exception to handle
            context: Error context
            
        Returns:
            Recovery action result
        """
        context = context or {}
        operation = context.get('operation', 'unknown')
        
        # Classify error
        classifier = ErrorClassifier()
        category, severity = classifier.classify_error(error, context)
        
        # Get recovery strategy
        strategy_info = self.recovery_strategies.get(category)
        if not strategy_info:
            strategy_info = {'strategy': ErrorRecoveryStrategy.FAIL_FAST, 'handler': None}
        
        strategy = strategy_info['strategy']
        handler = strategy_info['handler']
        
        # Execute recovery
        start_time = time.time()
        success = False
        result = None
        error_message = None
        
        try:
            if strategy == ErrorRecoveryStrategy.RETRY:
                retry_mechanism = RetryMechanism(
                    max_attempts=self.config.default_retry_attempts,
                    base_delay=self.config.default_retry_delay
                )
                result = await retry_mechanism.execute_with_retry(handler or self._default_retry_handler, error, context)
                success = True
            
            elif strategy == ErrorRecoveryStrategy.FALLBACK:
                fallback_handler = self.fallback_handlers.get(operation)
                if fallback_handler:
                    result = await fallback_handler(error, context)
                    success = True
                else:
                    error_message = "No fallback handler available"
            
            elif strategy == ErrorRecoveryStrategy.CIRCUIT_BREAKER:
                circuit_breaker = CircuitBreaker(
                    failure_threshold=self.config.circuit_breaker_threshold,
                    recovery_timeout=self.config.circuit_breaker_timeout
                )
                result = await circuit_breaker.call(handler or self._default_circuit_breaker_handler, error, context)
                success = True
            
            elif strategy == ErrorRecoveryStrategy.DEGRADE:
                result = await self._degrade_service(error, context)
                success = True
            
            elif strategy == ErrorRecoveryStrategy.FAIL_FAST:
                error_message = "Fail fast strategy - no recovery attempted"
            
            elif strategy == ErrorRecoveryStrategy.IGNORE:
                success = True
                result = None
            
        except Exception as recovery_error:
            error_message = f"Recovery failed: {str(recovery_error)}"
        
        duration = time.time() - start_time
        
        # Create recovery action
        recovery_action = RecoveryAction(
            action_id=str(uuid.uuid4()),
            error_id=context.get('error_id', 'unknown'),
            strategy=strategy,
            timestamp=datetime.now(),
            success=success,
            duration=duration,
            result=result,
            error=error_message,
            metadata={
                'category': category.value,
                'severity': severity.value,
                'operation': operation
            }
        )
        
        logger.info("Error recovery completed", 
                   action_id=recovery_action.action_id,
                   strategy=strategy.value,
                   success=success,
                   duration=duration)
        
        return recovery_action
    
    async def _default_retry_handler(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Default retry handler."""
        # In a real implementation, this would retry the original operation
        raise error
    
    async def _default_circuit_breaker_handler(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Default circuit breaker handler."""
        # In a real implementation, this would execute the original operation
        raise error
    
    async def _degrade_service(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Degrade service functionality."""
        # Return degraded response
        return {
            'status': 'degraded',
            'message': 'Service is operating in degraded mode',
            'original_error': str(error)
        }

class ErrorHandler:
    """Main error handling system."""
    
    def __init__(self, config: ErrorHandlerConfig = None):
        """
        Initialize error handler.
        
        Args:
            config: Error handler configuration
        """
        self.config = config or ErrorHandlerConfig()
        self.error_history = deque(maxlen=self.config.max_error_history)
        self.recovery_manager = ErrorRecoveryManager(self.config)
        self.error_analytics = defaultdict(int)
        self._lock = threading.RLock()
        
        logger.info("Error handler initialized")
    
    def handle_error(self, error: Exception, component: str, operation: str,
                    context: Dict[str, Any] = None, user_id: str = None,
                    session_id: str = None, trace_id: str = None) -> ErrorInfo:
        """
        Handle an error.
        
        Args:
            error: The exception to handle
            component: Component where error occurred
            operation: Operation being performed
            context: Additional context
            user_id: User ID
            session_id: Session ID
            trace_id: Trace ID
            
        Returns:
            Error information
        """
        context = context or {}
        context.update({
            'component': component,
            'operation': operation,
            'user_id': user_id,
            'session_id': session_id,
            'trace_id': trace_id
        })
        
        # Classify error
        classifier = ErrorClassifier()
        category, severity = classifier.classify_error(error, context)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            error_category=category,
            severity=severity,
            component=component,
            operation=operation,
            stack_trace=traceback.format_exc(),
            context=context,
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id
        )
        
        # Store error
        with self._lock:
            self.error_history.append(error_info)
            self.error_analytics[f"{category.value}_{severity.value}"] += 1
        
        # Log error
        logger.error("Error occurred", 
                    error_id=error_info.error_id,
                    error_type=error_info.error_type,
                    error_message=error_info.error_message,
                    category=category.value,
                    severity=severity.value,
                    component=component,
                    operation=operation)
        
        # Attempt recovery if enabled
        if self.config.enable_recovery:
            asyncio.create_task(self._attempt_recovery(error_info))
        
        return error_info
    
    async def _attempt_recovery(self, error_info: ErrorInfo):
        """Attempt error recovery."""
        try:
            recovery_action = await self.recovery_manager.handle_error(
                Exception(error_info.error_message),
                error_info.context
            )
            
            if recovery_action.success:
                logger.info("Error recovery successful", 
                           error_id=error_info.error_id,
                           action_id=recovery_action.action_id)
            else:
                logger.error("Error recovery failed", 
                           error_id=error_info.error_id,
                           action_id=recovery_action.action_id,
                           error=recovery_action.error)
        
        except Exception as e:
            logger.error("Recovery attempt failed", 
                        error_id=error_info.error_id,
                        error=str(e))
    
    def get_error_analytics(self) -> Dict[str, Any]:
        """Get error analytics."""
        with self._lock:
            return {
                'total_errors': len(self.error_history),
                'error_counts': dict(self.error_analytics),
                'recent_errors': [
                    {
                        'error_id': error.error_id,
                        'timestamp': error.timestamp.isoformat(),
                        'error_type': error.error_type,
                        'category': error.error_category.value,
                        'severity': error.severity.value,
                        'component': error.component,
                        'operation': error.operation
                    }
                    for error in list(self.error_history)[-10:]  # Last 10 errors
                ]
            }
    
    def register_recovery_strategy(self, error_category: ErrorCategory, 
                                 strategy: ErrorRecoveryStrategy, 
                                 handler: Callable):
        """Register recovery strategy."""
        self.recovery_manager.register_recovery_strategy(error_category, strategy, handler)
    
    def register_fallback_handler(self, operation: str, handler: Callable):
        """Register fallback handler."""
        self.recovery_manager.register_fallback_handler(operation, handler)

def error_handler(config: ErrorHandlerConfig = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = ErrorHandler(config)
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                # Extract component and operation from function
                component = func.__module__ or 'unknown'
                operation = func.__name__ or 'unknown'
                
                error_info = handler.handle_error(e, component, operation)
                raise e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            handler = ErrorHandler(config)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Extract component and operation from function
                component = func.__module__ or 'unknown'
                operation = func.__name__ or 'unknown'
                
                error_info = handler.handle_error(e, component, operation)
                raise e
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def create_error_handler(config: ErrorHandlerConfig = None) -> ErrorHandler:
    """
    Create an error handler.
    
    Args:
        config: Error handler configuration
        
    Returns:
        ErrorHandler instance
    """
    return ErrorHandler(config)

if __name__ == "__main__":
    # Demo of error handler
    config = ErrorHandlerConfig(
        enable_error_tracking=True,
        enable_recovery=True,
        enable_circuit_breaker=True,
        enable_retry=True,
        max_error_history=10000,
        error_retention_days=30,
        default_retry_attempts=3,
        default_retry_delay=1.0,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=60.0,
        enable_error_analytics=True,
        error_reporting_interval=300.0
    )
    
    error_handler_instance = create_error_handler(config)
    
    # Register recovery strategies
    async def network_recovery_handler(error: Exception, context: Dict[str, Any]):
        return {"status": "recovered", "message": "Network error recovered"}
    
    error_handler_instance.register_recovery_strategy(
        ErrorCategory.NETWORK,
        ErrorRecoveryStrategy.RETRY,
        network_recovery_handler
    )
    
    # Register fallback handler
    async def fallback_handler(error: Exception, context: Dict[str, Any]):
        return {"status": "fallback", "message": "Using fallback response"}
    
    error_handler_instance.register_fallback_handler("api_call", fallback_handler)
    
    print("Error Handler created successfully!")
    
    # Test error handling
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_info = error_handler_instance.handle_error(
            e, "demo", "test_operation", 
            context={'test': True}, 
            user_id="user_123"
        )
        print(f"Error handled: {error_info.error_id}")
    
    # Get analytics
    analytics = error_handler_instance.get_error_analytics()
    print(f"Error analytics: {analytics}")
    
    # Test decorator
    @error_handler(config)
    async def test_function():
        raise RuntimeError("Decorator test error")
    
    # Note: In a real application, this would be called in an async context
    print("Error Handler demo completed")