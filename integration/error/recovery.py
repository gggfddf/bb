"""
Error Recovery System for Trading System

This module provides comprehensive error recovery mechanisms for the trading system,
including automatic recovery strategies, fallback mechanisms, and recovery policies.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import traceback
import hashlib

logger = logging.getLogger(__name__)

class RecoveryStrategy(Enum):
    """Recovery strategy enumeration"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    DEGRADED_MODE = "degraded_mode"
    RESTART = "restart"
    MANUAL = "manual"

class RecoveryStatus(Enum):
    """Recovery status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class RecoveryAction:
    """Recovery action structure"""
    action_id: str
    error_id: str
    strategy: RecoveryStrategy
    component: str
    description: str
    parameters: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    status: RecoveryStatus
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]

@dataclass
class RecoveryPolicy:
    """Recovery policy structure"""
    policy_id: str
    name: str
    component: str
    error_types: List[str]
    strategies: List[RecoveryStrategy]
    max_attempts: int
    timeout: int  # seconds
    backoff_factor: float
    fallback_config: Optional[Dict[str, Any]]
    enabled: bool
    priority: int

class ErrorRecovery:
    """
    Error recovery system for handling and recovering from various types of errors.
    """
    
    def __init__(self):
        self.recovery_policies: Dict[str, RecoveryPolicy] = {}
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.recovery_handlers: Dict[str, Callable] = {}
        
        # Recovery execution
        self.recovery_queue = asyncio.Queue()
        self.recovery_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'recovery_time_avg': 0.0
        }
        
        # Register default recovery handlers
        self._register_default_handlers()
        
        # Start recovery processor
        self.start()
        
        logger.info("Error recovery system initialized")
    
    def start(self):
        """Start the recovery system"""
        if not self.running:
            self.running = True
            self.recovery_thread = threading.Thread(target=self._recovery_processor, daemon=True)
            self.recovery_thread.start()
            logger.info("Error recovery system started")
    
    def stop(self):
        """Stop the recovery system"""
        if self.running:
            self.running = False
            if self.recovery_thread:
                self.recovery_thread.join(timeout=5)
            logger.info("Error recovery system stopped")
    
    def register_recovery_policy(self, policy: RecoveryPolicy) -> bool:
        """
        Register a recovery policy.
        
        Args:
            policy: Recovery policy to register
            
        Returns:
            True if policy was registered successfully
        """
        try:
            self.recovery_policies[policy.policy_id] = policy
            logger.info(f"Recovery policy registered: {policy.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register recovery policy: {e}")
            return False
    
    def register_recovery_handler(self, strategy: RecoveryStrategy, handler: Callable) -> bool:
        """
        Register a recovery handler for a specific strategy.
        
        Args:
            strategy: Recovery strategy
            handler: Recovery handler function
            
        Returns:
            True if handler was registered successfully
        """
        try:
            self.recovery_handlers[strategy.value] = handler
            logger.info(f"Recovery handler registered for strategy: {strategy.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register recovery handler: {e}")
            return False
    
    def trigger_recovery(self, error_id: str, component: str, error_type: str,
                        error_details: Dict[str, Any]) -> Optional[str]:
        """
        Trigger error recovery for a specific error.
        
        Args:
            error_id: Error ID
            component: Component name
            error_type: Type of error
            error_details: Error details
            
        Returns:
            Recovery action ID if recovery was triggered
        """
        try:
            # Find applicable recovery policy
            policy = self._find_applicable_policy(component, error_type)
            if not policy:
                logger.warning(f"No recovery policy found for component {component}, error type {error_type}")
                return None
            
            # Create recovery action
            action_id = self._generate_action_id()
            
            # Select recovery strategy
            strategy = self._select_recovery_strategy(policy, error_details)
            
            recovery_action = RecoveryAction(
                action_id=action_id,
                error_id=error_id,
                strategy=strategy,
                component=component,
                description=f"Recovery for {error_type} in {component}",
                parameters=self._build_recovery_parameters(policy, strategy, error_details),
                created_at=datetime.utcnow(),
                started_at=None,
                completed_at=None,
                status=RecoveryStatus.PENDING,
                result=None,
                error_message=None
            )
            
            self.recovery_actions[action_id] = recovery_action
            
            # Queue recovery action
            asyncio.run_coroutine_threadsafe(
                self.recovery_queue.put(recovery_action), 
                asyncio.get_event_loop()
            )
            
            logger.info(f"Recovery triggered: {action_id} for error {error_id}")
            return action_id
            
        except Exception as e:
            logger.error(f"Failed to trigger recovery: {e}")
            return None
    
    def get_recovery_status(self, action_id: str) -> Optional[RecoveryAction]:
        """
        Get recovery action status.
        
        Args:
            action_id: Recovery action ID
            
        Returns:
            Recovery action if found
        """
        return self.recovery_actions.get(action_id)
    
    def cancel_recovery(self, action_id: str) -> bool:
        """
        Cancel a recovery action.
        
        Args:
            action_id: Recovery action ID
            
        Returns:
            True if recovery was cancelled
        """
        try:
            if action_id in self.recovery_actions:
                action = self.recovery_actions[action_id]
                if action.status == RecoveryStatus.PENDING:
                    action.status = RecoveryStatus.CANCELLED
                    action.completed_at = datetime.utcnow()
                    logger.info(f"Recovery cancelled: {action_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel recovery: {e}")
            return False
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        Get recovery statistics.
        
        Returns:
            Recovery statistics
        """
        try:
            total_actions = len(self.recovery_actions)
            successful_actions = len([a for a in self.recovery_actions.values() 
                                   if a.status == RecoveryStatus.SUCCESSFUL])
            failed_actions = len([a for a in self.recovery_actions.values() 
                                if a.status == RecoveryStatus.FAILED])
            
            # Calculate average recovery time
            completed_actions = [a for a in self.recovery_actions.values() 
                               if a.completed_at and a.started_at]
            
            if completed_actions:
                avg_recovery_time = sum([
                    (a.completed_at - a.started_at).total_seconds() 
                    for a in completed_actions
                ]) / len(completed_actions)
            else:
                avg_recovery_time = 0.0
            
            # Strategy distribution
            strategy_counts = {}
            for action in self.recovery_actions.values():
                strategy = action.strategy.value
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            return {
                'total_actions': total_actions,
                'successful_actions': successful_actions,
                'failed_actions': failed_actions,
                'pending_actions': len([a for a in self.recovery_actions.values() 
                                     if a.status == RecoveryStatus.PENDING]),
                'in_progress_actions': len([a for a in self.recovery_actions.values() 
                                          if a.status == RecoveryStatus.IN_PROGRESS]),
                'avg_recovery_time': avg_recovery_time,
                'strategy_distribution': strategy_counts,
                'active_policies': len([p for p in self.recovery_policies.values() if p.enabled])
            }
            
        except Exception as e:
            logger.error(f"Failed to get recovery statistics: {e}")
            return {}
    
    def _recovery_processor(self):
        """Recovery action processor"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self.running:
                try:
                    # Process recovery actions
                    action = loop.run_until_complete(
                        asyncio.wait_for(self.recovery_queue.get(), timeout=1.0)
                    )
                    
                    if action:
                        self._execute_recovery_action(action)
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in recovery processor: {e}")
                    time.sleep(1)
                    
        finally:
            loop.close()
    
    def _execute_recovery_action(self, action: RecoveryAction):
        """Execute a recovery action"""
        try:
            action.started_at = datetime.utcnow()
            action.status = RecoveryStatus.IN_PROGRESS
            
            logger.info(f"Executing recovery action: {action.action_id}")
            
            # Get recovery handler
            handler = self.recovery_handlers.get(action.strategy.value)
            if not handler:
                raise ValueError(f"No handler found for strategy: {action.strategy.value}")
            
            # Execute recovery
            result = handler(action.component, action.parameters, action.error_id)
            
            action.completed_at = datetime.utcnow()
            action.result = result
            action.status = RecoveryStatus.SUCCESSFUL
            
            self.stats['successful_recoveries'] += 1
            logger.info(f"Recovery action completed successfully: {action.action_id}")
            
        except Exception as e:
            action.completed_at = datetime.utcnow()
            action.status = RecoveryStatus.FAILED
            action.error_message = str(e)
            
            self.stats['failed_recoveries'] += 1
            logger.error(f"Recovery action failed: {action.action_id}, error: {e}")
    
    def _find_applicable_policy(self, component: str, error_type: str) -> Optional[RecoveryPolicy]:
        """Find applicable recovery policy"""
        applicable_policies = []
        
        for policy in self.recovery_policies.values():
            if not policy.enabled:
                continue
            
            if policy.component == component and error_type in policy.error_types:
                applicable_policies.append(policy)
        
        if not applicable_policies:
            return None
        
        # Return policy with highest priority
        return max(applicable_policies, key=lambda p: p.priority)
    
    def _select_recovery_strategy(self, policy: RecoveryPolicy, error_details: Dict[str, Any]) -> RecoveryStrategy:
        """Select recovery strategy based on policy and error details"""
        # Simple strategy selection - can be enhanced with ML or rules engine
        if RecoveryStrategy.RETRY in policy.strategies:
            return RecoveryStrategy.RETRY
        elif RecoveryStrategy.FALLBACK in policy.strategies:
            return RecoveryStrategy.FALLBACK
        elif RecoveryStrategy.CIRCUIT_BREAKER in policy.strategies:
            return RecoveryStrategy.CIRCUIT_BREAKER
        elif RecoveryStrategy.DEGRADED_MODE in policy.strategies:
            return RecoveryStrategy.DEGRADED_MODE
        else:
            return RecoveryStrategy.MANUAL
    
    def _build_recovery_parameters(self, policy: RecoveryPolicy, strategy: RecoveryStrategy,
                                 error_details: Dict[str, Any]) -> Dict[str, Any]:
        """Build recovery parameters"""
        params = {
            'max_attempts': policy.max_attempts,
            'timeout': policy.timeout,
            'backoff_factor': policy.backoff_factor
        }
        
        if strategy == RecoveryStrategy.FALLBACK and policy.fallback_config:
            params.update(policy.fallback_config)
        
        # Add error-specific parameters
        params['error_details'] = error_details
        
        return params
    
    def _register_default_handlers(self):
        """Register default recovery handlers"""
        
        def retry_handler(component: str, parameters: Dict[str, Any], error_id: str) -> Dict[str, Any]:
            """Retry recovery handler"""
            max_attempts = parameters.get('max_attempts', 3)
            timeout = parameters.get('timeout', 30)
            backoff_factor = parameters.get('backoff_factor', 2.0)
            
            for attempt in range(max_attempts):
                try:
                    logger.info(f"Retry attempt {attempt + 1} for component {component}")
                    
                    # Simulate retry logic - replace with actual retry implementation
                    time.sleep(1)  # Simulate work
                    
                    return {
                        'attempt': attempt + 1,
                        'success': True,
                        'recovery_time': time.time()
                    }
                    
                except Exception as e:
                    if attempt < max_attempts - 1:
                        sleep_time = backoff_factor ** attempt
                        time.sleep(sleep_time)
                    else:
                        raise e
        
        def fallback_handler(component: str, parameters: Dict[str, Any], error_id: str) -> Dict[str, Any]:
            """Fallback recovery handler"""
            fallback_config = parameters.get('fallback_config', {})
            
            logger.info(f"Executing fallback for component {component}")
            
            # Simulate fallback logic - replace with actual fallback implementation
            time.sleep(1)  # Simulate work
            
            return {
                'fallback_executed': True,
                'fallback_config': fallback_config,
                'recovery_time': time.time()
            }
        
        def circuit_breaker_handler(component: str, parameters: Dict[str, Any], error_id: str) -> Dict[str, Any]:
            """Circuit breaker recovery handler"""
            logger.info(f"Executing circuit breaker for component {component}")
            
            # Simulate circuit breaker logic - replace with actual implementation
            time.sleep(1)  # Simulate work
            
            return {
                'circuit_breaker_executed': True,
                'recovery_time': time.time()
            }
        
        def degraded_mode_handler(component: str, parameters: Dict[str, Any], error_id: str) -> Dict[str, Any]:
            """Degraded mode recovery handler"""
            logger.info(f"Executing degraded mode for component {component}")
            
            # Simulate degraded mode logic - replace with actual implementation
            time.sleep(1)  # Simulate work
            
            return {
                'degraded_mode_executed': True,
                'recovery_time': time.time()
            }
        
        def restart_handler(component: str, parameters: Dict[str, Any], error_id: str) -> Dict[str, Any]:
            """Restart recovery handler"""
            logger.info(f"Executing restart for component {component}")
            
            # Simulate restart logic - replace with actual restart implementation
            time.sleep(2)  # Simulate restart time
            
            return {
                'restart_executed': True,
                'recovery_time': time.time()
            }
        
        # Register handlers
        self.register_recovery_handler(RecoveryStrategy.RETRY, retry_handler)
        self.register_recovery_handler(RecoveryStrategy.FALLBACK, fallback_handler)
        self.register_recovery_handler(RecoveryStrategy.CIRCUIT_BREAKER, circuit_breaker_handler)
        self.register_recovery_handler(RecoveryStrategy.DEGRADED_MODE, degraded_mode_handler)
        self.register_recovery_handler(RecoveryStrategy.RESTART, restart_handler)
    
    def _generate_action_id(self) -> str:
        """Generate recovery action ID"""
        return hashlib.md5(f"recovery_{datetime.utcnow().isoformat()}".encode()).hexdigest()


# Global instance management
_error_recovery_instance = None

def get_error_recovery() -> ErrorRecovery:
    """Get or create error recovery instance"""
    global _error_recovery_instance
    
    if _error_recovery_instance is None:
        _error_recovery_instance = ErrorRecovery()
    
    return _error_recovery_instance


def init_error_recovery() -> ErrorRecovery:
    """Initialize error recovery system"""
    global _error_recovery_instance
    
    if _error_recovery_instance:
        _error_recovery_instance.stop()
    
    _error_recovery_instance = ErrorRecovery()
    
    return _error_recovery_instance