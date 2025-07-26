"""
Component Lifecycle Manager for System Orchestration

This module provides component lifecycle management functionality for the trading system,
handling the startup, shutdown, and state transitions of system components.
"""

import logging
import threading
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import signal
import sys

from .service_registry import ServiceRegistry, ServiceStatus, ServiceType, ServiceInfo

logger = logging.getLogger(__name__)

class ComponentState(Enum):
    """Component state enumeration"""
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    RESTARTING = "restarting"

class ComponentPriority(Enum):
    """Component priority enumeration"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class ComponentConfig:
    """Component configuration"""
    name: str
    component_type: str
    priority: ComponentPriority
    dependencies: List[str]
    startup_timeout: int = 30  # seconds
    shutdown_timeout: int = 30  # seconds
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    restart_delay: int = 5  # seconds
    health_check_interval: int = 10  # seconds
    auto_start: bool = True

@dataclass
class ComponentInfo:
    """Component information"""
    config: ComponentConfig
    state: ComponentState
    service_id: Optional[str]
    start_time: Optional[datetime]
    stop_time: Optional[datetime]
    restart_count: int
    last_health_check: Optional[datetime]
    error_message: Optional[str]
    metadata: Dict[str, Any]

class LifecycleManager:
    """
    Component lifecycle manager for managing component startup, shutdown, and state transitions.
    """
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.components: Dict[str, ComponentInfo] = {}
        self.component_handlers: Dict[str, Callable] = {}
        self.state_callbacks: Dict[ComponentState, List[Callable]] = {}
        
        # Lifecycle management
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._health_check_thread = None
        self._running = False
        
        # Statistics
        self.startup_time = None
        self.shutdown_time = None
        self.total_restarts = 0
        
        # Signal handling
        self._setup_signal_handlers()
        
        logger.info("Lifecycle manager initialized")
    
    def register_component(self, config: ComponentConfig, 
                          startup_handler: Optional[Callable] = None,
                          shutdown_handler: Optional[Callable] = None,
                          health_check_handler: Optional[Callable] = None) -> bool:
        """
        Register a component with the lifecycle manager.
        
        Args:
            config: Component configuration
            startup_handler: Optional startup handler function
            shutdown_handler: Optional shutdown handler function
            health_check_handler: Optional health check handler function
            
        Returns:
            True if component was registered successfully
        """
        with self._lock:
            if config.name in self.components:
                logger.warning(f"Component {config.name} already registered")
                return False
            
            # Create component info
            component_info = ComponentInfo(
                config=config,
                state=ComponentState.INITIALIZED,
                service_id=None,
                start_time=None,
                stop_time=None,
                restart_count=0,
                last_health_check=None,
                error_message=None,
                metadata={}
            )
            
            # Register component
            self.components[config.name] = component_info
            
            # Register handlers
            if startup_handler:
                self.component_handlers[f"{config.name}_startup"] = startup_handler
            if shutdown_handler:
                self.component_handlers[f"{config.name}_shutdown"] = shutdown_handler
            if health_check_handler:
                self.component_handlers[f"{config.name}_health"] = health_check_handler
            
            # Update dependency graph
            self.dependency_graph[config.name] = config.dependencies.copy()
            
            logger.info(f"Component registered: {config.name} (priority: {config.priority.value})")
            return True
    
    def unregister_component(self, component_name: str) -> bool:
        """
        Unregister a component from the lifecycle manager.
        
        Args:
            component_name: Name of the component to unregister
            
        Returns:
            True if component was unregistered successfully
        """
        with self._lock:
            if component_name not in self.components:
                return False
            
            # Stop component if running
            if self.components[component_name].state in [ComponentState.RUNNING, ComponentState.STARTING]:
                self.stop_component(component_name)
            
            # Remove component
            del self.components[component_name]
            
            # Remove handlers
            for handler_key in list(self.component_handlers.keys()):
                if handler_key.startswith(f"{component_name}_"):
                    del self.component_handlers[handler_key]
            
            # Update dependency graph
            if component_name in self.dependency_graph:
                del self.dependency_graph[component_name]
            
            logger.info(f"Component unregistered: {component_name}")
            return True
    
    def start_component(self, component_name: str) -> bool:
        """
        Start a specific component.
        
        Args:
            component_name: Name of the component to start
            
        Returns:
            True if component was started successfully
        """
        with self._lock:
            if component_name not in self.components:
                logger.error(f"Component {component_name} not found")
                return False
            
            component_info = self.components[component_name]
            
            # Check if already running
            if component_info.state in [ComponentState.RUNNING, ComponentState.STARTING]:
                logger.warning(f"Component {component_name} is already {component_info.state.value}")
                return True
            
            # Check dependencies
            if not self._check_dependencies(component_name):
                logger.error(f"Component {component_name} dependencies not met")
                return False
            
            # Start component
            return self._start_component_internal(component_name)
    
    def stop_component(self, component_name: str) -> bool:
        """
        Stop a specific component.
        
        Args:
            component_name: Name of the component to stop
            
        Returns:
            True if component was stopped successfully
        """
        with self._lock:
            if component_name not in self.components:
                logger.error(f"Component {component_name} not found")
                return False
            
            component_info = self.components[component_name]
            
            # Check if already stopped
            if component_info.state in [ComponentState.STOPPED, ComponentState.STOPPING]:
                logger.warning(f"Component {component_name} is already {component_info.state.value}")
                return True
            
            # Stop component
            return self._stop_component_internal(component_name)
    
    def restart_component(self, component_name: str) -> bool:
        """
        Restart a specific component.
        
        Args:
            component_name: Name of the component to restart
            
        Returns:
            True if component was restarted successfully
        """
        with self._lock:
            if component_name not in self.components:
                logger.error(f"Component {component_name} not found")
                return False
            
            logger.info(f"Restarting component: {component_name}")
            
            # Stop component
            if not self.stop_component(component_name):
                return False
            
            # Wait for stop to complete
            timeout = self.components[component_name].config.shutdown_timeout
            start_time = time.time()
            while (self.components[component_name].state != ComponentState.STOPPED and 
                   time.time() - start_time < timeout):
                time.sleep(0.1)
            
            # Start component
            return self.start_component(component_name)
    
    def start_all_components(self) -> Dict[str, bool]:
        """
        Start all components in dependency order.
        
        Returns:
            Dictionary mapping component names to success status
        """
        with self._lock:
            self.startup_time = datetime.utcnow()
            logger.info("Starting all components...")
            
            # Calculate startup order
            self.startup_order = self._calculate_startup_order()
            
            results = {}
            for component_name in self.startup_order:
                component_info = self.components[component_name]
                
                if not component_info.config.auto_start:
                    logger.info(f"Skipping auto-start for component: {component_name}")
                    results[component_name] = True
                    continue
                
                success = self._start_component_internal(component_name)
                results[component_name] = success
                
                if not success:
                    logger.error(f"Failed to start component: {component_name}")
                    # Continue with other components
            
            logger.info("Component startup completed")
            return results
    
    def stop_all_components(self) -> Dict[str, bool]:
        """
        Stop all components in reverse dependency order.
        
        Returns:
            Dictionary mapping component names to success status
        """
        with self._lock:
            self.shutdown_time = datetime.utcnow()
            logger.info("Stopping all components...")
            
            # Calculate shutdown order
            self.shutdown_order = self._calculate_shutdown_order()
            
            results = {}
            for component_name in self.shutdown_order:
                success = self._stop_component_internal(component_name)
                results[component_name] = success
                
                if not success:
                    logger.error(f"Failed to stop component: {component_name}")
                    # Continue with other components
            
            logger.info("Component shutdown completed")
            return results
    
    def get_component_state(self, component_name: str) -> Optional[ComponentState]:
        """
        Get the current state of a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component state or None if not found
        """
        with self._lock:
            if component_name not in self.components:
                return None
            return self.components[component_name].state
    
    def get_component_info(self, component_name: str) -> Optional[ComponentInfo]:
        """
        Get detailed information about a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component information or None if not found
        """
        with self._lock:
            return self.components.get(component_name)
    
    def get_all_components(self) -> Dict[str, ComponentInfo]:
        """
        Get information about all components.
        
        Returns:
            Dictionary mapping component names to component information
        """
        with self._lock:
            return self.components.copy()
    
    def add_state_callback(self, state: ComponentState, callback: Callable):
        """
        Add a callback for state changes.
        
        Args:
            state: Component state to monitor
            callback: Callback function
        """
        if state not in self.state_callbacks:
            self.state_callbacks[state] = []
        self.state_callbacks[state].append(callback)
    
    def start_health_monitoring(self):
        """Start health monitoring for all components"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._health_check_thread = threading.Thread(target=self._health_check_worker, daemon=True)
            self._health_check_thread.start()
            logger.info("Health monitoring started")
    
    def stop_health_monitoring(self):
        """Stop health monitoring"""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._health_check_thread:
                self._health_check_thread.join(timeout=5)
            logger.info("Health monitoring stopped")
    
    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """
        Get lifecycle management statistics.
        
        Returns:
            Lifecycle statistics
        """
        with self._lock:
            total_components = len(self.components)
            state_counts = {}
            running_components = 0
            
            for component_info in self.components.values():
                state = component_info.state.value
                state_counts[state] = state_counts.get(state, 0) + 1
                if component_info.state == ComponentState.RUNNING:
                    running_components += 1
            
            return {
                'total_components': total_components,
                'running_components': running_components,
                'state_counts': state_counts,
                'total_restarts': self.total_restarts,
                'startup_time': self.startup_time.isoformat() if self.startup_time else None,
                'shutdown_time': self.shutdown_time.isoformat() if self.shutdown_time else None,
                'uptime': (datetime.utcnow() - self.startup_time).total_seconds() if self.startup_time else 0
            }
    
    def _start_component_internal(self, component_name: str) -> bool:
        """Internal method to start a component"""
        component_info = self.components[component_name]
        
        try:
            # Update state
            component_info.state = ComponentState.STARTING
            component_info.start_time = datetime.utcnow()
            component_info.error_message = None
            
            # Call startup handler
            startup_handler = self.component_handlers.get(f"{component_name}_startup")
            if startup_handler:
                startup_handler()
            
            # Register with service registry if applicable
            if hasattr(component_info, 'service_type'):
                service_id = self.service_registry.register_service(
                    service_name=component_info.config.name,
                    service_type=component_info.service_type,
                    host="localhost",  # Default host
                    port=0,  # Default port
                    version="1.0.0"
                )
                component_info.service_id = service_id
            
            # Update state
            component_info.state = ComponentState.RUNNING
            component_info.last_health_check = datetime.utcnow()
            
            # Trigger callbacks
            self._trigger_state_callbacks(component_name, ComponentState.RUNNING)
            
            logger.info(f"Component started: {component_name}")
            return True
            
        except Exception as e:
            component_info.state = ComponentState.ERROR
            component_info.error_message = str(e)
            component_info.stop_time = datetime.utcnow()
            
            # Trigger callbacks
            self._trigger_state_callbacks(component_name, ComponentState.ERROR)
            
            logger.error(f"Failed to start component {component_name}: {e}")
            return False
    
    def _stop_component_internal(self, component_name: str) -> bool:
        """Internal method to stop a component"""
        component_info = self.components[component_name]
        
        try:
            # Update state
            component_info.state = ComponentState.STOPPING
            
            # Call shutdown handler
            shutdown_handler = self.component_handlers.get(f"{component_name}_shutdown")
            if shutdown_handler:
                shutdown_handler()
            
            # Unregister from service registry if applicable
            if component_info.service_id:
                self.service_registry.unregister_service(component_info.service_id)
                component_info.service_id = None
            
            # Update state
            component_info.state = ComponentState.STOPPED
            component_info.stop_time = datetime.utcnow()
            
            # Trigger callbacks
            self._trigger_state_callbacks(component_name, ComponentState.STOPPED)
            
            logger.info(f"Component stopped: {component_name}")
            return True
            
        except Exception as e:
            component_info.state = ComponentState.ERROR
            component_info.error_message = str(e)
            
            # Trigger callbacks
            self._trigger_state_callbacks(component_name, ComponentState.ERROR)
            
            logger.error(f"Failed to stop component {component_name}: {e}")
            return False
    
    def _check_dependencies(self, component_name: str) -> bool:
        """Check if component dependencies are met"""
        dependencies = self.dependency_graph.get(component_name, [])
        
        for dep in dependencies:
            if dep not in self.components:
                logger.error(f"Dependency {dep} not found for component {component_name}")
                return False
            
            dep_state = self.components[dep].state
            if dep_state != ComponentState.RUNNING:
                logger.error(f"Dependency {dep} is not running (state: {dep_state.value}) for component {component_name}")
                return False
        
        return True
    
    def _calculate_startup_order(self) -> List[str]:
        """Calculate the order in which components should be started"""
        # Sort by priority first
        sorted_components = sorted(
            self.components.keys(),
            key=lambda name: self.components[name].config.priority.value
        )
        
        # Then ensure dependencies are started first
        startup_order = []
        visited = set()
        
        def visit(component_name):
            if component_name in visited:
                return
            
            visited.add(component_name)
            
            # Visit dependencies first
            for dep in self.dependency_graph.get(component_name, []):
                if dep in self.components:
                    visit(dep)
            
            startup_order.append(component_name)
        
        for component_name in sorted_components:
            visit(component_name)
        
        return startup_order
    
    def _calculate_shutdown_order(self) -> List[str]:
        """Calculate the order in which components should be stopped"""
        # Reverse the startup order
        return list(reversed(self.startup_order))
    
    def _health_check_worker(self):
        """Background worker for health checks"""
        while self._running:
            try:
                with self._lock:
                    for component_name, component_info in self.components.items():
                        if component_info.state == ComponentState.RUNNING:
                            self._perform_health_check(component_name)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in health check worker: {e}")
    
    def _perform_health_check(self, component_name: str):
        """Perform health check for a component"""
        component_info = self.components[component_name]
        
        try:
            # Call health check handler
            health_handler = self.component_handlers.get(f"{component_name}_health")
            if health_handler:
                health_result = health_handler()
                
                if not health_result:
                    logger.warning(f"Health check failed for component: {component_name}")
                    
                    # Restart component if configured
                    if (component_info.config.restart_on_failure and 
                        component_info.restart_count < component_info.config.max_restart_attempts):
                        self._schedule_restart(component_name)
            
            component_info.last_health_check = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error in health check for component {component_name}: {e}")
    
    def _schedule_restart(self, component_name: str):
        """Schedule a component restart"""
        component_info = self.components[component_name]
        component_info.restart_count += 1
        self.total_restarts += 1
        
        logger.info(f"Scheduling restart for component {component_name} (attempt {component_info.restart_count})")
        
        # Schedule restart in a separate thread
        def delayed_restart():
            time.sleep(component_info.config.restart_delay)
            self.restart_component(component_name)
        
        restart_thread = threading.Thread(target=delayed_restart, daemon=True)
        restart_thread.start()
    
    def _trigger_state_callbacks(self, component_name: str, state: ComponentState):
        """Trigger state change callbacks"""
        for callback in self.state_callbacks.get(state, []):
            try:
                callback(component_name, state)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.stop_all_components()
            self.stop_health_monitoring()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


# Global instance management
_lifecycle_manager_instance = None

def get_lifecycle_manager(service_registry: ServiceRegistry = None) -> LifecycleManager:
    """Get or create lifecycle manager instance"""
    global _lifecycle_manager_instance
    
    if _lifecycle_manager_instance is None:
        if service_registry is None:
            from .service_registry import get_service_registry
            service_registry = get_service_registry()
        
        _lifecycle_manager_instance = LifecycleManager(service_registry)
    
    return _lifecycle_manager_instance


def init_lifecycle_manager(service_registry: ServiceRegistry) -> LifecycleManager:
    """Initialize lifecycle manager with custom service registry"""
    global _lifecycle_manager_instance
    
    if _lifecycle_manager_instance:
        _lifecycle_manager_instance.stop_health_monitoring()
    
    _lifecycle_manager_instance = LifecycleManager(service_registry)
    
    return _lifecycle_manager_instance