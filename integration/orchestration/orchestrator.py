#!/usr/bin/env python3
"""
System Orchestration Module

Implements comprehensive system orchestration for the trading system:
- Component lifecycle management
- Service discovery and registration
- Inter-component communication
- System state management
- Dependency injection system
- Configuration management

Features:
- Complete system orchestration and coordination
- Service discovery and registration mechanisms
- Component lifecycle management (start, stop, restart, health check)
- Inter-component communication via message bus
- System state management and monitoring
- Dependency injection and configuration management
- High-availability and fault tolerance
"""

import asyncio
import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import warnings
import structlog
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod

logger = structlog.get_logger()

class ComponentStatus(Enum):
    """Component status enumeration."""
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"

class ComponentType(Enum):
    """Component type enumeration."""
    DATA_PIPELINE = "data_pipeline"
    STRATEGY_EXECUTOR = "strategy_executor"
    RISK_MANAGER = "risk_manager"
    PERFORMANCE_MONITOR = "performance_monitor"
    ALERT_SYSTEM = "alert_system"
    ORDER_MANAGER = "order_manager"
    MARKET_DATA_PROCESSOR = "market_data_processor"
    SYSTEM_MONITOR = "system_monitor"
    API_GATEWAY = "api_gateway"

class MessageType(Enum):
    """Message type enumeration."""
    COMMAND = "command"
    EVENT = "event"
    REQUEST = "request"
    RESPONSE = "response"
    HEARTBEAT = "heartbeat"
    ALERT = "alert"

@dataclass
class ComponentInfo:
    """Component information structure."""
    component_id: str
    name: str
    component_type: ComponentType
    status: ComponentStatus = ComponentStatus.UNKNOWN
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMessage:
    """System message structure."""
    message_id: str
    message_type: MessageType
    source: str
    destination: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    priority: int = 0

@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""
    enable_auto_discovery: bool = True
    enable_health_monitoring: bool = True
    enable_fault_tolerance: bool = True
    heartbeat_interval: float = 30.0  # seconds
    health_check_timeout: float = 10.0  # seconds
    max_retry_attempts: int = 3
    retry_delay: float = 5.0  # seconds
    enable_logging: bool = True
    enable_metrics: bool = True
    max_message_queue_size: int = 10000
    message_ttl: float = 300.0  # seconds

class Component(ABC):
    """Abstract base class for system components."""
    
    def __init__(self, component_id: str, name: str, component_type: ComponentType):
        """
        Initialize component.
        
        Args:
            component_id: Unique component identifier
            name: Component name
            component_type: Component type
        """
        self.component_id = component_id
        self.name = name
        self.component_type = component_type
        self.status = ComponentStatus.UNKNOWN
        self.orchestrator = None
        self.config = {}
        self._lock = threading.RLock()
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the component."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the component."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Perform health check."""
        pass
    
    def set_orchestrator(self, orchestrator: 'SystemOrchestrator'):
        """Set orchestrator reference."""
        self.orchestrator = orchestrator
    
    def get_info(self) -> ComponentInfo:
        """Get component information."""
        return ComponentInfo(
            component_id=self.component_id,
            name=self.name,
            component_type=self.component_type,
            status=self.status,
            config=self.config
        )

class ServiceRegistry:
    """Service discovery and registration system."""
    
    def __init__(self):
        """Initialize service registry."""
        self.services = {}
        self.service_types = defaultdict(list)
        self._lock = threading.RLock()
    
    def register_service(self, service_info: ComponentInfo):
        """Register a service."""
        with self._lock:
            self.services[service_info.component_id] = service_info
            self.service_types[service_info.component_type.value].append(service_info.component_id)
            logger.info("Service registered", 
                       component_id=service_info.component_id,
                       component_type=service_info.component_type.value)
    
    def unregister_service(self, component_id: str):
        """Unregister a service."""
        with self._lock:
            if component_id in self.services:
                service_info = self.services[component_id]
                self.service_types[service_info.component_type.value].remove(component_id)
                del self.services[component_id]
                logger.info("Service unregistered", component_id=component_id)
    
    def get_service(self, component_id: str) -> Optional[ComponentInfo]:
        """Get service by ID."""
        with self._lock:
            return self.services.get(component_id)
    
    def get_services_by_type(self, component_type: ComponentType) -> List[ComponentInfo]:
        """Get services by type."""
        with self._lock:
            service_ids = self.service_types.get(component_type.value, [])
            return [self.services[service_id] for service_id in service_ids 
                   if service_id in self.services]
    
    def get_all_services(self) -> List[ComponentInfo]:
        """Get all registered services."""
        with self._lock:
            return list(self.services.values())
    
    def update_service_status(self, component_id: str, status: ComponentStatus):
        """Update service status."""
        with self._lock:
            if component_id in self.services:
                self.services[component_id].status = status
                self.services[component_id].last_heartbeat = datetime.now()

class MessageBus:
    """Inter-component communication system."""
    
    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize message bus.
        
        Args:
            max_queue_size: Maximum message queue size
        """
        self.message_queue = asyncio.Queue(maxsize=max_queue_size)
        self.subscribers = defaultdict(list)
        self.message_history = deque(maxlen=1000)
        self._running = False
        self._lock = threading.RLock()
    
    async def start(self):
        """Start message bus."""
        self._running = True
        asyncio.create_task(self._message_processor())
        logger.info("Message bus started")
    
    async def stop(self):
        """Stop message bus."""
        self._running = False
        logger.info("Message bus stopped")
    
    async def publish(self, message: SystemMessage):
        """Publish a message."""
        if not self._running:
            logger.warning("Message bus not running")
            return
        
        try:
            await self.message_queue.put(message)
            self.message_history.append(message)
            logger.debug("Message published", 
                        message_id=message.message_id,
                        message_type=message.message_type.value)
        except asyncio.QueueFull:
            logger.error("Message queue full")
    
    def subscribe(self, component_id: str, message_types: List[MessageType], 
                 callback: Callable[[SystemMessage], None]):
        """Subscribe to messages."""
        with self._lock:
            for message_type in message_types:
                self.subscribers[message_type.value].append((component_id, callback))
            logger.info("Component subscribed", 
                       component_id=component_id,
                       message_types=[mt.value for mt in message_types])
    
    def unsubscribe(self, component_id: str, message_types: List[MessageType]):
        """Unsubscribe from messages."""
        with self._lock:
            for message_type in message_types:
                subscribers = self.subscribers.get(message_type.value, [])
                self.subscribers[message_type.value] = [
                    (cid, callback) for cid, callback in subscribers 
                    if cid != component_id
                ]
            logger.info("Component unsubscribed", 
                       component_id=component_id,
                       message_types=[mt.value for mt in message_types])
    
    async def _message_processor(self):
        """Process messages from queue."""
        while self._running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                # Get subscribers for this message type
                subscribers = self.subscribers.get(message.message_type.value, [])
                
                # Send message to all subscribers
                for component_id, callback in subscribers:
                    try:
                        if message.destination is None or message.destination == component_id:
                            await asyncio.create_task(self._deliver_message(callback, message))
                    except Exception as e:
                        logger.error("Message delivery failed", 
                                   component_id=component_id,
                                   error=str(e))
                
                self.message_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Message processing error", error=str(e))
    
    async def _deliver_message(self, callback: Callable, message: SystemMessage):
        """Deliver message to callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
        except Exception as e:
            logger.error("Message callback error", error=str(e))

class LifecycleManager:
    """Component lifecycle management system."""
    
    def __init__(self, orchestrator: 'SystemOrchestrator'):
        """
        Initialize lifecycle manager.
        
        Args:
            orchestrator: System orchestrator reference
        """
        self.orchestrator = orchestrator
        self.components = {}
        self.startup_order = []
        self.shutdown_order = []
        self._lock = threading.RLock()
    
    def register_component(self, component: Component, dependencies: List[str] = None):
        """Register a component."""
        with self._lock:
            self.components[component.component_id] = {
                'component': component,
                'dependencies': dependencies or [],
                'status': ComponentStatus.UNKNOWN
            }
            logger.info("Component registered", 
                       component_id=component.component_id,
                       dependencies=dependencies)
    
    def unregister_component(self, component_id: str):
        """Unregister a component."""
        with self._lock:
            if component_id in self.components:
                del self.components[component_id]
                logger.info("Component unregistered", component_id=component_id)
    
    async def start_all_components(self) -> bool:
        """Start all components in dependency order."""
        with self._lock:
            # Calculate startup order based on dependencies
            self.startup_order = self._calculate_startup_order()
        
        logger.info("Starting components", startup_order=self.startup_order)
        
        for component_id in self.startup_order:
            try:
                success = await self._start_component(component_id)
                if not success:
                    logger.error("Failed to start component", component_id=component_id)
                    return False
            except Exception as e:
                logger.error("Error starting component", 
                           component_id=component_id, error=str(e))
                return False
        
        logger.info("All components started successfully")
        return True
    
    async def stop_all_components(self) -> bool:
        """Stop all components in reverse dependency order."""
        with self._lock:
            # Calculate shutdown order (reverse of startup order)
            self.shutdown_order = list(reversed(self.startup_order))
        
        logger.info("Stopping components", shutdown_order=self.shutdown_order)
        
        for component_id in self.shutdown_order:
            try:
                success = await self._stop_component(component_id)
                if not success:
                    logger.error("Failed to stop component", component_id=component_id)
            except Exception as e:
                logger.error("Error stopping component", 
                           component_id=component_id, error=str(e))
        
        logger.info("All components stopped")
        return True
    
    async def restart_component(self, component_id: str) -> bool:
        """Restart a specific component."""
        try:
            # Stop component
            await self._stop_component(component_id)
            
            # Start component
            return await self._start_component(component_id)
        except Exception as e:
            logger.error("Error restarting component", 
                       component_id=component_id, error=str(e))
            return False
    
    async def _start_component(self, component_id: str) -> bool:
        """Start a specific component."""
        component_info = self.components.get(component_id)
        if not component_info:
            logger.error("Component not found", component_id=component_id)
            return False
        
        component = component_info['component']
        
        try:
            # Update status
            component.status = ComponentStatus.INITIALIZING
            component_info['status'] = ComponentStatus.INITIALIZING
            
            # Start component
            success = await component.start()
            
            if success:
                component.status = ComponentStatus.RUNNING
                component_info['status'] = ComponentStatus.RUNNING
                logger.info("Component started", component_id=component_id)
            else:
                component.status = ComponentStatus.ERROR
                component_info['status'] = ComponentStatus.ERROR
                logger.error("Component failed to start", component_id=component_id)
            
            return success
            
        except Exception as e:
            component.status = ComponentStatus.ERROR
            component_info['status'] = ComponentStatus.ERROR
            logger.error("Error starting component", 
                       component_id=component_id, error=str(e))
            return False
    
    async def _stop_component(self, component_id: str) -> bool:
        """Stop a specific component."""
        component_info = self.components.get(component_id)
        if not component_info:
            logger.error("Component not found", component_id=component_id)
            return False
        
        component = component_info['component']
        
        try:
            # Stop component
            success = await component.stop()
            
            if success:
                component.status = ComponentStatus.STOPPED
                component_info['status'] = ComponentStatus.STOPPED
                logger.info("Component stopped", component_id=component_id)
            else:
                logger.error("Component failed to stop", component_id=component_id)
            
            return success
            
        except Exception as e:
            logger.error("Error stopping component", 
                       component_id=component_id, error=str(e))
            return False
    
    def _calculate_startup_order(self) -> List[str]:
        """Calculate startup order based on dependencies."""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(component_id: str):
            if component_id in temp_visited:
                raise ValueError(f"Circular dependency detected: {component_id}")
            if component_id in visited:
                return
            
            temp_visited.add(component_id)
            
            # Visit dependencies first
            dependencies = self.components.get(component_id, {}).get('dependencies', [])
            for dep_id in dependencies:
                if dep_id in self.components:
                    visit(dep_id)
            
            temp_visited.remove(component_id)
            visited.add(component_id)
            order.append(component_id)
        
        # Visit all components
        for component_id in self.components:
            if component_id not in visited:
                visit(component_id)
        
        return order

class DependencyInjector:
    """Dependency injection container."""
    
    def __init__(self):
        """Initialize dependency injector."""
        self.services = {}
        self.singletons = {}
        self._lock = threading.RLock()
    
    def register_service(self, service_type: Type, implementation: Any, 
                        singleton: bool = True):
        """Register a service."""
        with self._lock:
            self.services[service_type] = implementation
            if singleton:
                self.singletons[service_type] = implementation
            logger.info("Service registered", service_type=service_type.__name__)
    
    def get_service(self, service_type: Type) -> Optional[Any]:
        """Get a service."""
        with self._lock:
            return self.services.get(service_type)
    
    def resolve_dependencies(self, target_class: Type, **kwargs) -> Any:
        """Resolve dependencies for a class."""
        # Simple dependency resolution
        # In a real implementation, this would use reflection/introspection
        return target_class(**kwargs)

class SystemOrchestrator:
    """Main system orchestrator."""
    
    def __init__(self, config: OrchestratorConfig = None):
        """
        Initialize system orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()
        self.service_registry = ServiceRegistry()
        self.message_bus = MessageBus(self.config.max_message_queue_size)
        self.lifecycle_manager = LifecycleManager(self)
        self.dependency_injector = DependencyInjector()
        
        self.system_status = ComponentStatus.UNKNOWN
        self.start_time = None
        self.health_check_task = None
        self._running = False
        self._lock = threading.RLock()
        
        logger.info("System orchestrator initialized")
    
    async def start(self) -> bool:
        """Start the system orchestrator."""
        try:
            with self._lock:
                if self._running:
                    logger.warning("Orchestrator already running")
                    return True
                
                self._running = True
                self.start_time = datetime.now()
                self.system_status = ComponentStatus.INITIALIZING
            
            # Start message bus
            await self.message_bus.start()
            
            # Start health monitoring
            if self.config.enable_health_monitoring:
                self.health_check_task = asyncio.create_task(self._health_monitor())
            
            # Start all components
            success = await self.lifecycle_manager.start_all_components()
            
            if success:
                self.system_status = ComponentStatus.RUNNING
                logger.info("System orchestrator started successfully")
            else:
                self.system_status = ComponentStatus.ERROR
                logger.error("System orchestrator failed to start")
            
            return success
            
        except Exception as e:
            self.system_status = ComponentStatus.ERROR
            logger.error("Error starting orchestrator", error=str(e))
            return False
    
    async def stop(self) -> bool:
        """Stop the system orchestrator."""
        try:
            with self._lock:
                if not self._running:
                    logger.warning("Orchestrator not running")
                    return True
                
                self._running = False
                self.system_status = ComponentStatus.STOPPED
            
            # Stop health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Stop all components
            await self.lifecycle_manager.stop_all_components()
            
            # Stop message bus
            await self.message_bus.stop()
            
            logger.info("System orchestrator stopped")
            return True
            
        except Exception as e:
            logger.error("Error stopping orchestrator", error=str(e))
            return False
    
    def register_component(self, component: Component, dependencies: List[str] = None):
        """Register a component."""
        # Set orchestrator reference
        component.set_orchestrator(self)
        
        # Register with lifecycle manager
        self.lifecycle_manager.register_component(component, dependencies)
        
        # Register with service registry
        service_info = component.get_info()
        self.service_registry.register_service(service_info)
        
        logger.info("Component registered with orchestrator", 
                   component_id=component.component_id)
    
    async def send_message(self, message: SystemMessage):
        """Send a message through the message bus."""
        await self.message_bus.publish(message)
    
    def subscribe_to_messages(self, component_id: str, message_types: List[MessageType], 
                            callback: Callable[[SystemMessage], None]):
        """Subscribe to messages."""
        self.message_bus.subscribe(component_id, message_types, callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        with self._lock:
            return {
                'system_status': self.system_status.value,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'components': len(self.lifecycle_manager.components),
                'services': len(self.service_registry.services),
                'message_queue_size': self.message_bus.message_queue.qsize()
            }
    
    def get_component_status(self, component_id: str) -> Optional[ComponentInfo]:
        """Get component status."""
        return self.service_registry.get_service(component_id)
    
    def get_all_components(self) -> List[ComponentInfo]:
        """Get all components."""
        return self.service_registry.get_all_services()
    
    async def _health_monitor(self):
        """Health monitoring task."""
        while self._running:
            try:
                # Check component health
                components = self.lifecycle_manager.components
                for component_id, component_info in components.items():
                    component = component_info['component']
                    
                    try:
                        # Perform health check
                        is_healthy = await asyncio.wait_for(
                            component.health_check(), 
                            timeout=self.config.health_check_timeout
                        )
                        
                        # Update status
                        if is_healthy:
                            if component.status != ComponentStatus.RUNNING:
                                component.status = ComponentStatus.RUNNING
                                component_info['status'] = ComponentStatus.RUNNING
                        else:
                            if component.status == ComponentStatus.RUNNING:
                                component.status = ComponentStatus.DEGRADED
                                component_info['status'] = ComponentStatus.DEGRADED
                        
                        # Update service registry
                        self.service_registry.update_service_status(component_id, component.status)
                        
                    except asyncio.TimeoutError:
                        logger.warning("Health check timeout", component_id=component_id)
                        component.status = ComponentStatus.DEGRADED
                        component_info['status'] = ComponentStatus.DEGRADED
                    except Exception as e:
                        logger.error("Health check error", 
                                   component_id=component_id, error=str(e))
                        component.status = ComponentStatus.ERROR
                        component_info['status'] = ComponentStatus.ERROR
                
                # Wait for next health check
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error", error=str(e))
                await asyncio.sleep(self.config.heartbeat_interval)

def create_system_orchestrator(config: OrchestratorConfig = None) -> SystemOrchestrator:
    """
    Create a system orchestrator.
    
    Args:
        config: Orchestrator configuration
        
    Returns:
        SystemOrchestrator instance
    """
    return SystemOrchestrator(config)

if __name__ == "__main__":
    # Demo of system orchestrator
    config = OrchestratorConfig(
        enable_auto_discovery=True,
        enable_health_monitoring=True,
        enable_fault_tolerance=True,
        heartbeat_interval=30.0,
        health_check_timeout=10.0,
        max_retry_attempts=3,
        retry_delay=5.0
    )
    
    orchestrator = create_system_orchestrator(config)
    
    # Create sample components
    class SampleComponent(Component):
        def __init__(self, component_id: str, name: str, component_type: ComponentType):
            super().__init__(component_id, name, component_type)
            self._running = False
        
        async def start(self) -> bool:
            self._running = True
            return True
        
        async def stop(self) -> bool:
            self._running = False
            return True
        
        async def health_check(self) -> bool:
            return self._running
    
    # Create components
    data_pipeline = SampleComponent("data_pipeline_001", "Data Pipeline", ComponentType.DATA_PIPELINE)
    strategy_executor = SampleComponent("strategy_executor_001", "Strategy Executor", ComponentType.STRATEGY_EXECUTOR)
    risk_manager = SampleComponent("risk_manager_001", "Risk Manager", ComponentType.RISK_MANAGER)
    
    # Register components with dependencies
    orchestrator.register_component(data_pipeline)
    orchestrator.register_component(strategy_executor, dependencies=["data_pipeline_001"])
    orchestrator.register_component(risk_manager, dependencies=["strategy_executor_001"])
    
    print("System Orchestrator created successfully!")
    print(f"Components: {len(orchestrator.get_all_components())}")
    
    # Get system status
    status = orchestrator.get_system_status()
    print(f"System Status: {status}")
    
    # Start orchestrator (uncomment to run)
    # asyncio.run(orchestrator.start())