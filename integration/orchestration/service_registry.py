"""
Service Registry for System Orchestration

This module provides service discovery and registration functionality for the trading system,
allowing components to register themselves and discover other services dynamically.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"

class ServiceType(Enum):
    """Service type enumeration"""
    DATA_INGESTION = "data_ingestion"
    FEATURE_ENGINEERING = "feature_engineering"
    ML_MODEL = "ml_model"
    BACKTESTING = "backtesting"
    API_GATEWAY = "api_gateway"
    WEBSOCKET = "websocket"
    SECURITY = "security"
    MONITORING = "monitoring"
    ORCHESTRATION = "orchestration"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"

@dataclass
class ServiceInfo:
    """Service information structure"""
    service_id: str
    service_name: str
    service_type: ServiceType
    version: str
    host: str
    port: int
    status: ServiceStatus
    health_check_url: Optional[str]
    metadata: Dict[str, Any]
    registered_at: datetime
    last_heartbeat: datetime
    dependencies: List[str]
    endpoints: List[str]

@dataclass
class ServiceHealth:
    """Service health information"""
    service_id: str
    status: ServiceStatus
    response_time: float
    error_count: int
    last_check: datetime
    details: Dict[str, Any]

class ServiceRegistry:
    """
    Service registry for managing service discovery and registration.
    """
    
    def __init__(self, registry_name: str = "trading_system_registry"):
        self.registry_name = registry_name
        self.services: Dict[str, ServiceInfo] = {}
        self.health_checks: Dict[str, ServiceHealth] = {}
        self.service_types: Dict[ServiceType, List[str]] = {}
        self.dependencies: Dict[str, List[str]] = {}
        
        # Registry metadata
        self.created_at = datetime.utcnow()
        self.last_cleanup = datetime.utcnow()
        
        # Configuration
        self.heartbeat_timeout = 30  # seconds
        self.cleanup_interval = 60  # seconds
        self.max_services = 1000
        
        # Threading
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._running = False
        
        # Event callbacks
        self._service_callbacks: Dict[str, List[Callable]] = {
            'registered': [],
            'unregistered': [],
            'status_changed': [],
            'health_changed': []
        }
        
        logger.info(f"Service registry '{registry_name}' initialized")
    
    def start(self):
        """Start the service registry"""
        with self._lock:
            if not self._running:
                self._running = True
                self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
                self._cleanup_thread.start()
                logger.info("Service registry started")
    
    def stop(self):
        """Stop the service registry"""
        with self._lock:
            if self._running:
                self._running = False
                if self._cleanup_thread:
                    self._cleanup_thread.join(timeout=5)
                logger.info("Service registry stopped")
    
    def register_service(self, service_name: str, service_type: ServiceType, 
                        host: str, port: int, version: str = "1.0.0",
                        health_check_url: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        dependencies: Optional[List[str]] = None,
                        endpoints: Optional[List[str]] = None) -> str:
        """
        Register a new service in the registry.
        
        Args:
            service_name: Name of the service
            service_type: Type of service
            host: Host address
            port: Port number
            version: Service version
            health_check_url: URL for health checks
            metadata: Additional service metadata
            dependencies: List of service dependencies
            endpoints: List of service endpoints
            
        Returns:
            Service ID
        """
        with self._lock:
            # Check if service already exists
            existing_service = self._find_service_by_name_and_host(service_name, host, port)
            if existing_service:
                logger.warning(f"Service {service_name} already registered at {host}:{port}")
                return existing_service.service_id
            
            # Create service ID
            service_id = str(uuid.uuid4())
            
            # Create service info
            service_info = ServiceInfo(
                service_id=service_id,
                service_name=service_name,
                service_type=service_type,
                version=version,
                host=host,
                port=port,
                status=ServiceStatus.STARTING,
                health_check_url=health_check_url,
                metadata=metadata or {},
                registered_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                dependencies=dependencies or [],
                endpoints=endpoints or []
            )
            
            # Register service
            self.services[service_id] = service_info
            
            # Update service types index
            if service_type not in self.service_types:
                self.service_types[service_type] = []
            self.service_types[service_type].append(service_id)
            
            # Update dependencies
            for dep in service_info.dependencies:
                if dep not in self.dependencies:
                    self.dependencies[dep] = []
                self.dependencies[dep].append(service_id)
            
            # Initialize health check
            self.health_checks[service_id] = ServiceHealth(
                service_id=service_id,
                status=ServiceStatus.UNKNOWN,
                response_time=0.0,
                error_count=0,
                last_check=datetime.utcnow(),
                details={}
            )
            
            # Trigger callbacks
            self._trigger_callbacks('registered', service_info)
            
            logger.info(f"Service registered: {service_name} ({service_id}) at {host}:{port}")
            return service_id
    
    def unregister_service(self, service_id: str) -> bool:
        """
        Unregister a service from the registry.
        
        Args:
            service_id: Service ID to unregister
            
        Returns:
            True if service was unregistered, False otherwise
        """
        with self._lock:
            if service_id not in self.services:
                return False
            
            service_info = self.services[service_id]
            
            # Remove from services
            del self.services[service_id]
            
            # Remove from service types
            if service_info.service_type in self.service_types:
                self.service_types[service_info.service_type] = [
                    sid for sid in self.service_types[service_info.service_type] 
                    if sid != service_id
                ]
            
            # Remove from dependencies
            for dep in service_info.dependencies:
                if dep in self.dependencies:
                    self.dependencies[dep] = [
                        sid for sid in self.dependencies[dep] 
                        if sid != service_id
                    ]
            
            # Remove health check
            if service_id in self.health_checks:
                del self.health_checks[service_id]
            
            # Trigger callbacks
            self._trigger_callbacks('unregistered', service_info)
            
            logger.info(f"Service unregistered: {service_info.service_name} ({service_id})")
            return True
    
    def update_service_status(self, service_id: str, status: ServiceStatus) -> bool:
        """
        Update service status.
        
        Args:
            service_id: Service ID
            status: New status
            
        Returns:
            True if status was updated, False otherwise
        """
        with self._lock:
            if service_id not in self.services:
                return False
            
            old_status = self.services[service_id].status
            self.services[service_id].status = status
            
            # Update health check
            if service_id in self.health_checks:
                self.health_checks[service_id].status = status
                self.health_checks[service_id].last_check = datetime.utcnow()
            
            # Trigger callbacks if status changed
            if old_status != status:
                self._trigger_callbacks('status_changed', self.services[service_id])
            
            logger.info(f"Service status updated: {service_id} -> {status.value}")
            return True
    
    def heartbeat(self, service_id: str) -> bool:
        """
        Update service heartbeat.
        
        Args:
            service_id: Service ID
            
        Returns:
            True if heartbeat was updated, False otherwise
        """
        with self._lock:
            if service_id not in self.services:
                return False
            
            self.services[service_id].last_heartbeat = datetime.utcnow()
            return True
    
    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """
        Get service information by ID.
        
        Args:
            service_id: Service ID
            
        Returns:
            Service information or None
        """
        with self._lock:
            return self.services.get(service_id)
    
    def get_services_by_type(self, service_type: ServiceType) -> List[ServiceInfo]:
        """
        Get all services of a specific type.
        
        Args:
            service_type: Service type
            
        Returns:
            List of service information
        """
        with self._lock:
            service_ids = self.service_types.get(service_type, [])
            return [self.services[sid] for sid in service_ids if sid in self.services]
    
    def get_services_by_name(self, service_name: str) -> List[ServiceInfo]:
        """
        Get all services with a specific name.
        
        Args:
            service_name: Service name
            
        Returns:
            List of service information
        """
        with self._lock:
            return [
                service for service in self.services.values()
                if service.service_name == service_name
            ]
    
    def get_healthy_services(self, service_type: Optional[ServiceType] = None) -> List[ServiceInfo]:
        """
        Get all healthy services, optionally filtered by type.
        
        Args:
            service_type: Optional service type filter
            
        Returns:
            List of healthy service information
        """
        with self._lock:
            services = self.services.values()
            
            if service_type:
                services = [s for s in services if s.service_type == service_type]
            
            # Filter by health status
            healthy_services = []
            for service in services:
                health = self.health_checks.get(service.service_id)
                if health and health.status == ServiceStatus.RUNNING:
                    healthy_services.append(service)
            
            return healthy_services
    
    def get_service_dependencies(self, service_id: str) -> List[ServiceInfo]:
        """
        Get all services that depend on the specified service.
        
        Args:
            service_id: Service ID
            
        Returns:
            List of dependent service information
        """
        with self._lock:
            dependent_ids = self.dependencies.get(service_id, [])
            return [self.services[sid] for sid in dependent_ids if sid in self.services]
    
    def check_service_health(self, service_id: str) -> Optional[ServiceHealth]:
        """
        Get service health information.
        
        Args:
            service_id: Service ID
            
        Returns:
            Service health information or None
        """
        with self._lock:
            return self.health_checks.get(service_id)
    
    def update_service_health(self, service_id: str, health: ServiceHealth) -> bool:
        """
        Update service health information.
        
        Args:
            service_id: Service ID
            health: Health information
            
        Returns:
            True if health was updated, False otherwise
        """
        with self._lock:
            if service_id not in self.services:
                return False
            
            old_health = self.health_checks.get(service_id)
            self.health_checks[service_id] = health
            
            # Trigger callbacks if health changed
            if old_health and old_health.status != health.status:
                self._trigger_callbacks('health_changed', self.services[service_id])
            
            return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Registry statistics
        """
        with self._lock:
            total_services = len(self.services)
            service_type_counts = {
                service_type.value: len(services)
                for service_type, services in self.service_types.items()
            }
            
            status_counts = {}
            for service in self.services.values():
                status = service.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'total_services': total_services,
                'service_type_counts': service_type_counts,
                'status_counts': status_counts,
                'created_at': self.created_at.isoformat(),
                'last_cleanup': self.last_cleanup.isoformat(),
                'uptime': (datetime.utcnow() - self.created_at).total_seconds()
            }
    
    def add_callback(self, event_type: str, callback: Callable):
        """
        Add event callback.
        
        Args:
            event_type: Event type ('registered', 'unregistered', 'status_changed', 'health_changed')
            callback: Callback function
        """
        if event_type in self._service_callbacks:
            self._service_callbacks[event_type].append(callback)
    
    def _trigger_callbacks(self, event_type: str, service_info: ServiceInfo):
        """Trigger event callbacks"""
        for callback in self._service_callbacks.get(event_type, []):
            try:
                callback(service_info)
            except Exception as e:
                logger.error(f"Error in service callback: {e}")
    
    def _find_service_by_name_and_host(self, service_name: str, host: str, port: int) -> Optional[ServiceInfo]:
        """Find service by name, host, and port"""
        for service in self.services.values():
            if (service.service_name == service_name and 
                service.host == host and 
                service.port == port):
                return service
        return None
    
    def _cleanup_worker(self):
        """Background worker for cleaning up stale services"""
        while self._running:
            try:
                self._cleanup_stale_services()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    def _cleanup_stale_services(self):
        """Remove services that haven't sent heartbeats"""
        with self._lock:
            current_time = datetime.utcnow()
            stale_services = []
            
            for service_id, service_info in self.services.items():
                time_since_heartbeat = (current_time - service_info.last_heartbeat).total_seconds()
                if time_since_heartbeat > self.heartbeat_timeout:
                    stale_services.append(service_id)
            
            for service_id in stale_services:
                logger.warning(f"Removing stale service: {service_id}")
                self.unregister_service(service_id)
            
            if stale_services:
                self.last_cleanup = current_time
                logger.info(f"Cleaned up {len(stale_services)} stale services")


# Global instance management
_service_registry_instance = None

def get_service_registry() -> ServiceRegistry:
    """Get or create service registry instance"""
    global _service_registry_instance
    
    if _service_registry_instance is None:
        _service_registry_instance = ServiceRegistry()
        _service_registry_instance.start()
    
    return _service_registry_instance


def init_service_registry(registry_name: str = "trading_system_registry") -> ServiceRegistry:
    """Initialize service registry with custom name"""
    global _service_registry_instance
    
    if _service_registry_instance:
        _service_registry_instance.stop()
    
    _service_registry_instance = ServiceRegistry(registry_name)
    _service_registry_instance.start()
    
    return _service_registry_instance