"""
Load Balancer for Trading System

This module provides load balancing functionality for distributing requests
across multiple service instances and optimizing performance.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import random
import hashlib
import statistics

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategy enumeration"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"

class ServerStatus(Enum):
    """Server status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"

@dataclass
class Server:
    """Server information"""
    server_id: str
    host: str
    port: int
    weight: int
    max_connections: int
    current_connections: int
    response_time: float
    status: ServerStatus
    last_health_check: datetime
    health_check_interval: int
    failure_count: int
    success_count: int

@dataclass
class LoadBalancerStats:
    """Load balancer statistics"""
    total_requests: int
    requests_per_server: Dict[str, int]
    average_response_time: float
    server_health_status: Dict[str, ServerStatus]
    last_updated: datetime

class LoadBalancer:
    """
    Load balancer for distributing requests across multiple servers.
    """
    
    def __init__(self, name: str, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.name = name
        self.strategy = strategy
        
        # Server management
        self.servers: Dict[str, Server] = {}
        self.server_list: List[str] = []
        self.current_index = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Health checking
        self.health_check_thread = None
        self.running = False
        self.health_check_interval = 30  # seconds
        
        # Statistics
        self.stats = LoadBalancerStats(
            total_requests=0,
            requests_per_server={},
            average_response_time=0.0,
            server_health_status={},
            last_updated=datetime.utcnow()
        )
        
        # Callbacks
        self.server_failure_callbacks: List[Callable] = []
        self.server_recovery_callbacks: List[Callable] = []
        
        # Start health checking
        self.start()
        
        logger.info(f"Load balancer initialized: {name} with strategy {strategy.value}")
    
    def start(self):
        """Start the load balancer"""
        if not self.running:
            self.running = True
            self.health_check_thread = threading.Thread(target=self._health_check_worker, daemon=True)
            self.health_check_thread.start()
            logger.info(f"Load balancer started: {self.name}")
    
    def stop(self):
        """Stop the load balancer"""
        if self.running:
            self.running = False
            if self.health_check_thread:
                self.health_check_thread.join(timeout=5)
            logger.info(f"Load balancer stopped: {self.name}")
    
    def add_server(self, server_id: str, host: str, port: int, weight: int = 1,
                   max_connections: int = 100, health_check_interval: int = 30) -> bool:
        """
        Add a server to the load balancer.
        
        Args:
            server_id: Server identifier
            host: Server hostname or IP
            port: Server port
            weight: Server weight for weighted strategies
            max_connections: Maximum connections for this server
            health_check_interval: Health check interval in seconds
            
        Returns:
            True if server was added successfully
        """
        try:
            with self._lock:
                if server_id in self.servers:
                    logger.warning(f"Server {server_id} already exists in load balancer {self.name}")
                    return False
                
                server = Server(
                    server_id=server_id,
                    host=host,
                    port=port,
                    weight=weight,
                    max_connections=max_connections,
                    current_connections=0,
                    response_time=0.0,
                    status=ServerStatus.HEALTHY,
                    last_health_check=datetime.utcnow(),
                    health_check_interval=health_check_interval,
                    failure_count=0,
                    success_count=0
                )
                
                self.servers[server_id] = server
                self.server_list.append(server_id)
                self.stats.requests_per_server[server_id] = 0
                self.stats.server_health_status[server_id] = ServerStatus.HEALTHY
                
                logger.info(f"Server {server_id} added to load balancer {self.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add server {server_id}: {e}")
            return False
    
    def remove_server(self, server_id: str) -> bool:
        """
        Remove a server from the load balancer.
        
        Args:
            server_id: Server identifier
            
        Returns:
            True if server was removed successfully
        """
        try:
            with self._lock:
                if server_id not in self.servers:
                    logger.warning(f"Server {server_id} not found in load balancer {self.name}")
                    return False
                
                del self.servers[server_id]
                if server_id in self.server_list:
                    self.server_list.remove(server_id)
                
                if server_id in self.stats.requests_per_server:
                    del self.stats.requests_per_server[server_id]
                
                if server_id in self.stats.server_health_status:
                    del self.stats.server_health_status[server_id]
                
                logger.info(f"Server {server_id} removed from load balancer {self.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove server {server_id}: {e}")
            return False
    
    def get_server(self, client_ip: Optional[str] = None, request_id: Optional[str] = None) -> Optional[Server]:
        """
        Get a server based on the load balancing strategy.
        
        Args:
            client_ip: Client IP address (for IP-based strategies)
            request_id: Request ID (for consistent hashing)
            
        Returns:
            Selected server or None if no healthy servers available
        """
        try:
            with self._lock:
                healthy_servers = [s for s in self.servers.values() if s.status == ServerStatus.HEALTHY]
                
                if not healthy_servers:
                    logger.warning(f"No healthy servers available in load balancer {self.name}")
                    return None
                
                if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                    return self._round_robin_select(healthy_servers)
                elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                    return self._least_connections_select(healthy_servers)
                elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                    return self._weighted_round_robin_select(healthy_servers)
                elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                    return self._least_response_time_select(healthy_servers)
                elif self.strategy == LoadBalancingStrategy.IP_HASH:
                    return self._ip_hash_select(healthy_servers, client_ip)
                elif self.strategy == LoadBalancingStrategy.RANDOM:
                    return self._random_select(healthy_servers)
                elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                    return self._consistent_hash_select(healthy_servers, request_id)
                else:
                    return self._round_robin_select(healthy_servers)
                    
        except Exception as e:
            logger.error(f"Error selecting server: {e}")
            return None
    
    def update_server_stats(self, server_id: str, response_time: float, success: bool):
        """
        Update server statistics after a request.
        
        Args:
            server_id: Server identifier
            response_time: Response time in seconds
            success: Whether the request was successful
        """
        try:
            with self._lock:
                if server_id not in self.servers:
                    return
                
                server = self.servers[server_id]
                
                # Update connection count
                if success:
                    server.current_connections = max(0, server.current_connections - 1)
                    server.success_count += 1
                else:
                    server.failure_count += 1
                
                # Update response time (exponential moving average)
                alpha = 0.1  # Smoothing factor
                server.response_time = (alpha * response_time) + ((1 - alpha) * server.response_time)
                
                # Update load balancer stats
                self.stats.total_requests += 1
                self.stats.requests_per_server[server_id] += 1
                self.stats.last_updated = datetime.utcnow()
                
                # Calculate average response time
                response_times = [s.response_time for s in self.servers.values() if s.response_time > 0]
                if response_times:
                    self.stats.average_response_time = statistics.mean(response_times)
                
        except Exception as e:
            logger.error(f"Error updating server stats: {e}")
    
    def increment_connection(self, server_id: str) -> bool:
        """
        Increment connection count for a server.
        
        Args:
            server_id: Server identifier
            
        Returns:
            True if connection was incremented successfully
        """
        try:
            with self._lock:
                if server_id not in self.servers:
                    return False
                
                server = self.servers[server_id]
                if server.current_connections < server.max_connections:
                    server.current_connections += 1
                    return True
                else:
                    logger.warning(f"Server {server_id} at maximum connections")
                    return False
                    
        except Exception as e:
            logger.error(f"Error incrementing connection: {e}")
            return False
    
    def get_stats(self) -> LoadBalancerStats:
        """Get load balancer statistics"""
        with self._lock:
            return LoadBalancerStats(
                total_requests=self.stats.total_requests,
                requests_per_server=self.stats.requests_per_server.copy(),
                average_response_time=self.stats.average_response_time,
                server_health_status=self.stats.server_health_status.copy(),
                last_updated=self.stats.last_updated
            )
    
    def get_servers(self) -> List[Server]:
        """Get all servers"""
        with self._lock:
            return list(self.servers.values())
    
    def add_server_failure_callback(self, callback: Callable):
        """Add callback for server failures"""
        self.server_failure_callbacks.append(callback)
    
    def add_server_recovery_callback(self, callback: Callable):
        """Add callback for server recoveries"""
        self.server_recovery_callbacks.append(callback)
    
    def _round_robin_select(self, servers: List[Server]) -> Server:
        """Round robin server selection"""
        if not servers:
            return None
        
        server = servers[self.current_index % len(servers)]
        self.current_index = (self.current_index + 1) % len(servers)
        return server
    
    def _least_connections_select(self, servers: List[Server]) -> Server:
        """Least connections server selection"""
        if not servers:
            return None
        
        return min(servers, key=lambda s: s.current_connections)
    
    def _weighted_round_robin_select(self, servers: List[Server]) -> Server:
        """Weighted round robin server selection"""
        if not servers:
            return None
        
        # Create weighted list
        weighted_servers = []
        for server in servers:
            weighted_servers.extend([server] * server.weight)
        
        if not weighted_servers:
            return None
        
        server = weighted_servers[self.current_index % len(weighted_servers)]
        self.current_index = (self.current_index + 1) % len(weighted_servers)
        return server
    
    def _least_response_time_select(self, servers: List[Server]) -> Server:
        """Least response time server selection"""
        if not servers:
            return None
        
        # Filter servers with valid response times
        valid_servers = [s for s in servers if s.response_time > 0]
        if not valid_servers:
            return servers[0]  # Return first server if no response time data
        
        return min(valid_servers, key=lambda s: s.response_time)
    
    def _ip_hash_select(self, servers: List[Server], client_ip: Optional[str]) -> Server:
        """IP hash server selection"""
        if not servers:
            return None
        
        if not client_ip:
            return servers[0]  # Fallback to first server
        
        # Hash the IP address
        hash_value = hash(client_ip) % len(servers)
        return servers[hash_value]
    
    def _random_select(self, servers: List[Server]) -> Server:
        """Random server selection"""
        if not servers:
            return None
        
        return random.choice(servers)
    
    def _consistent_hash_select(self, servers: List[Server], request_id: Optional[str]) -> Server:
        """Consistent hash server selection"""
        if not servers:
            return None
        
        if not request_id:
            return servers[0]  # Fallback to first server
        
        # Simple consistent hashing
        hash_value = hash(request_id) % len(servers)
        return servers[hash_value]
    
    def _health_check_worker(self):
        """Health check worker thread"""
        while self.running:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check worker: {e}")
                time.sleep(5)
    
    def _perform_health_checks(self):
        """Perform health checks on all servers"""
        current_time = datetime.utcnow()
        
        for server in self.servers.values():
            try:
                # Check if it's time for health check
                if (current_time - server.last_health_check).total_seconds() < server.health_check_interval:
                    continue
                
                # Perform health check
                is_healthy = self._check_server_health(server)
                old_status = server.status
                
                if is_healthy:
                    if server.status != ServerStatus.HEALTHY:
                        server.status = ServerStatus.HEALTHY
                        self._trigger_server_recovery(server)
                else:
                    if server.status == ServerStatus.HEALTHY:
                        server.status = ServerStatus.UNHEALTHY
                        self._trigger_server_failure(server)
                
                server.last_health_check = current_time
                
                # Update stats
                self.stats.server_health_status[server.server_id] = server.status
                
            except Exception as e:
                logger.error(f"Error checking health for server {server.server_id}: {e}")
    
    def _check_server_health(self, server: Server) -> bool:
        """
        Check if a server is healthy.
        
        Args:
            server: Server to check
            
        Returns:
            True if server is healthy
        """
        try:
            # Simple health check - can be enhanced with actual HTTP/TCP checks
            # For now, we'll use a simple heuristic based on failure rate
            
            total_requests = server.success_count + server.failure_count
            if total_requests == 0:
                return True
            
            failure_rate = server.failure_count / total_requests
            return failure_rate < 0.5  # Consider unhealthy if >50% failure rate
            
        except Exception as e:
            logger.error(f"Error in health check for server {server.server_id}: {e}")
            return False
    
    def _trigger_server_failure(self, server: Server):
        """Trigger server failure callbacks"""
        for callback in self.server_failure_callbacks:
            try:
                callback(server)
            except Exception as e:
                logger.error(f"Error in server failure callback: {e}")
    
    def _trigger_server_recovery(self, server: Server):
        """Trigger server recovery callbacks"""
        for callback in self.server_recovery_callbacks:
            try:
                callback(server)
            except Exception as e:
                logger.error(f"Error in server recovery callback: {e}")


class LoadBalancerManager:
    """
    Manager for multiple load balancers.
    """
    
    def __init__(self):
        self.load_balancers: Dict[str, LoadBalancer] = {}
        self._lock = threading.RLock()
        
        logger.info("Load balancer manager initialized")
    
    def create_load_balancer(self, name: str, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN) -> LoadBalancer:
        """
        Create a new load balancer.
        
        Args:
            name: Load balancer name
            strategy: Load balancing strategy
            
        Returns:
            Created load balancer
        """
        with self._lock:
            if name in self.load_balancers:
                logger.warning(f"Load balancer {name} already exists")
                return self.load_balancers[name]
            
            load_balancer = LoadBalancer(name, strategy)
            self.load_balancers[name] = load_balancer
            
            logger.info(f"Load balancer created: {name}")
            return load_balancer
    
    def get_load_balancer(self, name: str) -> Optional[LoadBalancer]:
        """
        Get a load balancer by name.
        
        Args:
            name: Load balancer name
            
        Returns:
            Load balancer if found
        """
        return self.load_balancers.get(name)
    
    def remove_load_balancer(self, name: str) -> bool:
        """
        Remove a load balancer.
        
        Args:
            name: Load balancer name
            
        Returns:
            True if load balancer was removed
        """
        with self._lock:
            if name in self.load_balancers:
                load_balancer = self.load_balancers[name]
                load_balancer.stop()
                del self.load_balancers[name]
                logger.info(f"Load balancer removed: {name}")
                return True
            return False
    
    def get_all_load_balancers(self) -> Dict[str, LoadBalancer]:
        """Get all load balancers"""
        return self.load_balancers.copy()
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all load balancers.
        
        Returns:
            Load balancer statistics
        """
        stats = {}
        
        for name, lb in self.load_balancers.items():
            stats[name] = asdict(lb.get_stats())
        
        return stats


# Global instance management
_load_balancer_manager_instance = None

def get_load_balancer_manager() -> LoadBalancerManager:
    """Get or create load balancer manager instance"""
    global _load_balancer_manager_instance
    
    if _load_balancer_manager_instance is None:
        _load_balancer_manager_instance = LoadBalancerManager()
    
    return _load_balancer_manager_instance


def init_load_balancer_manager() -> LoadBalancerManager:
    """Initialize load balancer manager"""
    global _load_balancer_manager_instance
    
    _load_balancer_manager_instance = LoadBalancerManager()
    
    return _load_balancer_manager_instance