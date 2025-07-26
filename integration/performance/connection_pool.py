"""
Connection Pool for Trading System

This module provides connection pooling functionality for managing database
and network connections efficiently.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import contextlib
import weakref

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """Connection state enumeration"""
    IDLE = "idle"
    IN_USE = "in_use"
    BROKEN = "broken"
    CREATING = "creating"
    CLOSING = "closing"

class PoolExhaustedError(Exception):
    """Exception raised when connection pool is exhausted"""
    pass

class ConnectionTimeoutError(Exception):
    """Exception raised when connection acquisition times out"""
    pass

@dataclass
class Connection:
    """Connection information"""
    connection_id: str
    created_at: datetime
    last_used: datetime
    state: ConnectionState
    usage_count: int
    error_count: int
    is_valid: bool

@dataclass
class PoolConfig:
    """Connection pool configuration"""
    min_connections: int = 5
    max_connections: int = 20
    acquire_timeout: float = 30.0  # seconds
    connection_timeout: float = 300.0  # seconds
    idle_timeout: float = 600.0  # seconds
    max_usage_count: int = 1000
    health_check_interval: float = 60.0  # seconds
    cleanup_interval: float = 300.0  # seconds

@dataclass
class PoolStats:
    """Connection pool statistics"""
    total_connections: int
    idle_connections: int
    in_use_connections: int
    broken_connections: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    wait_time_avg: float
    connection_time_avg: float

class ConnectionPool:
    """
    Generic connection pool for managing connections efficiently.
    """
    
    def __init__(self, name: str, config: PoolConfig, 
                 connection_factory: Callable, 
                 connection_validator: Optional[Callable] = None,
                 connection_cleanup: Optional[Callable] = None):
        self.name = name
        self.config = config
        self.connection_factory = connection_factory
        self.connection_validator = connection_validator
        self.connection_cleanup = connection_cleanup
        
        # Connection management
        self.connections: Dict[str, Connection] = {}
        self.connection_objects: Dict[str, Any] = {}
        self.idle_queue = queue.Queue()
        self.connection_lock = threading.RLock()
        
        # Statistics
        self.stats = PoolStats(
            total_connections=0,
            idle_connections=0,
            in_use_connections=0,
            broken_connections=0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            wait_time_avg=0.0,
            connection_time_avg=0.0
        )
        
        # Background threads
        self.health_check_thread = None
        self.cleanup_thread = None
        self.running = False
        
        # Initialize pool
        self._initialize_pool()
        self.start()
        
        logger.info(f"Connection pool initialized: {name}")
    
    def start(self):
        """Start the connection pool"""
        if not self.running:
            self.running = True
            
            # Start health check thread
            self.health_check_thread = threading.Thread(target=self._health_check_worker, daemon=True)
            self.health_check_thread.start()
            
            # Start cleanup thread
            self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.cleanup_thread.start()
            
            logger.info(f"Connection pool started: {self.name}")
    
    def stop(self):
        """Stop the connection pool"""
        if self.running:
            self.running = False
            
            # Wait for threads to finish
            if self.health_check_thread:
                self.health_check_thread.join(timeout=5)
            if self.cleanup_thread:
                self.cleanup_thread.join(timeout=5)
            
            # Close all connections
            self._close_all_connections()
            
            logger.info(f"Connection pool stopped: {self.name}")
    
    def get_connection(self, timeout: Optional[float] = None) -> Any:
        """
        Get a connection from the pool.
        
        Args:
            timeout: Timeout for acquiring connection
            
        Returns:
            Connection object
            
        Raises:
            PoolExhaustedError: If pool is exhausted
            ConnectionTimeoutError: If connection acquisition times out
        """
        start_time = time.time()
        timeout = timeout or self.config.acquire_timeout
        
        try:
            # Try to get an idle connection
            try:
                connection_id = self.idle_queue.get(timeout=timeout)
                connection = self.connections.get(connection_id)
                
                if connection and self._validate_connection(connection_id):
                    self._mark_connection_in_use(connection_id)
                    self._update_stats_success(time.time() - start_time)
                    return self.connection_objects[connection_id]
                else:
                    # Connection is invalid, try to get another one
                    if connection:
                        self._mark_connection_broken(connection_id)
                    return self.get_connection(timeout - (time.time() - start_time))
                    
            except queue.Empty:
                # No idle connections available, try to create a new one
                if len(self.connections) < self.config.max_connections:
                    return self._create_and_get_connection(timeout - (time.time() - start_time))
                else:
                    raise PoolExhaustedError(f"Connection pool {self.name} is exhausted")
                    
        except Exception as e:
            self._update_stats_failure(time.time() - start_time)
            raise
    
    def return_connection(self, connection_obj: Any):
        """
        Return a connection to the pool.
        
        Args:
            connection_obj: Connection object to return
        """
        try:
            connection_id = self._find_connection_id(connection_obj)
            if not connection_id:
                logger.warning(f"Connection not found in pool {self.name}")
                return
            
            connection = self.connections.get(connection_id)
            if not connection:
                return
            
            # Validate connection before returning
            if self._validate_connection(connection_id):
                self._mark_connection_idle(connection_id)
            else:
                self._mark_connection_broken(connection_id)
                
        except Exception as e:
            logger.error(f"Error returning connection to pool {self.name}: {e}")
    
    def close_connection(self, connection_obj: Any):
        """
        Close a connection and remove it from the pool.
        
        Args:
            connection_obj: Connection object to close
        """
        try:
            connection_id = self._find_connection_id(connection_obj)
            if not connection_id:
                return
            
            self._close_connection(connection_id)
            
        except Exception as e:
            logger.error(f"Error closing connection in pool {self.name}: {e}")
    
    def get_stats(self) -> PoolStats:
        """Get connection pool statistics"""
        with self.connection_lock:
            return PoolStats(
                total_connections=self.stats.total_connections,
                idle_connections=self.stats.idle_connections,
                in_use_connections=self.stats.in_use_connections,
                broken_connections=self.stats.broken_connections,
                total_requests=self.stats.total_requests,
                successful_requests=self.stats.successful_requests,
                failed_requests=self.stats.failed_requests,
                wait_time_avg=self.stats.wait_time_avg,
                connection_time_avg=self.stats.connection_time_avg
            )
    
    @contextlib.contextmanager
    def connection(self, timeout: Optional[float] = None):
        """
        Context manager for connection usage.
        
        Args:
            timeout: Timeout for acquiring connection
            
        Yields:
            Connection object
        """
        conn = None
        try:
            conn = self.get_connection(timeout)
            yield conn
        except Exception as e:
            if conn:
                self.close_connection(conn)
            raise
        else:
            self.return_connection(conn)
    
    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections"""
        for _ in range(self.config.min_connections):
            try:
                self._create_connection()
            except Exception as e:
                logger.error(f"Failed to create initial connection in pool {self.name}: {e}")
    
    def _create_connection(self) -> str:
        """Create a new connection"""
        try:
            connection_id = self._generate_connection_id()
            
            # Create connection object
            connection_obj = self.connection_factory()
            
            # Create connection info
            connection = Connection(
                connection_id=connection_id,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                state=ConnectionState.IDLE,
                usage_count=0,
                error_count=0,
                is_valid=True
            )
            
            with self.connection_lock:
                self.connections[connection_id] = connection
                self.connection_objects[connection_id] = connection_obj
                self.idle_queue.put(connection_id)
                
                # Update statistics
                self.stats.total_connections += 1
                self.stats.idle_connections += 1
            
            logger.debug(f"Created connection {connection_id} in pool {self.name}")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to create connection in pool {self.name}: {e}")
            raise
    
    def _create_and_get_connection(self, timeout: float) -> Any:
        """Create a new connection and return it"""
        connection_id = self._create_connection()
        
        # Wait for connection to be available
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                connection_id = self.idle_queue.get(timeout=0.1)
                connection = self.connections.get(connection_id)
                
                if connection and self._validate_connection(connection_id):
                    self._mark_connection_in_use(connection_id)
                    return self.connection_objects[connection_id]
                else:
                    if connection:
                        self._mark_connection_broken(connection_id)
                    
            except queue.Empty:
                continue
        
        raise ConnectionTimeoutError(f"Timeout creating connection in pool {self.name}")
    
    def _validate_connection(self, connection_id: str) -> bool:
        """Validate a connection"""
        try:
            if not self.connection_validator:
                return True
            
            connection_obj = self.connection_objects.get(connection_id)
            if not connection_obj:
                return False
            
            return self.connection_validator(connection_obj)
            
        except Exception as e:
            logger.error(f"Error validating connection {connection_id}: {e}")
            return False
    
    def _mark_connection_in_use(self, connection_id: str):
        """Mark a connection as in use"""
        with self.connection_lock:
            connection = self.connections.get(connection_id)
            if connection:
                connection.state = ConnectionState.IN_USE
                connection.last_used = datetime.utcnow()
                connection.usage_count += 1
                
                self.stats.idle_connections -= 1
                self.stats.in_use_connections += 1
    
    def _mark_connection_idle(self, connection_id: str):
        """Mark a connection as idle"""
        with self.connection_lock:
            connection = self.connections.get(connection_id)
            if connection:
                connection.state = ConnectionState.IDLE
                connection.last_used = datetime.utcnow()
                
                self.stats.in_use_connections -= 1
                self.stats.idle_connections += 1
                
                # Add to idle queue
                self.idle_queue.put(connection_id)
    
    def _mark_connection_broken(self, connection_id: str):
        """Mark a connection as broken"""
        with self.connection_lock:
            connection = self.connections.get(connection_id)
            if connection:
                connection.state = ConnectionState.BROKEN
                connection.is_valid = False
                connection.error_count += 1
                
                if connection.state == ConnectionState.IN_USE:
                    self.stats.in_use_connections -= 1
                elif connection.state == ConnectionState.IDLE:
                    self.stats.idle_connections -= 1
                
                self.stats.broken_connections += 1
    
    def _find_connection_id(self, connection_obj: Any) -> Optional[str]:
        """Find connection ID by connection object"""
        for connection_id, obj in self.connection_objects.items():
            if obj is connection_obj:
                return connection_id
        return None
    
    def _close_connection(self, connection_id: str):
        """Close a specific connection"""
        try:
            connection = self.connections.get(connection_id)
            connection_obj = self.connection_objects.get(connection_id)
            
            if connection_obj and self.connection_cleanup:
                self.connection_cleanup(connection_obj)
            
            with self.connection_lock:
                if connection_id in self.connections:
                    del self.connections[connection_id]
                if connection_id in self.connection_objects:
                    del self.connection_objects[connection_id]
                
                # Update statistics
                self.stats.total_connections -= 1
                if connection:
                    if connection.state == ConnectionState.IN_USE:
                        self.stats.in_use_connections -= 1
                    elif connection.state == ConnectionState.IDLE:
                        self.stats.idle_connections -= 1
                    elif connection.state == ConnectionState.BROKEN:
                        self.stats.broken_connections -= 1
            
            logger.debug(f"Closed connection {connection_id} in pool {self.name}")
            
        except Exception as e:
            logger.error(f"Error closing connection {connection_id}: {e}")
    
    def _close_all_connections(self):
        """Close all connections in the pool"""
        connection_ids = list(self.connections.keys())
        for connection_id in connection_ids:
            self._close_connection(connection_id)
    
    def _health_check_worker(self):
        """Health check worker thread"""
        while self.running:
            try:
                self._perform_health_checks()
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check worker: {e}")
                time.sleep(5)
    
    def _cleanup_worker(self):
        """Cleanup worker thread"""
        while self.running:
            try:
                self._perform_cleanup()
                time.sleep(self.config.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                time.sleep(5)
    
    def _perform_health_checks(self):
        """Perform health checks on connections"""
        current_time = datetime.utcnow()
        broken_connections = []
        
        for connection_id, connection in self.connections.items():
            try:
                # Check if connection is too old
                if (current_time - connection.created_at).total_seconds() > self.config.connection_timeout:
                    broken_connections.append(connection_id)
                    continue
                
                # Check if connection has been used too many times
                if connection.usage_count > self.config.max_usage_count:
                    broken_connections.append(connection_id)
                    continue
                
                # Check if connection is idle for too long
                if (connection.state == ConnectionState.IDLE and 
                    (current_time - connection.last_used).total_seconds() > self.config.idle_timeout):
                    broken_connections.append(connection_id)
                    continue
                
                # Validate connection
                if not self._validate_connection(connection_id):
                    broken_connections.append(connection_id)
                    continue
                
            except Exception as e:
                logger.error(f"Error checking health of connection {connection_id}: {e}")
                broken_connections.append(connection_id)
        
        # Close broken connections
        for connection_id in broken_connections:
            self._close_connection(connection_id)
        
        # Ensure minimum connections
        while len(self.connections) < self.config.min_connections:
            try:
                self._create_connection()
            except Exception as e:
                logger.error(f"Failed to create connection during health check: {e}")
                break
    
    def _perform_cleanup(self):
        """Perform cleanup operations"""
        try:
            # Remove excess idle connections
            excess_connections = self.stats.idle_connections - self.config.min_connections
            if excess_connections > 0:
                removed_count = 0
                for connection_id, connection in list(self.connections.items()):
                    if (connection.state == ConnectionState.IDLE and 
                        removed_count < excess_connections):
                        self._close_connection(connection_id)
                        removed_count += 1
                        
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
    
    def _update_stats_success(self, wait_time: float):
        """Update statistics for successful connection acquisition"""
        with self.connection_lock:
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            
            # Update average wait time
            alpha = 0.1
            self.stats.wait_time_avg = (alpha * wait_time) + ((1 - alpha) * self.stats.wait_time_avg)
    
    def _update_stats_failure(self, wait_time: float):
        """Update statistics for failed connection acquisition"""
        with self.connection_lock:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            
            # Update average wait time
            alpha = 0.1
            self.stats.wait_time_avg = (alpha * wait_time) + ((1 - alpha) * self.stats.wait_time_avg)
    
    def _generate_connection_id(self) -> str:
        """Generate a unique connection ID"""
        import hashlib
        return hashlib.md5(f"{self.name}_{time.time()}_{threading.get_ident()}".encode()).hexdigest()


class ConnectionPoolManager:
    """
    Manager for multiple connection pools.
    """
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self._lock = threading.RLock()
        
        logger.info("Connection pool manager initialized")
    
    def create_pool(self, name: str, config: PoolConfig, 
                   connection_factory: Callable,
                   connection_validator: Optional[Callable] = None,
                   connection_cleanup: Optional[Callable] = None) -> ConnectionPool:
        """
        Create a new connection pool.
        
        Args:
            name: Pool name
            config: Pool configuration
            connection_factory: Function to create connections
            connection_validator: Function to validate connections
            connection_cleanup: Function to cleanup connections
            
        Returns:
            Created connection pool
        """
        with self._lock:
            if name in self.pools:
                logger.warning(f"Connection pool {name} already exists")
                return self.pools[name]
            
            pool = ConnectionPool(name, config, connection_factory, connection_validator, connection_cleanup)
            self.pools[name] = pool
            
            logger.info(f"Connection pool created: {name}")
            return pool
    
    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """
        Get a connection pool by name.
        
        Args:
            name: Pool name
            
        Returns:
            Connection pool if found
        """
        return self.pools.get(name)
    
    def remove_pool(self, name: str) -> bool:
        """
        Remove a connection pool.
        
        Args:
            name: Pool name
            
        Returns:
            True if pool was removed
        """
        with self._lock:
            if name in self.pools:
                pool = self.pools[name]
                pool.stop()
                del self.pools[name]
                logger.info(f"Connection pool removed: {name}")
                return True
            return False
    
    def get_all_pools(self) -> Dict[str, ConnectionPool]:
        """Get all connection pools"""
        return self.pools.copy()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all connection pools.
        
        Returns:
            Connection pool statistics
        """
        stats = {}
        
        for name, pool in self.pools.items():
            stats[name] = asdict(pool.get_stats())
        
        return stats


# Global instance management
_connection_pool_manager_instance = None

def get_connection_pool_manager() -> ConnectionPoolManager:
    """Get or create connection pool manager instance"""
    global _connection_pool_manager_instance
    
    if _connection_pool_manager_instance is None:
        _connection_pool_manager_instance = ConnectionPoolManager()
    
    return _connection_pool_manager_instance


def init_connection_pool_manager() -> ConnectionPoolManager:
    """Initialize connection pool manager"""
    global _connection_pool_manager_instance
    
    _connection_pool_manager_instance = ConnectionPoolManager()
    
    return _connection_pool_manager_instance